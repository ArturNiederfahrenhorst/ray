import numpy as np
import tree
import unittest

import ray
import ray.rllib.algorithms.ppo as ppo

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, LEARNER_STATS_KEY
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import (
    get_shared_encoder_config,
    get_separate_encoder_config,
    PPOTorchRLModule,
    get_ppo_loss,
)
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.test_utils import (
    check,
    check_compute_single_action,
    check_train_results,
    framework_iterator,
)


class MyCallbacks(DefaultCallbacks):
    @staticmethod
    def _check_lr_torch(policy, policy_id):
        for j, opt in enumerate(policy._optimizers):
            for p in opt.param_groups:
                assert p["lr"] == policy.cur_lr, "LR scheduling error!"

    @staticmethod
    def _check_lr_tf(policy, policy_id):
        lr = policy.cur_lr
        sess = policy.get_session()
        if sess:
            lr = sess.run(lr)
            optim_lr = sess.run(policy._optimizer._lr)
        else:
            lr = lr.numpy()
            optim_lr = policy._optimizer.lr.numpy()
        assert lr == optim_lr, "LR scheduling error!"

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        stats = result["info"][LEARNER_INFO][DEFAULT_POLICY_ID][LEARNER_STATS_KEY]
        # Learning rate should go to 0 after 1 iter.
        check(stats["cur_lr"], 5e-5 if algorithm.iteration == 1 else 0.0)
        # Entropy coeff goes to 0.05, then 0.0 (per iter).
        check(stats["entropy_coeff"], 0.1 if algorithm.iteration == 1 else 0.05)

        algorithm.workers.foreach_policy(
            self._check_lr_torch
            if algorithm.config.framework_str == "torch"
            else self._check_lr_tf
        )


class TestPPO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init()

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_ppo_compilation_and_schedule_mixins(self):
        """Test whether PPO can be built with all frameworks."""

        # Build a PPOConfig object.
        config = (
            ppo.PPOConfig()
            .training(
                num_sgd_iter=2,
                # Setup lr schedule for testing.
                lr_schedule=[[0, 5e-5], [128, 0.0]],
                # Set entropy_coeff to a faulty value to proof that it'll get
                # overridden by the schedule below (which is expected).
                entropy_coeff=100.0,
                entropy_coeff_schedule=[[0, 0.1], [256, 0.0]],
                train_batch_size=128,
                model=dict(
                    # Settings in case we use an LSTM.
                    lstm_cell_size=10,
                    max_seq_len=20,
                ),
            )
            .rollouts(
                num_rollout_workers=1,
                # Test with compression.
                compress_observations=True,
                enable_connectors=True,
            )
            .callbacks(MyCallbacks)
            .rl_module(_enable_rl_module_api=True)
        )  # For checking lr-schedule correctness.

        num_iterations = 2

        # TODO (Kourosh): for now just do torch
        for fw in framework_iterator(
            config, frameworks=("torch"), with_eager_tracing=True
        ):
            # TODO (Kourosh) Bring back "FrozenLake-v1" and "MsPacmanNoFrameskip-v4"
            for env in ["CartPole-v1", "Pendulum-v1"]:
                print("Env={}".format(env))
                # TODO (Kourosh): for now just do lstm=False
                for lstm in [False]:
                    print("LSTM={}".format(lstm))
                    config.training(
                        model=dict(
                            use_lstm=lstm,
                            lstm_use_prev_action=lstm,
                            lstm_use_prev_reward=lstm,
                            vf_share_layers=lstm,
                        )
                    )

                    algo = config.build(env=env)
                    policy = algo.get_policy()
                    entropy_coeff = algo.get_policy().entropy_coeff
                    lr = policy.cur_lr
                    if fw == "tf":
                        entropy_coeff, lr = policy.get_session().run(
                            [entropy_coeff, lr]
                        )
                    check(entropy_coeff, 0.1)
                    check(lr, config.lr)

                    for i in range(num_iterations):
                        results = algo.train()
                        check_train_results(results)
                        print(results)

                    check_compute_single_action(
                        algo, include_prev_action_reward=True, include_state=lstm
                    )
                    algo.stop()

    def test_ppo_exploration_setup(self):
        """Tests, whether PPO runs with different exploration setups."""
        config = (
            ppo.PPOConfig()
            .environment(
                "FrozenLake-v1",
                env_config={"is_slippery": False, "map_name": "4x4"},
            )
            .rollouts(
                # Run locally.
                num_rollout_workers=0,
                enable_connectors=True,
            )
            .rl_module(_enable_rl_module_api=True)
        )
        obs = np.array(0)

        # TODO (Kourosh) Test against all frameworks.
        for fw in framework_iterator(config, frameworks=("torch")):
            # Default Agent should be setup with StochasticSampling.
            trainer = config.build()
            # explore=False, always expect the same (deterministic) action.
            a_ = trainer.compute_single_action(
                obs, explore=False, prev_action=np.array(2), prev_reward=np.array(1.0)
            )

            for _ in range(50):
                a = trainer.compute_single_action(
                    obs,
                    explore=False,
                    prev_action=np.array(2),
                    prev_reward=np.array(1.0),
                )
                check(a, a_)

            # With explore=True (default), expect stochastic actions.
            actions = []
            for _ in range(300):
                actions.append(
                    trainer.compute_single_action(
                        obs, prev_action=np.array(2), prev_reward=np.array(1.0)
                    )
                )
            check(np.mean(actions), 1.5, atol=0.2)
            trainer.stop()

    def test_torch_model_creation(self):
        pass

    def test_torch_model_creation_lstm(self):
        pass

    def test_rollouts(self):
        for env_name in ["CartPole-v1", "Pendulum-v1"]:
            for fwd_fn in ["forward_exploration", "forward_inference"]:
                for shared_encoder in [False, True]:
                    print(
                        f"[ENV={env_name}] | [FWD={fwd_fn}] | [SHARED={shared_encoder}]"
                    )
                    import gym

                    env = gym.make(env_name)

                    if shared_encoder:
                        config = get_shared_encoder_config(env)
                    else:
                        config = get_separate_encoder_config(env)
                    module = PPOTorchRLModule(config)

                    obs = env.reset()
                    tstep = 0
                    while tstep < 10:

                        if fwd_fn == "forward_exploration":
                            fwd_out = module.forward_exploration(
                                {SampleBatch.OBS: convert_to_torch_tensor(obs)[None]}
                            )
                            action = convert_to_numpy(
                                fwd_out[SampleBatch.ACTION_DIST].sample().squeeze(0)
                            )
                        elif fwd_fn == "forward_inference":
                            # check if I sample twice, I get the same action
                            fwd_out = module.forward_inference(
                                {SampleBatch.OBS: convert_to_torch_tensor(obs)[None]}
                            )
                            action = convert_to_numpy(
                                fwd_out[SampleBatch.ACTION_DIST].sample().squeeze(0)
                            )
                            action2 = convert_to_numpy(
                                fwd_out[SampleBatch.ACTION_DIST].sample().squeeze(0)
                            )
                            check(action, action2)

                        obs, reward, done, info = env.step(action)
                        print(
                            f"obs: {obs}, action: {action}, reward: {reward}, "
                            f"done: {done}, info: {info}"
                        )
                        tstep += 1

    def test_forward_train(self):
        for env_name in ["CartPole-v1", "Pendulum-v1"]:
            for shared_encoder in [False, True]:
                print("-" * 80)
                print(f"[ENV={env_name}] | [SHARED={shared_encoder}]")
                import gym

                env = gym.make(env_name)

                if shared_encoder:
                    config = get_shared_encoder_config(env)
                else:
                    config = get_separate_encoder_config(env)

                module = PPOTorchRLModule(config)

                # collect a batch of data
                batch = []
                obs = env.reset()
                tstep = 0
                while tstep < 10:
                    fwd_out = module.forward_exploration(
                        {"obs": convert_to_torch_tensor(obs)[None]}
                    )
                    action = convert_to_numpy(
                        fwd_out["action_dist"].sample().squeeze(0)
                    )
                    new_obs, reward, done, _ = env.step(action)
                    batch.append(
                        {
                            SampleBatch.OBS: obs,
                            SampleBatch.NEXT_OBS: new_obs,
                            SampleBatch.ACTIONS: action,
                            SampleBatch.REWARDS: np.array(reward),
                            SampleBatch.DONES: np.array(done),
                        }
                    )
                    obs = new_obs
                    tstep += 1

                # convert the list of dicts to dict of lists
                batch = tree.map_structure(lambda *x: list(x), *batch)
                # convert dict of lists to dict of tensors
                fwd_in = {
                    k: convert_to_torch_tensor(np.array(v)) for k, v in batch.items()
                }

                # forward train
                # before training make sure it's on the right device and it's on
                # trianing mode
                module.to("cpu")
                module.train()
                fwd_out = module.forward_train(fwd_in)
                loss = get_ppo_loss(fwd_in, fwd_out)
                loss.backward()

                # check that all neural net parameters have gradients
                for param in module.parameters():
                    self.assertIsNotNone(param.grad)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
