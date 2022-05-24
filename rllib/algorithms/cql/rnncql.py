from typing import Type, Optional

from ray.rllib.algorithms.sac.rnnsac_torch_policy import RNNSACTorchPolicy
from ray.rllib.algorithms.cql.cql import CQLConfig
from ray.rllib.algorithms.sac.rnnsac import RNNSACConfig, RNNSACTrainer
from ray.rllib.algorithms.sac.sac import (
    SACConfig,
)
from ray.rllib.execution.train_ops import (
    multi_gpu_train_one_step,
    train_one_step,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
    DEPRECATED_VALUE,
    deprecation_warning,
    Deprecated,
)
from ray.rllib.utils.metrics import (
    LAST_TARGET_UPDATE_TS,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_TRAINED,
    NUM_TARGET_UPDATES,
    TARGET_NET_UPDATE_TIMER,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.replay_buffers.utils import update_priorities_in_replay_buffer
from ray.rllib.utils.typing import ResultDict, TrainerConfigDict



class RNNCQLConfig(RNNSACConfig, CQLConfig):
    """Defines a configuration class from which an RNNSACTrainer can be built.

    Example:
        >>> config = RNNCQLConfig().training(gamma=0.9, lr=0.01)\
        ...     .resources(num_gpus=0)\
        ...     .rollouts(num_rollout_workers=4)
        >>> print(config.to_dict())
        >>> # Build a Trainer object from the config and run 1 training iteration.
        >>> trainer = config.build(env="CartPole-v1")
        >>> trainer.train()
    """

    def __init__(self, trainer_class=None):
        RNNSACConfig.__init__(self, trainer_class=trainer_class or RNNSACTrainer)
        CQLConfig.__init__(self, trainer_class=trainer_class or RNNSACTrainer)


    @override(SACConfig)
    def training(
        self,
        *,
        zero_init_states: Optional[bool] = None,
        **kwargs,
    ) -> "RNNSACConfig":
        """Sets the training related configuration.

        Args:
            zero_init_states: If True, assume a zero-initialized state input (no matter
                where in the episode the sequence is located).
                If False, store the initial states along with each SampleBatch, use
                it (as initial state when running through the network for training),
                and update that initial state during training (from the internal
                state outputs of the immediately preceding sequence).

        Returns:
            This updated TrainerConfig object.
        """
        RNNSACConfig.training(**kwargs)
        CQLConfig.training(**kwargs)

        return self


class RNNCQLTrainer(RNNSACTrainer):
    @classmethod
    @override(RNNSACTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return RNNCQLConfig().to_dict()

    @override(RNNSACTrainer)
    def validate_config(self, config: TrainerConfigDict) -> None:
        # First check, whether old `timesteps_per_iteration` is used. If so
        # convert right away as for CQL, we must measure in training timesteps,
        # never sampling timesteps (CQL does not sample).
        if config.get("timesteps_per_iteration", DEPRECATED_VALUE) != DEPRECATED_VALUE:
            deprecation_warning(
                old="timesteps_per_iteration",
                new="min_train_timesteps_per_reporting",
                error=False,
            )
            config["min_train_timesteps_per_reporting"] = config[
                "timesteps_per_iteration"
            ]
            config["timesteps_per_iteration"] = DEPRECATED_VALUE

        # Call super's validation method.
        super().validate_config(config)

        if config["num_gpus"] > 1:
            raise ValueError("`num_gpus` > 1 not yet supported for CQL!")

        # CQL-torch performs the optimizer steps inside the loss function.
        # Using the multi-GPU optimizer will therefore not work (see multi-GPU
        # check above) and we must use the simple optimizer for now.
        if config["simple_optimizer"] is not True and config["framework"] == "torch":
            config["simple_optimizer"] = True


    @override(RNNSACTrainer)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        return RNNSACTorchPolicy

    @override(RNNSACTrainer)
    def training_iteration(self) -> ResultDict:

        # Sample training batch from replay buffer.
        train_batch = self.local_replay_buffer.sample(self.config["train_batch_size"])

        # Old-style replay buffers return None if learning has not started
        if not train_batch:
            return {}

        # Postprocess batch before we learn on it.
        post_fn = self.config.get("before_learn_on_batch") or (lambda b, *a: b)
        train_batch = post_fn(train_batch, self.workers, self.config)

        # Learn on training batch.
        # Use simple optimizer (only for multi-agent or tf-eager; all other
        # cases should use the multi-GPU optimizer, even if only using 1 GPU)
        if self.config.get("simple_optimizer") is True:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        # Update replay buffer priorities.
        update_priorities_in_replay_buffer(
            self.local_replay_buffer,
            self.config,
            train_batch,
            train_results,
        )

        # Update target network every `target_network_update_freq` training steps.
        cur_ts = self._counters[
            NUM_AGENT_STEPS_TRAINED if self._by_agent_steps else NUM_ENV_STEPS_TRAINED
        ]
        last_update = self._counters[LAST_TARGET_UPDATE_TS]
        if cur_ts - last_update >= self.config["target_network_update_freq"]:
            with self._timers[TARGET_NET_UPDATE_TIMER]:
                to_update = self.workers.local_worker().get_policies_to_train()
                self.workers.local_worker().foreach_policy_to_train(
                    lambda p, pid: pid in to_update and p.update_target()
                )
            self._counters[NUM_TARGET_UPDATES] += 1
            self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

        # Update remote workers's weights after learning on local worker
        if self.workers.remote_workers():
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights()

        # Return all collected metrics for the iteration.
        return train_results


class _deprecated_default_config(dict):
    def __init__(self):
        super().__init__(RNNCQLConfig().to_dict())

    @Deprecated(
        old="ray.rllib.algorithms.sac.rnncql.DEFAULT_CONFIG",
        new="ray.rllib.algorithms.sac.rnncql.RNNSACConfig(...)",
        error=False,
    )
    def __getitem__(self, item):
        return super().__getitem__(item)


DEFAULT_CONFIG = _deprecated_default_config()
