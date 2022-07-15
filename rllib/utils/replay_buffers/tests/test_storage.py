import unittest

import numpy as np

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.replay_buffers.storage import InMemoryStorage, OnDiskStorage
from ray.rllib.utils.test_utils import check


class TestStorage(unittest.TestCase):
    batch_id = 0

    def _add_data_to_storage(self, _storage, batch_size, num_batches=5, **kwargs):
        def _generate_data():
            return SampleBatch(
                {
                    SampleBatch.T: [np.random.random((4,))],
                    SampleBatch.ACTIONS: [np.random.choice([0, 1])],
                    SampleBatch.OBS: [np.random.random((4,))],
                    SampleBatch.NEXT_OBS: [np.random.random((4,))],
                    SampleBatch.REWARDS: [np.random.rand()],
                    SampleBatch.DONES: [np.random.choice([False, True])],
                    "batch_id": [self.batch_id],
                }
            )

        for _ in range(num_batches):
            data = [_generate_data() for _ in range(batch_size)]
            self.batch_id += 1
            batch = SampleBatch.concat_samples(data)
            _storage.add(batch, **kwargs)

    def test_stats(self):
        """Tests stats by adding and sampling few samples and checking the
        values of the storage's stats.
        """
        self.batch_id = 0

        batch_size = 5
        buffer_size = 15

        storage = InMemoryStorage(capacity_ts=buffer_size)

        # Test add/sample
        self._add_data_to_storage(storage, batch_size=batch_size, num_batches=1)

        # After adding a single batch to a storage, it should not be full
        assert len(storage) == 1
        assert storage.num_timesteps_added == 5
        assert storage.num_timesteps == 5
        assert storage._oldest_item_idx == 0
        assert storage.eviction_started is False

        # Retrieving the item should yield the first batch
        assert storage[0]["batch_id"][0] == 0

        self._add_data_to_storage(storage, batch_size=batch_size, num_batches=2)

        # After adding two more batches, the storage should be full,
        # but eviction has not started yet
        assert len(storage) == 3
        assert storage.num_timesteps_added == 15
        assert storage.num_timesteps == 15
        assert storage._oldest_item_idx == 0
        assert storage.eviction_started is False

        self._add_data_to_storage(storage, batch_size=batch_size, num_batches=1)

        # After adding one more batch, eviction has started
        assert len(storage) == 3
        assert storage.num_timesteps_added == 20
        assert storage.num_timesteps == 15
        assert storage._oldest_item_idx == 1
        assert storage.eviction_started is True

    def test_add_and_set(self):
        """Tests adding, sampling, get-/set state, and eviction with
        experiences stored by timesteps.
        """
        self.batch_id = 0

        batch_size = 5
        buffer_size = 15

        storage = InMemoryStorage(capacity_ts=buffer_size)

        # Test add
        self._add_data_to_storage(storage, batch_size=batch_size, num_batches=2)
        assert len(storage) == 2
        assert storage.num_timesteps_added == 2 * 5
        assert storage.num_timesteps == 2 * 5
        assert storage._oldest_item_idx == 0
        assert storage.eviction_started is False

        self._add_data_to_storage(storage, batch_size=batch_size, num_batches=1)
        assert len(storage) == 3
        assert storage.num_timesteps_added == 3 * 5
        assert storage.num_timesteps == 3 * 5
        assert storage._oldest_item_idx == 0
        assert storage.eviction_started is False

        self._add_data_to_storage(storage, batch_size=batch_size, num_batches=1)
        assert len(storage) == 3
        assert storage.num_timesteps_added == 4 * 5
        assert storage.num_timesteps == 3 * 5
        assert storage._oldest_item_idx == 1
        assert storage.eviction_started is True

        # Test set/get item
        b_0 = storage[0]
        b_2 = storage[2]
        storage[2] = b_0
        assert id(b_0) == id(storage[0])
        assert id(b_0) != id(storage[1])
        assert id(b_0) == id(storage[2])
        assert id(b_2) != id(storage[2])

        # Test set/get state
        state = storage.get_state()
        other_storage = InMemoryStorage(capacity_ts=buffer_size)
        self._add_data_to_storage(other_storage, 1)
        other_storage.set_state(state)

        assert other_storage._samples == storage._samples
        assert other_storage._oldest_item_idx == storage._oldest_item_idx
        assert other_storage._num_timesteps_added == storage._num_timesteps_added
        assert other_storage._num_timesteps == storage._num_timesteps
        assert other_storage.eviction_started == storage.eviction_started
        assert other_storage.size_bytes == storage.size_bytes
        assert len(other_storage) == len(storage)

    def test_index_conversion(self):
        """Test conversion between external and internal storage indices."""
        storage = InMemoryStorage(capacity_ts=43)

        for i in range(len(storage)):
            i_idx = storage._get_internal_index(i)
            e_idx = storage._get_external_index(i_idx)
            assert (i % len(storage)) == e_idx

        for i in range(storage.capacity_items):
            e_idx = storage._get_external_index(i)
            i_idx = storage._get_internal_index(e_idx)
            assert (i % storage.capacity_items) == i_idx

        self._add_data_to_storage(storage, batch_size=5, num_batches=3)
        assert len(storage) == 3
        assert storage.num_timesteps_added == 3 * 5
        assert storage.num_timesteps == 3 * 5
        assert storage._oldest_item_idx == 0
        assert storage.eviction_started is False

        for i in range(len(storage)):
            i_idx = storage._get_internal_index(i)
            e_idx = storage._get_external_index(i_idx)
            assert (i % len(storage)) == e_idx

        for i in range(storage.capacity_items):
            e_idx = storage._get_external_index(i)
            i_idx = storage._get_internal_index(e_idx)
            assert (i % storage.capacity_items) == i_idx

        self._add_data_to_storage(storage, batch_size=10, num_batches=3)
        assert len(storage) == 5
        assert storage.num_timesteps_added == 3 * 5 + 3 * 10
        assert storage.num_timesteps == 2 * 5 + 3 * 10
        assert storage._oldest_item_idx == 1
        assert storage.eviction_started is True

        for i in range(len(storage)):
            i_idx = storage._get_internal_index(i)
            e_idx = storage._get_external_index(i_idx)
            assert (i % len(storage)) == e_idx

        for i in range(storage.capacity_items):
            e_idx = storage._get_external_index(i)
            i_idx = storage._get_internal_index(e_idx)
            assert (i % storage.capacity_items) == i_idx

    def test_memory_and_disk_behavior(self):
        """Apply same operations to `InMemoryStorage` and `OnDiskStorage`
        and check if both storages behave the same.
        """
        m_storage = InMemoryStorage(capacity_ts=100)
        d_storage = OnDiskStorage(capacity_ts=100)

        self._add_data_to_storage(m_storage, batch_size=10, num_batches=3)
        for b in m_storage:
            d_storage.add(b)

        check(list(m_storage), list(d_storage))

        self._add_data_to_storage(m_storage, batch_size=15, num_batches=4)
        assert m_storage.eviction_started is False
        for i in range(3, len(m_storage)):
            d_storage.add(m_storage[i])

        self._add_data_to_storage(m_storage, batch_size=15, num_batches=1)
        assert m_storage.eviction_started is True
        d_storage.add(m_storage[len(m_storage) - 1])

        assert m_storage._oldest_item_idx == d_storage._oldest_item_idx
        assert m_storage._num_timesteps_added == d_storage._num_timesteps_added
        assert m_storage._num_timesteps == d_storage._num_timesteps
        assert m_storage.eviction_started == d_storage.eviction_started
        assert m_storage.size_bytes == d_storage.size_bytes
        check(list(m_storage), list(d_storage))

    def test_allocation_plans(self):
        """Apply same operations to `InMemoryStorage` with `one-time`
        and `dynamic` allocation plan and check if both storages behave
        the same.
        """
        m_storage = InMemoryStorage(capacity_ts=100, allocation_plan="one-time")
        d_storage = InMemoryStorage(capacity_ts=100, allocation_plan="dynamic")

        self._add_data_to_storage(m_storage, batch_size=10, num_batches=3)
        for b in m_storage:
            d_storage.add(b)

        check(list(m_storage), list(d_storage))

        self._add_data_to_storage(m_storage, batch_size=15, num_batches=4)
        assert m_storage.eviction_started is False
        for i in range(3, len(m_storage)):
            d_storage.add(m_storage[i])

        self._add_data_to_storage(m_storage, batch_size=15, num_batches=1)
        assert m_storage.eviction_started is True
        d_storage.add(m_storage[len(m_storage) - 1])

        assert m_storage._oldest_item_idx == d_storage._oldest_item_idx
        assert m_storage._num_timesteps_added == d_storage._num_timesteps_added
        assert m_storage._num_timesteps == d_storage._num_timesteps
        assert m_storage.eviction_started == d_storage.eviction_started
        assert m_storage.size_bytes == d_storage.size_bytes
        check(list(m_storage), list(d_storage))

    def test_slicing(self):
        """Create slices of a storage."""
        storage = InMemoryStorage(capacity_ts=100)
        self._add_data_to_storage(storage, batch_size=10, num_batches=7)

        # Create double view which contains storage / ring buffer twice
        view = storage[:14]
        assert 2 * len(storage) == len(view)

        # Check if upper and lower part of double view are the same
        # and equal to original storage
        view1 = view[:7]
        view2 = view[7:]
        check(list(view1), list(view2))
        check(list(view1), list(storage))

        # Create another double view in reverse order
        view_r = storage[13::-1]
        check(list(view_r), list(reversed(view)))

        # Check if slicing behaves the same as list slicing
        slices = [
            (None, None, None),
            (None, None, 1),
            (None, None, -1),
            (4, None, 1),
            (None, 4, 1),
            (4, None, -1),
            (None, 4, -1),
            (2, 5, 1),
            (2, 5, -1),
            (5, 2, 1),
            (5, 2, -1),
        ]
        storage_list = list(storage)
        for start, stop, step in slices:
            sl = slice(start, stop, step)
            storage_slice = storage[sl]
            check(list(storage_slice), storage_list[sl])
            check(list(storage_slice), storage_list[storage_slice.slice])


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
