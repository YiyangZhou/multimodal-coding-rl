"""Trainer-side plugin; imports NeMo RL from the pinned training environment."""

from nemo_rl.data.datasets.response_datasets.nemogym_dataset import NemoGymDataset

from .mixed_balancing import balanced_indices


class BalancedMixedGymDataset(NemoGymDataset):
    def __init__(self, data_path, seed=42, balance_batch_size=8, repeat=1, benchmark_weights=None, benchmarks=None, excluded_tasks=None, **kwargs):
        if repeat != 1:
            raise ValueError("Balanced exposure owns repetition; leave repeat=1")
        super().__init__(data_path=data_path, repeat=1, **kwargs)
        indices, self.balance_report = balanced_indices(
            self.dataset["extra_env_info"], seed=seed, batch_size=balance_batch_size, benchmark_weights=benchmark_weights, benchmarks=benchmarks, excluded_tasks=excluded_tasks,
        )
        # Arrow keeps one physical copy of each JSON/image payload; only its
        # lightweight row-index mapping repeats. Source identities stay intact.
        self.dataset = self.dataset.select(indices)


