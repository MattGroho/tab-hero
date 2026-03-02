"""Custom samplers for efficient batching.

BucketBatchSampler groups samples by length to minimize padding waste.
WeightedInstrumentSampler addresses class imbalance across instruments.
CurriculumSampler implements difficulty-based curriculum learning.
"""

import random
from collections import Counter
from typing import Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import Sampler, WeightedRandomSampler


class BucketBatchSampler(Sampler[List[int]]):
    """
    Sampler that groups samples into buckets by length.

    Reduces padding waste by batching samples of similar length together.
    Particularly useful for variable-length audio/sequence data.

    Args:
        lengths: List of sample lengths (e.g., mel frame counts)
        batch_size: Number of samples per batch
        num_buckets: Number of length buckets
        shuffle: Whether to shuffle within and across buckets
        drop_last: Whether to drop incomplete final batch
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        num_buckets: int = 10,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Create buckets based on length distribution
        self.buckets = self._create_buckets()

    def _create_buckets(self) -> List[List[int]]:
        """Assign sample indices to buckets based on length."""
        if len(self.lengths) == 0:
            return [[] for _ in range(self.num_buckets)]

        # Get length range
        min_len = min(self.lengths)
        max_len = max(self.lengths)
        bucket_width = (max_len - min_len + 1) / self.num_buckets

        if bucket_width == 0:
            # All samples same length
            return [list(range(len(self.lengths)))]

        # Assign to buckets
        buckets = [[] for _ in range(self.num_buckets)]
        for idx, length in enumerate(self.lengths):
            bucket_idx = min(
                int((length - min_len) / bucket_width),
                self.num_buckets - 1
            )
            buckets[bucket_idx].append(idx)

        # Remove empty buckets
        return [b for b in buckets if len(b) > 0]

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle within buckets if needed
        if self.shuffle:
            buckets = [random.sample(b, len(b)) for b in self.buckets]
        else:
            buckets = [list(b) for b in self.buckets]

        # Create batches from each bucket
        all_batches = []
        for bucket in buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        # Shuffle batches across buckets
        if self.shuffle:
            random.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        total = 0
        for bucket in self.buckets:
            n_batches = len(bucket) // self.batch_size
            if not self.drop_last and len(bucket) % self.batch_size > 0:
                n_batches += 1
            total += n_batches
        return total


class DynamicBatchSampler(Sampler[List[int]]):
    """
    Sampler that creates batches with roughly equal total tokens.

    Instead of fixed batch size, batches are formed to have approximately
    max_tokens total, allowing more short samples or fewer long samples.

    Args:
        lengths: List of sample lengths (e.g., token counts)
        max_tokens: Maximum total tokens per batch
        shuffle: Whether to shuffle samples
        drop_last: Whether to drop incomplete final batch
    """

    def __init__(
        self,
        lengths: List[int],
        max_tokens: int = 16384,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.lengths)))

        if self.shuffle:
            random.shuffle(indices)

        # Form batches with max_tokens constraint
        batches = []
        current_batch = []
        current_max_len = 0

        for idx in indices:
            sample_len = self.lengths[idx]

            # Calculate batch size if we add this sample
            new_max_len = max(current_max_len, sample_len)
            new_batch_tokens = new_max_len * (len(current_batch) + 1)

            if new_batch_tokens <= self.max_tokens or len(current_batch) == 0:
                # Add to current batch
                current_batch.append(idx)
                current_max_len = new_max_len
            else:
                # Start new batch
                batches.append(current_batch)
                current_batch = [idx]
                current_max_len = sample_len

        # Handle last batch
        if current_batch and not self.drop_last:
            batches.append(current_batch)

        yield from batches

    def __len__(self) -> int:
        # Approximate - actual count depends on length distribution
        total_tokens = sum(self.lengths)
        return max(1, total_tokens // self.max_tokens)



class WeightedInstrumentSampler(Sampler[int]):
    """Sampler that oversamples underrepresented instruments.

    Computes inverse-frequency weights per instrument class so that each
    instrument is sampled with roughly equal probability per epoch.

    Args:
        instrument_ids: Per-sample instrument class (0-3).
        num_samples: Number of samples to draw per epoch.  Defaults to
            ``len(instrument_ids)`` (one full epoch).
        replacement: Whether to sample with replacement (default True).
    """

    def __init__(
        self,
        instrument_ids: Sequence[int],
        num_samples: int | None = None,
        replacement: bool = True,
    ):
        counts = Counter(instrument_ids)
        total = len(instrument_ids)
        # Weight = total / (n_classes * count_for_class)
        n_classes = len(counts)
        class_weight = {
            cls: total / (n_classes * cnt) for cls, cnt in counts.items()
        }
        self.weights = torch.tensor(
            [class_weight[int(c)] for c in instrument_ids], dtype=torch.double
        )
        self.num_samples = num_samples if num_samples is not None else total
        self.replacement = replacement

    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(
            self.weights, self.num_samples, replacement=self.replacement
        )
        yield from indices.tolist()

    def __len__(self) -> int:
        return self.num_samples



class CurriculumSampler(Sampler[int]):
    """Sampler that progressively introduces harder difficulty levels.

    Implements curriculum learning by restricting the pool of eligible
    samples based on the current training epoch.  Call ``set_epoch()``
    at the start of each epoch to update the schedule.

    The *schedule* is a sorted list of ``(epoch, max_difficulty)`` pairs.
    At a given epoch, the sampler finds the last entry whose epoch
    threshold has been reached and includes all samples with
    ``difficulty_id <= max_difficulty``.

    Example schedule (default)::

        [(0, 1), (10, 2), (25, 3)]

    - Epochs 0-9:   easy (0) and medium (1)
    - Epochs 10-24: add hard (2)
    - Epochs 25+:   all difficulties including expert (3)

    Args:
        difficulty_ids: Per-sample difficulty class (0-3).
        schedule: List of ``(epoch, max_difficulty)`` pairs, sorted by epoch.
        shuffle: Whether to shuffle eligible indices each epoch.
    """

    def __init__(
        self,
        difficulty_ids: Sequence[int],
        schedule: Sequence[Tuple[int, int]] = ((0, 1), (10, 2), (25, 3)),
        shuffle: bool = True,
    ):
        self.difficulty_ids = list(difficulty_ids)
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.shuffle = shuffle
        self.epoch = 0
        self._eligible: List[int] = []
        self._update_eligible()

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch and recompute eligible indices."""
        self.epoch = epoch
        self._update_eligible()

    def _current_max_difficulty(self) -> int:
        """Return the max difficulty allowed at the current epoch."""
        max_diff = self.schedule[0][1]  # default to first entry
        for ep_threshold, diff in self.schedule:
            if self.epoch >= ep_threshold:
                max_diff = diff
            else:
                break
        return max_diff

    def _update_eligible(self) -> None:
        max_diff = self._current_max_difficulty()
        self._eligible = [
            i for i, d in enumerate(self.difficulty_ids) if d <= max_diff
        ]

    def __iter__(self) -> Iterator[int]:
        indices = list(self._eligible)
        if self.shuffle:
            random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self._eligible)