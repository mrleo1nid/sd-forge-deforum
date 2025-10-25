"""Unit tests for seed generation functions."""

import pytest
from types import SimpleNamespace

from deforum.utils.generation.seeds import (
    MAX_SEED,
    calculate_seed_increment,
    calculate_next_control,
    generate_next_seed,
)


class TestCalculateSeedIncrement:
    """Test pure seed increment calculation."""

    def test_iter_on_interval(self):
        assert calculate_seed_increment('iter', 0, 5) == 1
        assert calculate_seed_increment('iter', 5, 5) == 1
        assert calculate_seed_increment('iter', 10, 5) == 1

    def test_iter_off_interval(self):
        assert calculate_seed_increment('iter', 1, 5) == 0
        assert calculate_seed_increment('iter', 4, 5) == 0

    def test_ladder(self):
        assert calculate_seed_increment('ladder', 0, 5) == 2
        assert calculate_seed_increment('ladder', 1, 5) == -1

    def test_alternate(self):
        assert calculate_seed_increment('alternate', 0, 5) == 1
        assert calculate_seed_increment('alternate', 1, 5) == -1

    def test_fixed(self):
        assert calculate_seed_increment('fixed', 0, 5) == 0
        assert calculate_seed_increment('fixed', 100, 5) == 0


class TestCalculateNextControl:
    """Test pure control state calculation."""

    def test_ladder_alternates(self):
        assert calculate_next_control('ladder', 0) == 1
        assert calculate_next_control('ladder', 1) == 0

    def test_alternate_alternates(self):
        assert calculate_next_control('alternate', 0) == 1
        assert calculate_next_control('alternate', 1) == 0

    def test_iter_increments(self):
        assert calculate_next_control('iter', 0) == 1
        assert calculate_next_control('iter', 5) == 6
        assert calculate_next_control('iter', 100) == 101

    def test_other_unchanged(self):
        assert calculate_next_control('fixed', 42) == 42
        assert calculate_next_control('random', 42) == 42


class TestGenerateNextSeed:
    """Test complete seed generation logic."""

    def test_iter_behavior(self):
        behavior = 'iter'
        iter_n = 3

        # First call (control=0, 0 % 3 == 0) -> increment
        seed, control = generate_next_seed(100, behavior, 0, iter_n)
        assert seed == 101
        assert control == 1

        # Second call (control=1, 1 % 3 != 0) -> no increment
        seed, control = generate_next_seed(seed, behavior, control, iter_n)
        assert seed == 101
        assert control == 2

        # Third call (control=2, 2 % 3 != 0) -> no increment
        seed, control = generate_next_seed(seed, behavior, control, iter_n)
        assert seed == 101
        assert control == 3

        # Fourth call (control=3, 3 % 3 == 0) -> increment
        seed, control = generate_next_seed(seed, behavior, control, iter_n)
        assert seed == 102
        assert control == 4

    def test_ladder_behavior(self):
        behavior = 'ladder'
        iter_n = 1

        seed, control = generate_next_seed(100, behavior, 0, iter_n)
        assert seed == 102  # +2
        assert control == 1

        seed, control = generate_next_seed(seed, behavior, control, iter_n)
        assert seed == 101  # -1
        assert control == 0

        seed, control = generate_next_seed(seed, behavior, control, iter_n)
        assert seed == 103  # +2
        assert control == 1

    def test_alternate_behavior(self):
        behavior = 'alternate'
        iter_n = 1

        seed, control = generate_next_seed(100, behavior, 0, iter_n)
        assert seed == 101  # +1
        assert control == 1

        seed, control = generate_next_seed(seed, behavior, control, iter_n)
        assert seed == 100  # -1
        assert control == 0

        seed, control = generate_next_seed(seed, behavior, control, iter_n)
        assert seed == 101  # +1
        assert control == 1

    def test_fixed_behavior(self):
        behavior = 'fixed'
        iter_n = 1

        seed, control = generate_next_seed(12345, behavior, 0, iter_n)
        assert seed == 12345
        assert control == 0

        seed, control = generate_next_seed(seed, behavior, control, iter_n)
        assert seed == 12345
        assert control == 0

    def test_random_behavior_range(self):
        behavior = 'random'
        iter_n = 1

        for _ in range(10):
            seed, control = generate_next_seed(100, behavior, 0, iter_n)
            assert 0 <= seed <= MAX_SEED
            assert control == 0  # Control unchanged for random


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
