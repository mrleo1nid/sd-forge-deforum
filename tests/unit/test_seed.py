"""Unit tests for seed generation functions."""

import pytest
from types import SimpleNamespace

import sys
sys.path.insert(0, 'scripts/deforum_helpers')

from seed import (
    MAX_SEED,
    calculate_seed_increment,
    calculate_next_control,
    generate_next_seed
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
        args = SimpleNamespace(seed_behavior='iter', seed_iter_N=3)

        # First call (control=0, 0 % 3 == 0) -> increment
        seed, control = generate_next_seed(args, 100, 0)
        assert seed == 101
        assert control == 1

        # Second call (control=1, 1 % 3 != 0) -> no increment
        seed, control = generate_next_seed(args, seed, control)
        assert seed == 101
        assert control == 2

        # Third call (control=2, 2 % 3 != 0) -> no increment
        seed, control = generate_next_seed(args, seed, control)
        assert seed == 101
        assert control == 3

        # Fourth call (control=3, 3 % 3 == 0) -> increment
        seed, control = generate_next_seed(args, seed, control)
        assert seed == 102
        assert control == 4

    def test_ladder_behavior(self):
        args = SimpleNamespace(seed_behavior='ladder', seed_iter_N=1)

        seed, control = generate_next_seed(args, 100, 0)
        assert seed == 102  # +2
        assert control == 1

        seed, control = generate_next_seed(args, seed, control)
        assert seed == 101  # -1
        assert control == 0

        seed, control = generate_next_seed(args, seed, control)
        assert seed == 103  # +2
        assert control == 1

    def test_alternate_behavior(self):
        args = SimpleNamespace(seed_behavior='alternate', seed_iter_N=1)

        seed, control = generate_next_seed(args, 100, 0)
        assert seed == 101  # +1
        assert control == 1

        seed, control = generate_next_seed(args, seed, control)
        assert seed == 100  # -1
        assert control == 0

        seed, control = generate_next_seed(args, seed, control)
        assert seed == 101  # +1
        assert control == 1

    def test_fixed_behavior(self):
        args = SimpleNamespace(seed_behavior='fixed', seed_iter_N=1)

        seed, control = generate_next_seed(args, 12345, 0)
        assert seed == 12345
        assert control == 0

        seed, control = generate_next_seed(args, seed, control)
        assert seed == 12345
        assert control == 0

    def test_random_behavior_range(self):
        args = SimpleNamespace(seed_behavior='random', seed_iter_N=1)

        for _ in range(10):
            seed, control = generate_next_seed(args, 100, 0)
            assert 0 <= seed <= MAX_SEED
            assert control == 0  # Control unchanged for random


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
