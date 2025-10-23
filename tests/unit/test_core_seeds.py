"""Unit tests for deforum.core.seeds module."""

import pytest
from types import SimpleNamespace

from deforum.core.seeds import (
    MAX_SEED,
    SeedIterator,
    next_seed,
)


class TestSeedIteratorInit:
    """Test SeedIterator initialization."""

    def test_default_initialization(self):
        """Default initialization with iter behavior."""
        iterator = SeedIterator(seed=42)
        assert iterator.seed == 42
        assert iterator.behavior == 'iter'
        assert iterator.iter_n == 1
        assert iterator.internal_counter == 0

    def test_custom_behavior(self):
        """Initialize with custom behavior."""
        iterator = SeedIterator(seed=100, behavior='ladder')
        assert iterator.seed == 100
        assert iterator.behavior == 'ladder'

    def test_custom_iter_n(self):
        """Initialize with custom iter_n."""
        iterator = SeedIterator(seed=50, behavior='iter', iter_n=10)
        assert iterator.iter_n == 10

    def test_invalid_behavior_raises_error(self):
        """Invalid behavior should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid seed behavior"):
            SeedIterator(seed=42, behavior='invalid')


class TestSeedIteratorIterMode:
    """Test SeedIterator with 'iter' behavior."""

    def test_iter_every_frame(self):
        """Iter mode with N=1 increments every frame."""
        iterator = SeedIterator(seed=42, behavior='iter', iter_n=1)
        # With N=1, increments on every call (0, 1, 2, 3, ...)
        assert iterator.next() == 43  # 42 + 1
        assert iterator.next() == 44  # 43 + 1
        assert iterator.next() == 45  # 44 + 1
        assert iterator.next() == 46  # 45 + 1

    def test_iter_every_2_frames(self):
        """Iter mode with N=2 increments every 2nd frame."""
        iterator = SeedIterator(seed=100, behavior='iter', iter_n=2)
        seeds = [iterator.next() for _ in range(6)]
        # Increments on call 0, 2, 4, ... (when counter % 2 == 0)
        assert seeds == [101, 101, 102, 102, 103, 103]

    def test_iter_every_5_frames(self):
        """Iter mode with N=5 increments every 5th frame."""
        iterator = SeedIterator(seed=0, behavior='iter', iter_n=5)
        seeds = [iterator.next() for _ in range(10)]
        # Increments on call 0, 5, ... (when counter % 5 == 0)
        assert seeds == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    def test_iter_every_10_frames(self):
        """Iter mode with N=10 for typical use case."""
        iterator = SeedIterator(seed=1000, behavior='iter', iter_n=10)
        seeds = [iterator.next() for _ in range(20)]
        # Increments on call 0, 10, ... (when counter % 10 == 0)
        expected = [1001] * 10 + [1002] * 10
        assert seeds == expected


class TestSeedIteratorLadderMode:
    """Test SeedIterator with 'ladder' behavior."""

    def test_ladder_pattern(self):
        """Ladder mode follows +2, -1, +2, -1 pattern."""
        iterator = SeedIterator(seed=100, behavior='ladder')
        seeds = [iterator.next() for _ in range(8)]
        # Starting at 100: +2=102, -1=101, +2=103, -1=102, +2=104, -1=103, +2=105, -1=104
        assert seeds == [102, 101, 103, 102, 104, 103, 105, 104]

    def test_ladder_from_zero(self):
        """Ladder mode starting from seed 0."""
        iterator = SeedIterator(seed=0, behavior='ladder')
        seeds = [iterator.next() for _ in range(6)]
        assert seeds == [2, 1, 3, 2, 4, 3]

    def test_ladder_maintains_pattern_after_reset(self):
        """Ladder pattern resets correctly."""
        iterator = SeedIterator(seed=50, behavior='ladder')
        iterator.next()
        iterator.next()
        iterator.reset(seed=100)
        seeds = [iterator.next() for _ in range(4)]
        assert seeds == [102, 101, 103, 102]


class TestSeedIteratorAlternateMode:
    """Test SeedIterator with 'alternate' behavior."""

    def test_alternate_pattern(self):
        """Alternate mode follows +1, -1, +1, -1 pattern."""
        iterator = SeedIterator(seed=100, behavior='alternate')
        seeds = [iterator.next() for _ in range(8)]
        # Starting at 100: +1=101, -1=100, +1=101, -1=100, ...
        assert seeds == [101, 100, 101, 100, 101, 100, 101, 100]

    def test_alternate_from_zero(self):
        """Alternate mode starting from seed 0."""
        iterator = SeedIterator(seed=0, behavior='alternate')
        seeds = [iterator.next() for _ in range(6)]
        # Starting at 0: +1=1, -1=0, +1=1, -1=0, +1=1, -1=0
        assert seeds == [1, 0, 1, 0, 1, 0]

    def test_alternate_oscillates(self):
        """Alternate mode oscillates between two values."""
        iterator = SeedIterator(seed=42, behavior='alternate')
        seeds = [iterator.next() for _ in range(20)]
        # Should oscillate between 43 and 42
        assert seeds == [43, 42] * 10


class TestSeedIteratorFixedMode:
    """Test SeedIterator with 'fixed' behavior."""

    def test_fixed_seed_constant(self):
        """Fixed mode keeps seed constant."""
        iterator = SeedIterator(seed=42, behavior='fixed')
        seeds = [iterator.next() for _ in range(100)]
        assert all(s == 42 for s in seeds)

    def test_fixed_different_seed(self):
        """Fixed mode with different initial seed."""
        iterator = SeedIterator(seed=999, behavior='fixed')
        seeds = [iterator.next() for _ in range(50)]
        assert all(s == 999 for s in seeds)


class TestSeedIteratorRandomMode:
    """Test SeedIterator with 'random' behavior."""

    def test_random_generates_different_seeds(self):
        """Random mode generates varying seeds."""
        iterator = SeedIterator(seed=42, behavior='random')
        seeds = [iterator.next() for _ in range(100)]
        # Should have variation (not all the same)
        assert len(set(seeds)) > 1

    def test_random_seeds_in_valid_range(self):
        """Random seeds should be in valid range."""
        iterator = SeedIterator(seed=42, behavior='random')
        seeds = [iterator.next() for _ in range(100)]
        assert all(0 <= s <= MAX_SEED for s in seeds)

    def test_random_produces_unique_sequences(self):
        """Two random iterators produce different sequences."""
        iter1 = SeedIterator(seed=42, behavior='random')
        iter2 = SeedIterator(seed=42, behavior='random')
        seeds1 = [iter1.next() for _ in range(50)]
        seeds2 = [iter2.next() for _ in range(50)]
        # Should be different (probability of collision is tiny)
        assert seeds1 != seeds2


class TestSeedIteratorReset:
    """Test SeedIterator reset functionality."""

    def test_reset_without_new_seed(self):
        """Reset keeps current seed."""
        iterator = SeedIterator(seed=42, behavior='iter', iter_n=1)
        iterator.next()
        iterator.next()
        iterator.next()
        assert iterator.seed == 45
        iterator.reset()
        assert iterator.seed == 45
        assert iterator.internal_counter == 0

    def test_reset_with_new_seed(self):
        """Reset with new seed value."""
        iterator = SeedIterator(seed=42, behavior='iter', iter_n=1)
        iterator.next()
        iterator.next()
        iterator.reset(seed=100)
        assert iterator.seed == 100
        assert iterator.internal_counter == 0

    def test_reset_restarts_pattern(self):
        """Reset restarts iteration pattern."""
        iterator = SeedIterator(seed=100, behavior='ladder')
        for _ in range(5):
            iterator.next()
        iterator.reset(seed=200)
        seeds = [iterator.next() for _ in range(4)]
        # Should restart ladder pattern from 200
        assert seeds == [202, 201, 203, 202]


class TestNextSeedLegacyFunction:
    """Test legacy next_seed function for backward compatibility."""

    def test_iter_mode_legacy(self):
        """Legacy function with iter mode."""
        args = SimpleNamespace(seed=42, seed_behavior='iter', seed_iter_N=2)
        root = SimpleNamespace(seed_internal=0)

        # First call: counter=0, 0%2==0, so increment
        seed1 = next_seed(args, root)
        assert seed1 == 43  # 42 + 1
        assert root.seed_internal == 1

        # Second call: counter=1, 1%2==1, so no increment
        seed2 = next_seed(args, root)
        assert seed2 == 43
        assert root.seed_internal == 2

        # Third call: counter=2, 2%2==0, so increment
        seed3 = next_seed(args, root)
        assert seed3 == 44  # 43 + 1
        assert root.seed_internal == 3

    def test_ladder_mode_legacy(self):
        """Legacy function with ladder mode."""
        args = SimpleNamespace(seed=100, seed_behavior='ladder')
        root = SimpleNamespace(seed_internal=0)

        seeds = []
        for _ in range(6):
            seeds.append(next_seed(args, root))

        assert seeds == [102, 101, 103, 102, 104, 103]

    def test_alternate_mode_legacy(self):
        """Legacy function with alternate mode."""
        args = SimpleNamespace(seed=100, seed_behavior='alternate')
        root = SimpleNamespace(seed_internal=0)

        seeds = []
        for _ in range(6):
            seeds.append(next_seed(args, root))

        assert seeds == [101, 100, 101, 100, 101, 100]

    def test_fixed_mode_legacy(self):
        """Legacy function with fixed mode."""
        args = SimpleNamespace(seed=42, seed_behavior='fixed')
        root = SimpleNamespace(seed_internal=0)

        seeds = []
        for _ in range(10):
            seeds.append(next_seed(args, root))

        assert all(s == 42 for s in seeds)

    def test_random_mode_legacy(self):
        """Legacy function with random mode."""
        args = SimpleNamespace(seed=42, seed_behavior='random')
        root = SimpleNamespace(seed_internal=0)

        seeds = []
        for _ in range(50):
            seeds.append(next_seed(args, root))

        # Should have variation
        assert len(set(seeds)) > 1
        # Should be in valid range
        assert all(0 <= s <= MAX_SEED for s in seeds)


class TestSeedIteratorIntegration:
    """Integration tests for SeedIterator."""

    def test_typical_animation_workflow(self):
        """Simulate typical animation seed iteration."""
        # 100 frame animation, increment seed every 10 frames
        iterator = SeedIterator(seed=42, behavior='iter', iter_n=10)
        seeds = [iterator.next() for _ in range(100)]

        # Should have 10 unique seeds (43-52, since first call increments)
        unique_seeds = list(dict.fromkeys(seeds))
        assert len(unique_seeds) == 10
        assert unique_seeds == list(range(43, 53))

    def test_resume_animation_workflow(self):
        """Simulate resuming animation from specific seed."""
        # Start animation
        iterator = SeedIterator(seed=1000, behavior='iter', iter_n=5)
        for _ in range(25):  # Process 25 frames
            iterator.next()

        # Resume from frame 25 (seed should be 1005 after 25 frames with iter_n=5)
        resume_seed = iterator.seed
        assert resume_seed == 1005

        # Continue animation
        new_iterator = SeedIterator(seed=resume_seed, behavior='iter', iter_n=5)
        new_iterator.internal_counter = 0  # Reset counter for resuming
        seeds = [new_iterator.next() for _ in range(25)]

        # Should continue incrementing from 1005
        expected_final_seed = 1010
        assert seeds[-1] == expected_final_seed

    def test_behavior_switching(self):
        """Test switching between different behaviors."""
        # Start with iter
        iter1 = SeedIterator(seed=100, behavior='iter', iter_n=1)
        seeds_iter = [iter1.next() for _ in range(5)]
        # Each call increments: 101, 102, 103, 104, 105
        assert seeds_iter == [101, 102, 103, 104, 105]

        # Switch to ladder from current seed
        iter2 = SeedIterator(seed=iter1.seed, behavior='ladder')
        seeds_ladder = [iter2.next() for _ in range(4)]
        # Starting from 105: +2=107, -1=106, +2=108, -1=107
        assert seeds_ladder[0] == 107  # 105 + 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
