"""Unit tests for functional programming utilities."""

import pytest
from PIL import Image

from deforum.utils.functional_utils import (
    flat_map,
    tube,
    put_all,
    put_if_present,
    call_or_use_on_cond,
    create_img,
    generate_random_seed,
)


class TestFlatMap:
    """Test flat_map function."""

    def test_flat_map_with_lists(self):
        result = list(flat_map(lambda x: [x, x * 2], [1, 2, 3]))
        assert result == [1, 2, 2, 4, 3, 6]

    def test_flat_map_with_single_values(self):
        result = list(flat_map(lambda x: x + 1, [1, 2, 3]))
        assert result == [2, 3, 4]

    def test_flat_map_empty_iterable(self):
        result = list(flat_map(lambda x: [x, x * 2], []))
        assert result == []


class TestTube:
    """Test function composition pipeline."""

    def test_tube_single_function(self):
        add_one = lambda x: x + 1
        pipeline = tube(add_one)
        assert pipeline(5) == 6

    def test_tube_multiple_functions(self):
        add_one = lambda x: x + 1
        double = lambda x: x * 2
        pipeline = tube(add_one, double)
        assert pipeline(5) == 12  # (5 + 1) * 2

    def test_tube_with_condition_true(self):
        add_one = lambda x: x + 1
        double = lambda x: x * 2
        pipeline = tube(add_one, double, is_do_process=lambda: True)
        assert pipeline(5) == 12

    def test_tube_with_condition_false(self):
        add_one = lambda x: x + 1
        double = lambda x: x * 2
        pipeline = tube(add_one, double, is_do_process=lambda: False)
        assert pipeline(5) == 5  # No processing

    def test_tube_three_functions(self):
        add_one = lambda x: x + 1
        double = lambda x: x * 2
        subtract_three = lambda x: x - 3
        pipeline = tube(add_one, double, subtract_three)
        assert pipeline(5) == 9  # ((5 + 1) * 2) - 3


class TestPutAll:
    """Test dictionary utilities."""

    def test_put_all_adds_to_all_dicts(self):
        dicts = [{'a': 1}, {'b': 2}, {'c': 3}]
        result = put_all(dicts, 'x', 99)
        assert result == [
            {'a': 1, 'x': 99},
            {'b': 2, 'x': 99},
            {'c': 3, 'x': 99}
        ]

    def test_put_all_empty_list(self):
        result = put_all([], 'x', 99)
        assert result == []

    def test_put_all_preserves_original(self):
        dicts = [{'a': 1}]
        result = put_all(dicts, 'x', 99)
        assert dicts == [{'a': 1}]  # Original unchanged
        assert result == [{'a': 1, 'x': 99}]

    def test_put_all_overwrites_existing_key(self):
        dicts = [{'x': 1}, {'x': 2}]
        result = put_all(dicts, 'x', 99)
        assert result == [{'x': 99}, {'x': 99}]


class TestPutIfPresent:
    """Test conditional dictionary update."""

    def test_put_if_present_with_value(self):
        d = {'a': 1}
        put_if_present(d, 'b', 2)
        assert d == {'a': 1, 'b': 2}

    def test_put_if_present_with_none(self):
        d = {'a': 1}
        put_if_present(d, 'b', None)
        assert d == {'a': 1}  # No change

    def test_put_if_present_overwrites_existing(self):
        d = {'a': 1}
        put_if_present(d, 'a', 99)
        assert d == {'a': 99}

    def test_put_if_present_with_zero(self):
        d = {'a': 1}
        put_if_present(d, 'b', 0)
        assert d == {'a': 1, 'b': 0}  # 0 is not None


class TestCallOrUseOnCond:
    """Test conditional execution."""

    def test_call_or_use_on_cond_with_callable_true(self):
        result = call_or_use_on_cond(True, lambda: 42)
        assert result == 42

    def test_call_or_use_on_cond_with_callable_false(self):
        result = call_or_use_on_cond(False, lambda: 42)
        assert result is None

    def test_call_or_use_on_cond_with_value_true(self):
        result = call_or_use_on_cond(True, 42)
        assert result == 42

    def test_call_or_use_on_cond_with_value_false(self):
        result = call_or_use_on_cond(False, 42)
        assert result is None


class TestCreateImg:
    """Test PIL image creation."""

    def test_create_img_dimensions(self):
        img = create_img((100, 50))
        assert img.size == (100, 50)

    def test_create_img_mode(self):
        img = create_img((10, 10))
        assert img.mode == '1'

    def test_create_img_filled_with_white(self):
        img = create_img((2, 2))
        pixels = list(img.getdata())
        assert all(p == 1 for p in pixels)


class TestGenerateRandomSeed:
    """Test random seed generation."""

    def test_generate_random_seed_in_range(self):
        seed = generate_random_seed()
        assert 0 <= seed <= 2**32 - 1

    def test_generate_random_seed_is_integer(self):
        seed = generate_random_seed()
        assert isinstance(seed, int)

    def test_generate_random_seed_varies(self):
        seeds = [generate_random_seed() for _ in range(10)]
        # Very unlikely all 10 seeds are identical
        assert len(set(seeds)) > 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
