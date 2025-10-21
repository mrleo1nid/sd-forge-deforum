"""
Smoke tests - Verify test infrastructure is working.

These are intentionally trivial tests just to ensure pytest and coverage work.
Real tests will be added during refactoring.
"""

import pytest


def test_assertions_work():
    """Verify basic assertions work"""
    assert True
    assert 1 == 1
    assert "hello" == "hello"


def test_math_works():
    """Verify basic math works"""
    assert 2 + 2 == 4
    assert 10 - 5 == 5


@pytest.mark.parametrize("a,b,expected", [
    (1, 1, 2),
    (2, 3, 5),
    (10, 20, 30),
])
def test_addition_parametrized(a, b, expected):
    """Verify parametrized tests work"""
    assert a + b == expected


class TestPytestFeatures:
    """Verify pytest features work"""

    def test_class_based_tests_work(self):
        """Verify class-based tests work"""
        assert True

    def test_multiple_assertions(self):
        """Verify multiple assertions in one test"""
        x = 5
        assert x == 5
        assert x > 0
        assert x < 10
