"""Sanity checks for alias_match layered matcher."""
from __future__ import annotations

from saber.data.alias_match import alias_match, normalise


def test_exact():
    assert alias_match("Paris", ["Paris"])
    assert alias_match("paris", ["Paris"])


def test_strip_articles():
    assert alias_match("the politician", ["politician"])
    assert alias_match("politician", ["a politician"])


def test_whole_word_positive():
    assert alias_match("The answer is London.", ["London"])
    assert alias_match("London", ["the answer is London"])


def test_substring_no_longer_matches_false_positive():
    # Prior substring-matcher accepted "pol" ⊂ "political"; the tightened
    # matcher must reject this.
    assert not alias_match("political", ["pol"])
    assert not alias_match("pol", ["political"])


def test_multi_word_jaccard():
    # Same tokens in different order after article stripping → Jaccard 1.0.
    assert alias_match("City of New York", ["New York City"])
    # Single extra filler word keeps Jaccard above the 0.8 threshold.
    assert alias_match("New York City", ["New York the City"])
    # Different-tokens example must NOT match (Jaccard below threshold).
    assert not alias_match("Steven Spielberg", ["George Spielberg"])


def test_no_match():
    assert not alias_match("journalist", ["politician"])
    assert not alias_match("", ["politician"])
    assert not alias_match("politician", [])
    assert not alias_match(None, ["politician"])


def test_normalise_punctuation():
    assert normalise("Mr. Smith, Jr.") == "mr smith jr"
