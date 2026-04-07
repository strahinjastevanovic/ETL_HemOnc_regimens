import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../<repo>/src
sys.path.insert(0, str(ROOT))

from tools.seq_collapse import collapse_naive, filter_et


def test_naive():
    seq_out = "27.bendamustine;1.bendamustine;"
    seq = "27.Bendamustine;1.Bendamustine;27.Bendamustine;1.Bendamustine"
    assert collapse_naive(seq) == seq_out

    seq_out = "7.daratumumab;7.daratumumab;"
    seq = "7.Daratumumab;7.Daratumumab;7.Daratumumab;7.Daratumumab"
    assert collapse_naive(seq) == seq_out

    # zero-day tokens should be filtered before collapse
    seq_out = "7.daratumumab;7.daratumumab;"
    seq = "7.daratumumab;0.daratumumab;7.daratumumab;0.daratumumab"
    assert collapse_naive(seq) == seq_out

    # single non-zero token → min-2-rule duplicates it
    seq_out = "7.daratumumab;7.daratumumab;"
    seq = "7.daratumumab;0.daratumumab"
    assert collapse_naive(seq) == seq_out


def test_et():
    # filter_et strips @len annotations and removes zero-day tokens
    seq_out = "7.daratumumab;7.daratumumab;7.daratumumab;7.daratumumab"
    seq = "7.daratumumab@len15;0.daratumumab@len22;0.daratumumab@len28;7.daratumumab@len22;7.daratumumab@len15;0.daratumumab@len22;7.daratumumab@len22"
    assert filter_et(seq) == seq_out
