import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.collapse_seq_naive import collapse 

def test_naive():
    seq_out="27.bendamustine;1.bendamustine;"
    seq="27.Bendamustine;1.Bendamustine;27.Bendamustine;1.Bendamustine"
    assert collapse(seq) == seq_out

    seq_out="7.daratumumab;7.daratumumab;"
    seq="7.Daratumumab;7.Daratumumab;7.Daratumumab;7.Daratumumab"
    assert collapse(seq) == seq_out
