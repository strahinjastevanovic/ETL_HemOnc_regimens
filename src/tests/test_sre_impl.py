import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../<repo>/src
sys.path.insert(0, str(ROOT))

from tools.sre_tools import collapse_event_matrix_wrapper, collapse_event_matrix



def test_collapse_event_matrix_examples():
    examples = [
        (
            {
                "Gemicitabine": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "Paclitaxel":   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            "14.Gemicitabine;0.Paclitaxel;7.Gemicitabine",
        ),
        (
            {"Docetaxel": [1, 0, 0, 0, 0, 0, 0]},
            "7.Docetaxel;7.Docetaxel",
        ),
        (
            {
                "Pembrolizumab": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "Cisplatin":     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "Gemcitabine":   [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            "15.Cisplatin;0.Gemcitabine;0.Pembrolizumab;6.Gemcitabine",
        ),
        (
            {"Pemb": [0, 0, 0, 1], "Adve": [1, 0, 0, 0], "Beijign": [0, 1, 1, 0], "Wara": [1, 1, 1, 1]},
            "1.Adve;0.Wara;1.Beijign;0.Wara;1.Beijign;0.Wara;1.Pemb;0.Wara",
        ),
        (
            {"bend": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "boro": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]},
            "4.bend;0.boro;1.bend;3.boro;4.boro",
        ),
        (
            {"Bendamustine": [1, 1] + [0] * 26},
            "27.Bendamustine;1.Bendamustine;27.Bendamustine;1.Bendamustine",
        ),
    ]

    for event_string, expected in examples:
        got = collapse_event_matrix(event_string)
        assert got == expected


def test_collapse_event_matrix_wrapper_esv_cases():
    esv1 = {
        "Cy": [("1", np.array([1] + [0] * 2, dtype=int))],
        "Ep": [("1", np.array([1] + [0] * 2, dtype=int))],
    }

    esv2 = {
        "Cy": [("1", np.array([1] + [0] * 2, dtype=int))],
        "Ep": [("1", np.array([1] + [0] * 2, dtype=int))],
        "Eb": [("1", np.array([1] + [0] * 6 + [1] + [0] * 3 + [1] + [0] * 2, dtype=int))],
    }

    esv3 = {
        "Cy": [("1,3,4", np.array([1] + [0] * 5, dtype=int))],
        "Ep": [("2,5",   np.array([1] + [0] * 6, dtype=int))],
    }

    esv4 = {
        "Cy": [("1,3,4", np.array([1] + [0] * 5, dtype=int))],
        "Ep": [("2,5",   np.array([1] + [0] * 6, dtype=int))],
        "Eb": [("2,5",   np.array([1] + [0] * 6, dtype=int))],
    }

    esv5 = {
        "Cy": [
            ("1,3,4", np.array([1] + [0] * 4, dtype=int)),
            ("1,3,4", np.array([1] + [0] * 9, dtype=int)),
        ],
        "Ep": [
            ("2,5", np.array([1] + [0] * 2 + [1] * 2, dtype=int)),
        ],
    }

    esv6 = {
        "Cy": [("1,3,4", np.array([1] + [0] * 5, dtype=int))],
        "Ep": [("2,5",   np.array([1] + [0] * 6, dtype=int))],
    }

    expected = [
        "3.Cy;0.Ep",
        "3.Cy;0.Eb;0.Ep;7.Eb;4.Eb",
        "7.Cy;0.Ep",
        "7.Cy;0.Eb;0.Ep",
        "1.Cy;0.Ep;3.Ep;1.Ep",
        "6.Cy;0.Ep;3.Ep;1.Ep",
        "7.Cy;0.Ep",
    ]

    got = []
    i=0
    for x in [esv1, esv2, esv3, esv4, esv5, esv6]:
        print(f"{i} x=",x)
        got.extend(collapse_event_matrix_wrapper(x))
        print("got:",got[i])
        i+=1

    assert got == expected


def test_collapse_event_matrix_raises_on_mismatched_lengths():
    with pytest.raises(ValueError, match="mismatched length"):
        collapse_event_matrix({"A": [1, 0, 1], "B": [1, 0]})


def test_collapse_event_matrix_wrapper_raises_on_all_zero_variant():
    bad = {
        "Cy": [("1", np.array([0, 0, 0], dtype=int))],
        "Ep": [("1", np.array([0, 0, 0], dtype=int))],
    }
    with pytest.raises(ValueError, match="zero-only"):
        collapse_event_matrix_wrapper(bad)