import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../<repo>/src
sys.path.insert(0, str(ROOT))

from tools.sre_tools import collapse_event_matrix_wrapper, collapse_event_matrix


# TODO: Make sure examples match running tests in __file__ ?

# def run_test():
#     examples = [
#         [
#             {'Gemicitabine': [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Paclitaxel': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
#               "14.Gemicitabine;0.Paclitaxel;7.Gemicitabine"
#         ],
#         [{'Docetaxel': [1, 0, 0, 0, 0, 0, 0]}, "7.Docetaxel;7.Docetaxel"],
#         [{'Pembrolizumab': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Cisplatin': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Gemcitabine': [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
#         "15.Cisplatin;0.Gemcitabine;0.Pembrolizumab;6.Gemcitabine"],
#         [
#         {"Pemb": [0,0,0,1], "Adve":[1,0,0,0], "Beijign":[0,1,1,0], "Wara":[1,1,1,1]},
#         "1.Adve;0.Wara;1.Beijign;0.Wara;1.Beijign;0.Wara;1.Pemb;0.Wara"
#         ],
#         [
#         {
#             "bend": [1,1,0,0,0,0,0,0,0,0,0,0], 
#             "boro": [1,0,0,0,1,0,0,0,1,0,0,0]
#         },
#         "4.bend;0.boro;1.bend;3.boro;4.boro" # note - you need to build entire matrix to estimate deltas (should work accross components NOT per component)
#         ],
#         [
#             {
#             'Bendamustine': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#             },
#             "27.Bendamustine;1.Bendamustine;27.Bendamustine;1.Bendamustine"
#         ],
#         [
#             {
#             'Bendamustine': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             'Bendamustine': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#             },
#             "23.Bendamustine;5.Bendamustine;23.Bendamustine;5.Bendamustine" # ??? looks at it as individual but shouldnt!
#         ]
#     ]
#     for example in examples:
#         # print(example)
#         rrr = collapse_event_matrix(example[0])
#         assert rrr == example[1], f"not eq: \n{example[1]}\n{rrr}"

   
#     esv1 = {  # case from indeterminate
#         'Cy': [("1", [1] + [0]*2)],
#         'Ep': [("1", [1] + [0]*2)],
#     }
#     esv1 = {k : [(v[0][0], np.array(v[0][1]))] for k,v in esv1.items()}
#     esv2 = {  # case from indeterminate + edge case if same cycle need padding
#         'Cy': [("1", [1] + [0]*2)],
#         'Ep': [("1", [1] + [0]*2)],
#         'Eb': [("1", [1] + [0]*6 + [1] + [0]*3 + [1] + [0]*2)],
#     }
#     esv2 = {k : [(v[0][0], np.array(v[0][1]))] for k,v in esv2.items()}
#     esv3 = { # case with different cycles
#         'Cy': [("1,3,4", [1] + [0]*5)],
#         'Ep': [("2,5", [1] + [0]*6)],
#     }
#     esv3 = {k : [(v[0][0], np.array(v[0][1]))] for k,v in esv3.items()}
#     esv4 = { # case with same cycles multiple groups
#         'Cy': [("1,3,4", [1] + [0]*5)],
#         'Ep': [("2,5", [1] + [0]*6)],
#         'Eb': [("2,5", [1] + [0]*6)],

#     }
#     esv4 = {k : [(v[0][0], np.array(v[0][1]))] for k,v in esv4.items()}

#     esv5 = { # case with ub,lb + conditional missing ub in one
#         'Cy': [("1,3,4", np.array([1] + [0]*4)), ("1,3,4", np.array([1]+[0]*9))],
#         'Ep': [("2,5", np.array([1] + [0]*2 + [1]*2)), ], 
#     }

#     esv6 = { 
#         'Cy': [("1,3,4", [1] + [0]*5)],
#         'Ep': [("2,5", [1] + [0]*5)],
#     }
#     esv6 = {k : [(v[0][0], np.array(v[0][1]))] for k,v in esv3.items()}
#     # print(event_string_example)
#     esv_expected = [
#        '3.Cy;0.Ep', 
#        '3.Cy;0.Eb;0.Ep;7.Eb;4.Eb', 
#        '7.Cy;0.Ep', 
#        '7.Cy;0.Eb;0.Ep', 
#        '1.Cy;0.Ep;3.Ep;1.Ep', 
#        '6.Cy;0.Ep;3.Ep;1.Ep', 
#        '7.Cy;0.Ep'

#     ]
#     ress = []
#     for x in [esv1,esv2, esv3,esv4,esv5,esv6]:
#         ress.extend(collapse_event_matrix_wrapper(x))

#     assert ress == esv_expected, f"No match!\nActual:   {ress}\nExpected: {esv_expected}"

#     print("All tests passed!")





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

