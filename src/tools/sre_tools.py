import re 
from collections import defaultdict
import numpy as np

def extract_number_deprecated(text):
    return int(re.search(r"\d+", str(text)).group()) if re.search(r"\d+", str(text)) else 0

import re

def extract_number(text):
    """
    Extracts the numeric interval (in days) and its unit from strings like:
    - '21-day cycle'
    - '21- to 28-day cycle'
    - '3-week schedule'

    Returns:
        (int days, str original_unit) or (None, str original_text) if unparseable
    """
    text = str(text).lower().strip()

    # Match either "21-day", or "21- to 28-day"
    match = re.search(r"(\d+)\s*-\s*(?:to\s*)?(\d+)?-?\s*([a-z]+)", text)
    if match:
        num = int(match.group(1))
        unit = match.group(3)

        if 'day' in unit:
            days = num
        elif 'week' in unit:
            days = num * 7
        elif 'month' in unit:
            days = num * 30
        elif 'year' in unit:
            days = num * 365
        else:
            return None, text  # Unknown unit

        return (days, unit)

    # Fallback: just 21 day in case 21- to 28-day ...
    match = re.search(r"(\d+)\s*([a-z]+)", text)
    if match:
        num = int(match.group(1))
        unit = match.group(2)

        if 'day' in unit:
            days = num
        elif 'week' in unit:
            days = num * 7
        elif 'month' in unit:
            days = num * 30
        elif 'year' in unit:
            days = num * 365
        else:
            return None, text

        return (days, unit)

    return None, text  # Still unparseable



def get_idays(text):
    return list(map(int, re.findall(r"-?\d+", text))) if re.findall(r"-?\d+", text) else 0

def build_component_vector(idays: list, component: str, csig=0, debug=False) -> dict:
    """
    Build a binary vector for a component based on active days (idays).
    
    If csig == 0, infer vector length from day range.
    If csig > 0, build vector of length csig using idays positions.

    Returns:
        dict: {
            <component>: [0, 1, 0, ...],
            "tracker": {...}  # Only if debug=True
        }
    """
    output = {}
    tracker = {}

    if debug:
        tracker["received"] = {"idays": idays, "component": component, "csig": csig}

    try:
        if not csig:
            min_day = min(idays)
            max_day = max(idays)
            full_range = np.arange(min_day, max_day + 1)
            offset_idays = [day - min_day for day in idays]
            vec = np.sum([np.eye(1, len(full_range), k=idx)[0] for idx in offset_idays], axis=0)
            output[component] = list(vec.astype(int))
            if debug:
                tracker["range"] = list(full_range)
        else:
            vec = np.sum([np.eye(1, csig, k=day - 1)[0] for day in idays], axis=0)
            output[component] = list(vec.astype(int))
    except Exception as e:
        if debug:
            tracker["error"] = str(e)
            tracker["Failed"] = 1

    return {**output, **({"tracker": tracker} if debug else {})}



def collapse_event_matrix(event_string):
    components = sorted(event_string.keys())
    num_days = len(next(iter(event_string.values())))

    for k, v in event_string.items():
        if len(v) != num_days:
            raise ValueError(f"Component '{k}' has mismatched length.")

    # Create a unified event matrix of 1s where any drug is active
    unified_events = [0] * num_days
    for v in event_string.values():
        for i, val in enumerate(v):
            if val == 1:
                unified_events[i] = 1

    # Precompute all event days
    event_days = [i for i, val in enumerate(unified_events) if val == 1]

    tag_entries = []
    for day in event_days:
        active_names = sorted([comp for comp in components if event_string[comp][day] == 1])
        if active_names:
            tag_entries.append((day, active_names))

    if not tag_entries:
        return ""

    last_day = tag_entries[-1][0]
    shift = num_days - last_day  # Same logic as before

    output = []
    used_shift = False
    event_index = 0
    component_first_use = set()

    for day, names in tag_entries:
        main = names[0]

        if not used_shift:
            tag = f"{shift}.{main}"
            used_shift = True
        else:
            delta = event_days[event_index] - event_days[event_index - 1]
            tag = f"{delta}.{main}"

        output.append(tag)
        component_first_use.add(main)

        for name in names[1:]:
            tag = f"0.{name}"
            output.append(tag)
            component_first_use.add(name)

        event_index += 1

    if len(component_first_use) == 1:
        return ";".join(output + output)

    return ";".join(output)

def collapse_event_matrix_wrapper(event_string):
    # Remove zero-only components
    filtered = {k: v for k, v in event_string.items() if any(val != 0 for val in v)}
    if not filtered:
        return [""], {}

    results = []
    tracker = {}

    # Group components by exact sequence length
    length_groups = defaultdict(dict)
    for k, v in filtered.items():
        length_groups[len(v)][k] = v

    # Run collapse_event_matrix on each length group
    for length, group in length_groups.items():
        result = collapse_event_matrix(group)
        results.append(result)
        tracker[f"len_{length}"] = len(group)

    return results, tracker

def run_test():
    examples = [
        [
            {'Gemicitabine': [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Paclitaxel': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
              "14.Gemicitabine;0.Paclitaxel;7.Gemicitabine"
        ],
        [{'Docetaxel': [1, 0, 0, 0, 0, 0, 0]}, "7.Docetaxel;7.Docetaxel"],
        [{'Pembrolizumab': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Cisplatin': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Gemcitabine': [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        "15.Cisplatin;0.Gemcitabine;0.Pembrolizumab;6.Gemcitabine"],
        [
        {"Pemb": [0,0,0,1], "Adve":[1,0,0,0], "Beijign":[0,1,1,0], "Wara":[1,1,1,1]},
        "1.Adve;0.Wara;1.Beijign;0.Wara;1.Beijign;0.Wara;1.Pemb;0.Wara"
        ],
        [
        {
            "bend": [1,1,0,0,0,0,0,0,0,0,0,0], 
            "boro": [1,0,0,0,1,0,0,0,1,0,0,0]
        },
        "4.bend;0.boro;1.bend;3.boro;4.boro" # note - you need to build entire matrix to estimate deltas (should work accross components NOT per component)
        ],
        [
            {
            'Bendamustine': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            "27.Bendamustine;1.Bendamustine;27.Bendamustine;1.Bendamustine"
        ],
        [
            {
            'Bendamustine': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Bendamustine': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            "23.Bendamustine;5.Bendamustine;23.Bendamustine;5.Bendamustine" # ??? looks at it as individual but shouldnt!
        ]
    ]
    for example in examples:
        # print(example)
        rrr = collapse_event_matrix(example[0])
        assert rrr == example[1], f"not eq: \n{example[1]}\n{rrr}"

    # case with uneven things...
    event_string_example = {
        'Cyclophosphamide': [1] + [0]*13,
        'Trastuzumab': [1] + [0]*20,
        'Epirubicin': [1] + [0]*13,
        'Paclitaxel': [1] + [0]*6 + [1] + [0]*3 + [1] + [0]*9,
        'Pertuzumab': [1] + [0]*20
    }
    # print(event_string_example)
    res = collapse_event_matrix_wrapper(event_string_example)

    # print("Expected:\n")
    # Expected combinations in list:
    expected_not = [ 
    "10.Paclitaxel;0.Pertuzumab;0.Trastuzumab;7.Paclitaxel;4.Paclitaxel",
    "14.Cyclophosphamide;0.Epirubicin",
    "10.Cyclophosphamide;0.Epirubicin;0.Paclitaxel;0.Pertuzumab;0.Trastuzumab;7.Paclitaxel;4.Paclitaxel",
    "3.Cyclophosphamide;0.Epirubicin;0.Paclitaxel;0.Pertuzumab;0.Trastuzumab;7.Paclitaxel;4.Paclitaxel"
    ] 
    expected = [ 
    "10.Paclitaxel;0.Pertuzumab;0.Trastuzumab;7.Paclitaxel;4.Paclitaxel",
    "14.Cyclophosphamide;0.Epirubicin",
    ] 

    assert sorted(res[0]) == sorted(expected), f"Failed.\nobtained={res[0]}\nexpected={expected}"
    assert sorted(res[0]) != sorted(expected_not), f"Failed.\nobtained={res[0]}\nexpected={expected}"

    print("All tests passed!")


def run_test_topup():
    rows = [
        {
        "component" : "bendamustine",
        "allDays": "1,2",
        "cyclesigs":"28-day cycle for 6 cycles"
        },
        {
        "component" : "bendamustine",
        "allDays": "1,2",
        "cyclesigs":"Full cycle"
        },
        {
        "component" : "bendamustine",
        "allDays": "None",
        "cyclesigs":"28-day cycle"
        }
    ]

    for row in rows:
        drug = str(row["component"]).strip().replace(" ", "").lower().capitalize()
        cycsigs_int = extract_number(row["cyclesigs"])

        if row['allDays']:
            idays = get_idays(row['allDays'])
        else:
            idays = [None]
            
        vector = build_component_vector(idays, drug, cycsigs_int)
        print("Input row:", row)
        print(vector)



run_test()
# run_test_topup()