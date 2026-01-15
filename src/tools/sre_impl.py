import re 
from typing import List, Tuple, Set, Dict, Optional
import numpy as np

def convert_to_days(num: float, unit: str, days_range:List[int]=None) -> int:
    num = float(num)
    unit = str(unit or "").lower()

    if "day" in unit:
        return int(num)
    if "week" in unit:
        return int(num * 7)
    if "month" in unit:
        return int(num * 30)
    if "year" in unit:
        return int(num * 365)
    if "indeterminate" in unit:
        if not days_range:
            raise ValueError("indeterminate unit but days_range missing")
        return int(max(int(x) for x in days_range))

    raise ValueError(f"Unhandled cycle_length unit: num={num}, unit={unit}")

def get_last_cycle(unique_list):
    """timing_sequence example: ["15,22,28", "30", "15,22"] -> returns 28"""
    return max([int(e) for e in [l for subli in [s.split(",") for s in unique_list] for l in subli]])

def get_idays(text):
    return list(map(int, re.findall(r"-?\d+", text))) if re.findall(r"-?\d+", text) else 0

# VECTOR

def build_component_vector_deprecated(idays: list, csig=0) -> dict:
    """
    Build a binary vector for a component based on integer days (idays).
    
    If csig == 0, infer vector length from day range.
    If csig > 0, build vector of length csig using idays positions.
    To impleent csiv != len(idays) 

    output: list = 0, 1, 0, ...]
    """

    if type(csig) != int:
        return ValueError(f"[ERR] unhandled csig: {csig}")
   
    vec = np.sum([np.eye(1, csig, k=day - 1)[0] for day in idays], axis=0)
    return vec.astype(int)

def build_component_vector(idays: List[int], csig: int) -> np.ndarray:
    """
    Build a binary vector of length csig from 1-based day indices.
    If idays contains values > csig, assume they are absolute-within-parent-cycle
    and shift them into 1..csig by subtracting (min(idays)-1).
    """
    if not isinstance(csig, int) or csig <= 0:
        raise ValueError(f"[ERR] bad csig: {csig}")

    if not idays or idays == 0:
        raise ValueError(f"[ERR] empty idays: {idays}")

    idays = [int(d) for d in idays]

    if min(idays) < 1:
        raise ValueError(f"[ERR] day index < 1 in idays: {idays}")

    # normalize if days don't fit the declared cycle length
    if max(idays) > csig:
        m = min(idays)
        idays = [d - (m - 1) for d in idays]

        if min(idays) < 1 or max(idays) > csig:
            raise ValueError(f"[ERR] cannot normalize idays into 1..{csig}: {idays}")

    vec = np.zeros(csig, dtype=int)
    for d in idays:
        vec[d - 1] = 1

    return vec


# MATRIX

def _get_variant_variant(
    variants: List[Tuple[str, np.ndarray]],
    i: int
) -> Tuple[Set[int], np.ndarray]:
    if i < len(variants):
        pos_str, vec = variants[i]
    else:
        pos_str, base_vec = variants[0] if variants else ("", np.array([0]))
        vec = np.zeros_like(base_vec)
    pos_set = set(map(int, pos_str.split(','))) if pos_str else set()
    return pos_set, vec

def _build_key_output(
    variants: List[Tuple[str, np.ndarray]],
    i: int,
    sorted_positions: List[int],
    position_to_len: Dict[int, int]
) -> np.ndarray:
    pos_set, vec = _get_variant_variant(variants, i)
    return np.concatenate([
        np.pad(vec, (0, position_to_len[pos] - vec.shape[0])) if pos in pos_set
        else np.zeros(position_to_len[pos], dtype=vec.dtype)
        for pos in sorted_positions
    ])


def _extract_position_lengths(
    input_dict: Dict[str, List[Tuple[str, np.ndarray]]],
    max_day_limit: int = 10000
) -> Tuple[bool, Tuple[List[int], Dict[int, int]]]:
    """Safely extract max vector length per position, with blockers on insane inputs."""
    position_to_len = {}

    for key, variants in input_dict.items():
        for idx, (pos_str, vec) in enumerate(variants):
            try:
                positions = list(map(int, pos_str.split(',')))
            except Exception as e:
                return False, (f"Invalid position string in key={key}, index={idx}: '{pos_str}' — {e}", {})

            for pos in positions:
                if pos > max_day_limit:
                    return False, (f"Position {pos} in key={key}, index={idx} exceeds max limit {max_day_limit}", {})
                vec_len = vec.shape[0]
                if vec_len > max_day_limit:
                    return False, (f"Vector length {vec_len} in key={key}, index={idx} exceeds max limit {max_day_limit}", {})
                position_to_len[pos] = max(position_to_len.get(pos, 0), vec_len)

    return True, (sorted(position_to_len), position_to_len)

def build_variant_outputs_numpy(
    input_dict: Dict[str, List[Tuple[str, np.ndarray]]]
) -> List[Dict[str, np.ndarray]]:
    sorted_positions, position_to_len = _extract_position_lengths(input_dict)
    max_depth = max((len(v) for v in input_dict.values()), default=1)

    return [
        {
            key: _build_key_output(variants, i, sorted_positions, position_to_len)
            for key, variants in input_dict.items()
        }
        for i in range(max_depth)
    ]


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
    shift = num_days - last_day  

    output = []
    used_shift = False
    event_index = 0
    component_first_use = set()

    for day, names in tag_entries:
        main = names[0]
        # Strip @cycleLen suffix if present
        main_clean = main.split("@")[0] if "@" in main else main

        if not used_shift:
            tag = f"{shift}.{main_clean}"
            used_shift = True
        else:
            delta = event_days[event_index] - event_days[event_index - 1]
            tag = f"{delta}.{main_clean}"

        output.append(tag)
        component_first_use.add(main_clean)

        for name in names[1:]:
            # Strip @cycleLen suffix if present
            name_clean = name.split("@")[0] if "@" in name else name
            tag = f"0.{name_clean}"
            output.append(tag)
            component_first_use.add(name_clean)

        event_index += 1

    if len(component_first_use) == 1:
        return ";".join(output + output)

    return ";".join(output)


def validate_and_split_variants(
    input_dict: Dict[str, List[Tuple[str, np.ndarray]]],
    *,
    allow_multipar_fallback: bool = True,
    logger: Optional[object] = None,
) -> List[Dict[str, Tuple[str, np.ndarray]]] | None:
    lengths = {k: len(v) for k, v in input_dict.items()}
    unique_lengths = set(lengths.values())

    if len(unique_lengths) == 1:
        n = next(iter(unique_lengths))
        return [{k: input_dict[k][i] for k in input_dict} for i in range(n)]

    if len(unique_lengths) == 2 and 1 in unique_lengths:
        long_key = max(lengths, key=lengths.get)

        if all(l == 1 or k == long_key for k, l in lengths.items()):
            n = lengths[long_key]
            return [
                {k: (input_dict[k][i] if k == long_key else input_dict[k][0]) for k in input_dict}
                for i in range(n)
            ]

        # multiple long keys => known "multipart timing spans" shape -> return None so caller can multipar_pad
        if allow_multipar_fallback:
            if logger is not None:
                logger.warning(f"[SRE] Mixed multipart pattern -> fallback (no split): {lengths}")
            return None

        # strict mode
        raise ValueError(f"Variant lengths mismatch: mixed variant pattern is invalid.\n{input_dict}")

    # truly obscure patterns (e.g. {1,2,3} depths) => raise
    raise ValueError(f"Unhandled variant depth pattern: {lengths}\n{input_dict}")


def pad_variant_dict(variant: Dict[str, Tuple[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    max_len = max(vec.shape[0] for _, vec in variant.values())
    return {
        k: np.pad(vec, (0, max_len - vec.shape[0]))
        for k, (_, vec) in variant.items()
    }

def _parse_cycles_clean(pos_str: str) -> List[int]:
    if not pos_str:
        return []
    return [int(x) for x in pos_str.split(",") if x]

def _join_cycles(cycles: List[int]) -> str:
    return ",".join(map(str, sorted(set(cycles))))

def normalize_multicycle_spans(
    input_dict: Dict[str, List[Tuple[str, np.ndarray]]]
) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    # fast-path: already normalized (every component has exactly 1 entry)
    # print(input_dict)
    if all(len(v) == 1 for v in input_dict.values()):
        return input_dict

    out: Dict[str, List[Tuple[str, np.ndarray]]] = {}

    for drug, entries in input_dict.items():
        buckets: Dict[int, Tuple[List[int], np.ndarray]] = {}

        for pos_str, vec in entries:
            v = np.asarray(vec, dtype=int)
            L = int(v.shape[0])
            cycles = _parse_cycles_clean(pos_str)

            if L not in buckets:
                buckets[L] = ([], v.copy())
            else:
                buckets[L] = (buckets[L][0], np.maximum(buckets[L][1], v))

            buckets[L][0].extend(cycles)

        if len(buckets) == 1:
            cycles, v = next(iter(buckets.values()))
            out[drug] = [(_join_cycles(cycles), v)]
        else:
            for L, (cycles, v) in buckets.items():
                out[f"{drug}@cycleLen{L}"] = [(_join_cycles(cycles), v)]

    return out

def multipar_padding(input_dict: Dict[str, List[Tuple[str, np.ndarray]]]) -> List[Dict[str, Tuple[str, np.ndarray]]]:
    """
    Fallback for multipart timing spans across components (multiple keys with len>1 after normalization).
    Builds ONE long timeline per component by repeating its per-cycle vector into the cycles listed
    in timing_sequence, producing a single-entry dict per component: {drug: ("", long_vec)}.

    This returns [single_variant_dict] so downstream pad+collapse works unchanged.
    """
    # infer days-per-cycle (block_len) and max cycle count
    max_cycle = 0
    block_len = 0

    for variants in input_dict.values():
        for pos_str, vec in variants:
            cycles = _parse_cycles_clean(pos_str)
            if cycles:
                max_cycle = max(max_cycle, max(cycles))
            v = np.asarray(vec)
            block_len = max(block_len, int(v.shape[0]))

    if max_cycle == 0 or block_len == 0:
        raise ValueError(f"multipar_padding: cannot infer max_cycle/block_len.\n{input_dict}")

    total_len = max_cycle * block_len

    out: Dict[str, Tuple[str, np.ndarray]] = {}
    for drug, variants in input_dict.items():
        full = np.zeros(total_len, dtype=int)

        for pos_str, vec in variants:
            v = np.asarray(vec, dtype=int)

            # enforce consistent per-cycle length inside this cycle_len bucket
            if v.shape[0] < block_len:
                v = np.pad(v, (0, block_len - v.shape[0]))
            elif v.shape[0] > block_len:
                v = v[:block_len]

            for c in _parse_cycles_clean(pos_str):
                s = (c - 1) * block_len
                e = s + block_len
                full[s:e] = np.maximum(full[s:e], v)  # OR-merge

        out[drug] = ("", full)

    return [out]

def collapse_event_matrix_wrapper(input_dict, logger=None):
    input_dict = normalize_multicycle_spans(input_dict)

    variant_dicts = validate_and_split_variants(input_dict, logger=logger)

    if variant_dicts is None:
        if logger is not None:
            lengths = {k: len(v) for k, v in input_dict.items()}
            logger.warning(f"[SRE] Mixed pattern -> multipar_padding: {lengths}")
        variant_dicts = multipar_padding(input_dict)

    results = []
    for vdict in variant_dicts:
        padded = pad_variant_dict(vdict)
        if not any(np.any(v) for v in padded.values()):
            raise ValueError(f"All components in variant are zero-only.\n{vdict}")
        results.append(collapse_event_matrix(padded))
    return results