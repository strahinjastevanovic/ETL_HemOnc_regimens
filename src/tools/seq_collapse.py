def collapse_naive(s):
    """
    Dumbest regimen string processor:
    1. Lowercase + trailing semicolon
    2. Find shortest repeating prefix that reconstructs the full string
    3. If only one token → duplicate it
    4. If collapsed output has only one token → duplicate it
    5. Return the collapsed string (min-2-rule)
    """
    s = s.lower().strip()
    if not s.endswith(";"):
        s += ";"

    tokens = s.split(";")[:-1]
    n = len(tokens)

    # Rule: if only one token, duplicate
    if n == 1:
        return f"{tokens[0]};{tokens[0]};"

    # Try every prefix from size 1 up to full length
    for size in range(1, n + 1):
        chunk = tokens[:size]
        if chunk * (n // size) == tokens[:size * (n // size)] and n % size == 0:
            collapsed = chunk
            break
    else:
        collapsed = tokens  # fallback

    # Enforce min-two-token rule
    if len(collapsed) == 1:
        collapsed = collapsed * 2

    return ";".join(collapsed) + ";" 


def filter_et(s: str) -> str:
    """
    Filter out entries with 0 positions and resolve multi-cycle information to 1D.
    
    Input example: 
        "7.daratumumab@len15;0.daratumumab@len22;0.daratumumab@len28;7.daratumumab@len22;7.daratumumab@len15;0.daratumumab@len22;7.daratumumab@len22"
    
    Output example:
        "7.daratumumab;7.daratumumab;7.daratumumab;7.daratumumab;"
    
    Logic:
    - Discard all parts starting with "0." (same drug same day)
    - Keep only <days>.<drug-name> part (remove everything after "@")
    - Add trailing semicolon for proper processing
    """
    if not s:
        return s
    
    # Split by semicolon
    parts = s.split(";")
    
    # Filter and clean each part
    filtered = []
    for part in parts:
        # Skip empty parts
        if not part:
            continue
        
        # Skip parts starting with "0."
        if part.startswith("0."):
            continue
        
        # Remove everything after "@" (dosage information like @len15)
        cleaned = part.split("@")[0]
        filtered.append(cleaned)
    
    # Join back with semicolon and add trailing semicolon
    return ";".join(filtered)