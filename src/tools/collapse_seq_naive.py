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

    return ";".join(collapsed) 


def collapse(s: str) -> str:
    s = s.lower().strip()
    if not s.endswith(";"):
        s += ";"

    tokens = s.split(";")[:-1]
    n = len(tokens)

    if n == 1:
        return f"{tokens[0]};{tokens[0]};"

    joined = ";".join(tokens) + ";"

    for size in range(1, n // 2 + 1):
        if n % size != 0:
            continue
        unit = ";".join(tokens[:size]) + ";"
        if unit * (n // size) == joined:
            collapsed = tokens[:size]
            break
    else:
        collapsed = tokens

    if len(collapsed) == 1:
        collapsed *= 2

    return ";".join(collapsed) + ";"


def test():
    seq="27.Bendamustine;1.Bendamustine;27.Bendamustine;1.Bendamustine"
    print(collapse_naive(seq))

# test()