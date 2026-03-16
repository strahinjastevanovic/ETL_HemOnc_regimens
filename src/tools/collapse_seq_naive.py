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

