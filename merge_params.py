def merge_params(p1: dict, p2: dict) -> dict:
    """
    Merges the fields in p2 into the fields in p1 and returns the result.
    If keys overlap, p2's values will overwrite p1's.
    """

    return {**p1, **p2}
