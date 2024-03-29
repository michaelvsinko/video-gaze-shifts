import numpy as np


def round_dict(dictionary: dict, n_digits: int = 5) -> dict:
    return {key: round(dictionary[key], n_digits) if not np.isnan(dictionary[key]) else 0.0 for key in dictionary}


def convert_pt(point: np.ndarray) -> tuple[int, int]:
    return tuple(np.round(point).astype(np.int).tolist())
