"""Tien ich seed RNG thong nhat, dung o moi noi sinh du liệu/train mo hinh de dam bao
tai lap duoc (reproducibility) xuyen suot du an.
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Tra ve 1 np.random.Generator moi tu seed. Dung ham nay thay vi np.random.seed()
    global o moi noi trong repo, de cac lan sinh du lieu doc lap khong anh huong lan nhau.

    TODO(ban tu code): 1 dong, return np.random.default_rng(seed).
    """
    raise NotImplementedError
