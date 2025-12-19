import numpy as np
from pytest import fixture


@fixture
def random_payload(payload_size: int) -> np.ndarray:
    rng = np.random.default_rng(0xC0FFEE + payload_size)
    return rng.integers(0, 256, payload_size, dtype=np.uint8)

