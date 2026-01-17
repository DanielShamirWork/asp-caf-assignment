import numpy as np
from pytest import fixture


@fixture
def random_payload(payload_size: int) -> np.ndarray:
    """Generate random payload data."""
    rng = np.random.default_rng(0xC0FFEE + payload_size)
    return rng.integers(0, 256, payload_size, dtype=np.uint8)


@fixture
def payload(payload_size: int, payload_type: str) -> np.ndarray:
    """Generate payload data based on the payload type.
    
    Args:
        payload_size: Size of the payload in bytes
        payload_type: Type of payload to generate:
            - 'random': Uniformly distributed random bytes
            - 'repetitive': 90% one value, 10% random variations
            - 'uniform': All same byte value (best compression)
    
    Returns:
        numpy array of uint8 values
    """
    if payload_type == 'random':
        rng = np.random.default_rng(0xC0FFEE + payload_size)
        return rng.integers(0, 256, payload_size, dtype=np.uint8)
    elif payload_type == 'repetitive':
        # 90% of one value, 10% random
        rng = np.random.default_rng(0xBEEF + payload_size)
        data = np.full(payload_size, ord('A'), dtype=np.uint8)
        num_variations = max(1, payload_size // 10)
        if payload_size > 0:
            variation_indices = rng.choice(payload_size, min(num_variations, payload_size), replace=False)
            data[variation_indices] = rng.integers(0, 256, len(variation_indices), dtype=np.uint8)
        return data
    elif payload_type == 'uniform':
        # All same byte (best compression)
        return np.full(payload_size, 42, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown payload_type: {payload_type}")
