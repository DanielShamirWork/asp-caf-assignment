import numpy as np
from pytest import mark

from libcaf import BitReader


@mark.parametrize('payload_size', [
    2 ** 4,
    2 ** 8,
    2 ** 12,
    2 ** 16,
])
def test_bitreader(random_payload: np.ndarray, payload_size: int) -> None:
    reader = BitReader(random_payload, payload_size * 8)
    
    for i in range(payload_size * 8):
        expected_bit = (random_payload[i // 8] >> (7 - i % 8)) & 1
        assert reader.read(1) == expected_bit
        assert not reader.done()
        reader.advance(1)
    
    assert reader.done()
