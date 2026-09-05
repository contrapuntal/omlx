# SPDX-License-Identifier: Apache-2.0
"""Native Unlimited-OCR ring cache on oMLX's serial generation lane.

Import lazily after the Unlimited-OCR compatibility loader has run: the pinned
mlx-vlm may obtain the native model package from oMLX's vendor namespace.
"""

from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache


class OMLXRingSlidingKVCache(RingSlidingKVCache):
    """Keep the native attention layout; reject lossy batch conversions."""

    @classmethod
    def merge(cls, caches):
        if len(caches) != 1:
            raise ValueError("Unlimited-OCR ring caches require serial requests")
        return caches[0]

    def to_batch(self, left_padding):
        if list(left_padding) != [0]:
            raise ValueError(
                "Unlimited-OCR ring caches require serial unpadded requests"
            )
        return self

    def filter(self, batch_indices):
        indices = list(batch_indices)
        if indices == [0]:
            return
        if not indices:
            self.keys = self.values = None
            self.offset = 0
            self.prefill_length = None
            self._ring_pos = 0
            return
        raise ValueError("Unlimited-OCR ring caches require serial requests")

    def extract(self, idx):
        if int(idx) != 0:
            raise IndexError("Unlimited-OCR ring cache only has row 0")
        return self

    def extend(self, other):
        raise ValueError("Unlimited-OCR ring caches require serial requests")

    def is_trimmable(self):
        # Prefill prefixes are contiguous; decode has overwritten history.
        return self.prefill_length is None

    def trim(self, n):
        if not self.is_trimmable():
            raise ValueError("Cannot trim a decoded Unlimited-OCR ring cache")
        return super().trim(n)
