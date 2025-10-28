from collections import Counter 
import hashlib
import pickle

import numpy as np


class ThreefoldRepetition:
    def __init__(self, k: int = 3):
        self.k = k
        self.cnt = Counter()

    def reset(self):
        self.cnt.clear()

    def add_and_check(self, layout_bytes: bytes) -> bool:
        h = self.key_of(layout_bytes)
        self.cnt[h] += 1
        return self.cnt[h] >= self.k
        
    def _to_bytes(self, x):
        return x.encode("utf-8")
   
    def key_of(self, state):
        b = self._to_bytes(state)
        return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


