class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]



def rot_xy(x, y, n, k):
    k %= 4
    if k == 0: return x, y
    if k == 1: return n-1-y, x
    if k == 2: return n-1-x, n-1-y
    if k == 3: return y, n-1-x
    return x, y

import numpy as np
from functools import lru_cache

def _rot_xy_vec(x, y, n, k):
    k %= 4
    if k == 0: return x, y
    if k == 1: return n-1-y, x
    if k == 2: return n-1-x, n-1-y
    return y, n-1-x

@lru_cache(maxsize=None)
def action_perms(n: int) -> np.ndarray:
    # 返回 shape=(8, n**4) 的置换表；顺序为 s=k*2+flip，pass 槽位保持不变
    a  = np.arange(n**4)
    x1 =  a % n
    y1 = (a // n)    % n
    x2 = (a // n**2) % n
    y2 = (a // n**3) % n

    perms = np.empty((8, n**4), dtype=np.int64)
    for k in range(4):
        rx1, ry1 = _rot_xy_vec(x1, y1, n, k)
        rx2, ry2 = _rot_xy_vec(x2, y2, n, k)
        for flip in (0, 1):
            fx1 = n-1-rx1 if flip else rx1
            fx2 = n-1-rx2 if flip else rx2
            new_idx = fx1 + ry1*n + fx2*(n**2) + ry2*(n**3)
            s = k*2 + flip
            perms[s] = new_idx
            #perms[s, -1] = n**4 - 1    
    return perms

def getNNImage(scalar, S, time):
    """
    返回 (C,H,W) float32，多平面语义图：
    [my_pawn, opp_pawn, my_king, opp_king, throne, (extra terrains...), side, rem_to_limit]
    约定：此函数在 **canonical 后的棋盘** 上调用（+1 为当前行动方）。
    """
    '''
    H, W = self.height, self.width
    # terrain*10 + piece 的“解码”
    scalar = np.zeros((H, W), dtype=np.int16)
    for x, y, typ in self.board:        # 地形编码（如 1=王座；可扩展）
        scalar[y, x] = typ * 10

    flip = bool(getattr(self, "_canon_flip", False))
    sign = -1 if flip else 1
    flip_king = bool(getattr(self, "_canon_flip_king", True))

    for x, y, typ in self.pieces:
        if x < 0:
            continue
        v = typ
        if sign == -1:
            if v in (-1, 1):
                v = -v
            elif v == 2 and flip_king:
                v = -2
        scalar[y, x] += v
    '''
    H, W = S, S

    # 用更稳健的可逆拆分：scalar = terrain*10 + piece，piece ∈ [-2..2]
    scalar = np.asarray(scalar, dtype=np.int16)  # 防止 int8 溢出/整除怪异
    piece  = ((scalar + 2) % 10) - 2            # 先还原 piece（保留符号，范围[-2..2]）
    terrain = (scalar - piece) // 10            # 再还原 terrain（0/1/2/...）
    # 王座检测：直接用 terrain==2；现在拆分已正确，不会再被负子值“降位”



    my_pawn  = (piece == +1).astype(np.float32)
    opp_pawn = (piece == -1).astype(np.float32)
    my_king  = (piece == +2).astype(np.float32)
    opp_king = (piece == -2).astype(np.float32)
    throne   = (terrain == 2).astype(np.float32)

    planes = [my_pawn, opp_pawn, my_king, opp_king, throne]

    LIMIT = 100 # 旧截止<=50
    rem_to_limit = np.full((H, W), float((LIMIT-time)/float(LIMIT)), dtype=np.float32)

    planes.append(rem_to_limit) 
    X = np.stack(planes, axis=0).astype(np.float32)            # (C,H,W) 6维
    return X

