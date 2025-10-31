import numpy as np
from tablut.utils.utils import *
from tablut.utils.Digits import *
from functools import lru_cache
# 固定的方向表
DIRS = [(+1,0), (-1,0), (0,+1), (0,-1)]  # 右、左、下、上

def _actually_build_maps(n):
    M = 4*(n-1)
    A_small = n*n*M
    A_big   = n**4
    small2big = -np.ones(A_small, dtype=np.int64)
    big2small = -np.ones(A_big,   dtype=np.int64)

    for y1 in range(n):
        for x1 in range(n):
            origin = y1*n + x1
            for d,(dx,dy) in enumerate(DIRS):
                for step in range(1, n):
                    small = origin*M + d*(n-1) + (step-1)  # 固定编号：方向×步长
                    x2, y2 = x1+dx*step, y1+dy*step
                    if 0 <= x2 < n and 0 <= y2 < n:
                        big = base2int([x1,y1,x2,y2], n)   # 仅在盘内才有大动作
                        small2big[small] = big
                        big2small[big]   = small
                    # 越界则 small2big[small] 保持 -1（无效槽位）
    return small2big, big2small



@lru_cache(maxsize=None)
def build_maps(n: int):
    small2big, big2small = _actually_build_maps(n)  # 你现有的构造
    # 防止外部误改缓存内容
    small2big.setflags(write=False)
    big2small.setflags(write=False)
    return small2big, big2small


# logits_small: (B, A_small)  —— 你网络的输出
# small2big:   (A_small,) LongTensor
# A_big:       int = n**4
def expand_small_to_big_probs(p_small, small2big, n, renorm=True):
    # p_small: (B, A_small) 概率（log_softmax 后 exp）
    A_big = n**4
    # 获得Batchsize
    B     = p_small.size(0)
    dev   = p_small.device

    s2b = small2big.to(dev)                  # LongTensor[A_small]
    mask = (s2b >= 0)                        # 只取有效小槽位
    idx  = s2b[mask]                         # (K,)

    p_big = p_small.new_zeros((B, A_big))
    p_big.scatter_(1, idx.unsqueeze(0).expand(B, -1), p_small[:, mask])

    if renorm:                               # 防止小概率落在无效槽位导致总和<1
        s = p_big.sum(dim=1, keepdim=True).clamp_min(1e-8)
        p_big = p_big / s
    return p_big




# pi_big:   (B, A_big)  —— MCTS 目标分布
# big2small:(A_big,) LongTensor，未映射处为 -1
# A_small:  int = n*n*4*(n-1)
def compress_big_to_small_pi(pi_big, big2small, n):
    # pi_big: (B, A_big) 概率/一热
    dev = pi_big.device
    b2s = big2small.to(dev)
    mask = (b2s >= 0)
    idx  = b2s[mask]                          # (K,)
    picked = pi_big[:, mask]                  # (B,K)

    A_small = n**2*4*(n-1) #int(idx.max().item() + 1) if idx.numel() > 0 else 0
    pi_small = pi_big.new_zeros(pi_big.size(0), A_small)
    pi_small.scatter_add_(1, idx.unsqueeze(0).expand(pi_big.size(0), -1), picked)

    s = pi_small.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return pi_small / s


