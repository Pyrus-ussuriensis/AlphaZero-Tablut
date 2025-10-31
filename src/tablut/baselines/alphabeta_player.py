import numpy as np
from math import pow
from tablut.baselines.greedy_player import _eval_board
from tablut.utils.Digits import base2int

PASS = lambda n: n**4 - 1

class AlphaBetaTaflPlayer:
    def __init__(self, game, depth=2):
        self.game = game
        self.depth = depth
        self.size = 9
        self.actionsize = pow(9,4)
        self.temp=1

    def startGame(self): pass
    def endGame(self):   pass

    # 工具函数
    def _legal_actions(self, board, cur):
        raw = board.get_legal_moves(cur)              # [(x1,y1,x2,y2),...]
        n = self.game.n
        return [base2int([x1,y1,x2,y2], n) for (x1,y1,x2,y2) in raw]

    def _term_origin(self, b, origin):
        r = self.game.getGameEnded(b, origin)         # 直接以 origin 视角判终局
        return None if r == 0 else r * 1e6

    def _eval_origin(self, b, origin):
        bc = self.game.getCanonicalForm(b, origin)    # 规范到 origin 视角
        return _eval_board(self.game, bc)             # bc中传入了我们的视角评估

    # αβ
    def _ab(self, board, cur, origin, d, alpha, beta):
        tv = self._term_origin(board, origin)
        if tv is not None: # 终局 origin 视角值
            return tv, None
        if d == 0: # 叶子评估 origin 视角值
            return self._eval_origin(board, origin), None

        pool = self._legal_actions(board, cur)
        if not pool:
            # 保险：无子可走应已由终局捕获
            return self._term_origin(board, origin) or -1e9, None

        best_a = None
        if cur == origin: # MAX（同阵营）
            v = -1e18
            for a in pool:
                nb, nxt = self.game.getNextState(board, cur, int(a))
                sv, _ = self._ab(nb, nxt, origin, d-1, alpha, beta)
                if sv > v: v, best_a = sv, int(a)
                if v > alpha: alpha = v
                if alpha >= beta: break
            return v, best_a
        else: # MIN（对手阵营）
            v =  1e18
            for a in pool:
                nb, nxt = self.game.getNextState(board, cur, int(a))
                sv, _ = self._ab(nb, nxt, origin, d-1, alpha, beta)
                if sv < v: v, best_a = sv, int(a)
                if v < beta: beta = v
                if alpha >= beta: break
            return v, best_a

    def __call__(self, board):
        origin = board.getPlayerToMove()
        _, a = self._ab(board, origin, origin, self.depth, -1e18, 1e18)
        if a is None:
            ms = self._legal_actions(board, origin)
            return int(ms[0]) if ms else 0
        return int(a)

    def getActionProb(self, board):
        origin = board.getPlayerToMove()
        A = self.game.getActionSize()
        pi = np.zeros(A, dtype=np.float32)

        legal = self._legal_actions(board, origin)
        if not legal:
            return pi  # 全 0；上游会用 valids 兜底

        # 先看有没有“一步终结”儿子：直接一热，效率高且数值稳定
        winning, drawing = [], []
        for act in legal:
            nb, nxt = self.game.getNextState(board, origin, int(act))
            tr = self._term_origin(nb, origin)
            if tr is None: 
                continue
            if tr > 0: winning.append(act)
            elif tr == 0: drawing.append(act)

        if winning:
            for a in winning: pi[a] = 1.0 / len(winning)
            return pi
        # 可选：若只有和棋儿子，把它们均分
        if drawing and len(drawing) == len(legal):
            for a in drawing: pi[a] = 1.0 / len(drawing)
            return pi

        # 否则，对每个合法子做一次 αβ（深度-1，因为已落一手）
        scores = []
        for act in legal:
            nb, nxt = self.game.getNextState(board, origin, int(act))
            sv, _ = self._ab(nb, nxt, origin, max(0, self.depth-1), -1e18, 1e18)
            scores.append((act, sv))

        # 数值稳定的 softmax（带温度 self.temp）
        acts, vals = zip(*scores)
        vals = np.array(vals, dtype=np.float64)
        # 抑制巨幅差异（去掉终局哨兵后仍可能很大）
        vmax = vals.max()
        logits = (vals - vmax) / max(1e-6, self.temp)
        w = np.exp(np.clip(logits, -50, 50))
        w_sum = w.sum()
        if w_sum <= 0 or not np.isfinite(w_sum):
            # 退化保护：均匀分配到合法子
            p = 1.0 / len(legal)
            for a in legal: pi[a] = p
            return pi

        for a, p in zip(acts, w / w_sum):
            pi[int(a)] = float(p)
        return pi
