import math

import numpy as np

EPS = 1e-8

from tablut.utils.log import logger

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    # 控制总流程，进行MCTS，然后根据temp选择
    def getActionProb(self, canonicalBoard, temp=1, noise_s=True):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        #for i in range(self.args.numMCTSSims):
        #    self.search(canonicalBoard)


        root_s = self.game.stringRepresentation(canonicalBoard)
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)
            # 在根扩展后（首轮）注入 Dirichlet 噪声
            if (i == 0 and noise_s and getattr(self.args, "noise_eps", 0) > 0 and getattr(self.args, "dirichlet_alpha", 0) > 0 and root_s in self.Ps):
                valids = self.Vs[root_s].astype(np.bool_)
                if valids.sum() > 0:
                    K = int(valids.sum())
                    alpha0 = 10.0
                    alpha = alpha0 / max(1, K)
                    noise = np.zeros_like(self.Ps[root_s], dtype=np.float64)
                    noise[valids] = np.random.dirichlet([alpha]*K)
                    self.Ps[root_s] = (1 - self.args.noise_eps)*self.Ps[root_s] + self.args.noise_eps*noise
                    ssum = np.sum(self.Ps[root_s])
                    if ssum > 0:
                        self.Ps[root_s] /= ssum



        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    # 整个流程是先获取要计算u需要的值，如果不存在的补一个初值，然后计算出所有动作的u值，
    # 然后得到最大的，再选择值最大的动作，给环境跑这个动作，递归的跑，然后得到返回值后更新相应的边的值
    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)
        p = canonicalBoard.getPlayerToMove()

        # 获取终局值的初值
        if s not in self.Es: # 如果这个状态没有记录过终局值
            self.Es[s] = self.game.getGameEnded(canonicalBoard, p) # 记录一次值
        # 检测跳出递归条件
        if self.Es[s] != 0: # 不为0，结束
            # terminal node
            return -self.Es[s]

        # 获取初始的pi
        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard) # 用网络预测初值
            valids = self.game.getValidMoves(canonicalBoard, p) # 得到合法动作
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves # 筛选合法动作
            sum_Ps_s = np.sum(self.Ps[s]) # 得到所有合法动作值之和
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize # 算每个合法动作的百分比
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                logger.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids # 记录
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        legal = [a for a in range(self.game.getActionSize()) if valids[a]]
        N_sum = self.Ns[s]
        pb_c = math.log((N_sum + self.args.cpuct_c_base + 1) / self.args.cpuct_c_base) + self.args.cpuct_c_init
        # pick the action with the highest upper confidence bound
        for a in legal:
            if (s, a) in self.Qsa:
                u = pb_c * self.Ps[s][a] * math.sqrt(N_sum) / (1 + self.Nsa.get((s,a),0))
                u = self.Qsa.get((s,a),0) + u
                #u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / ( # 综合数据对于是否选这条边的评估
                #        1 + self.Nsa[(s, a)])
            else:
                u = pb_c * self.Ps[s][a] * math.sqrt(N_sum) / (1 + self.Nsa.get((s,a),0))
                #u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, p, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1) # 边的平均回报的更新
            self.Nsa[(s, a)] += 1 # 边的经过次数

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
