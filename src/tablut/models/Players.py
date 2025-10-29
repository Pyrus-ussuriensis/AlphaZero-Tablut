# players.py
import numpy as np
from tablut.models.MCTS import MCTS

class MCTSPlayer:
    def __init__(self, game, nnet, args, temp=0, noise=False):
        self.game, self.nnet, self.args, self.temp, self.noise = game, nnet, args, temp, noise
        self.mcts = MCTS(game, nnet, args)

    def __call__(self, board):
        # board 已是 canonical
        probs = self.mcts.getActionProb(board, temp=self.temp, noise_s=self.noise)
        return int(np.argmax(probs))

    def startGame(self):   # Arena 每局都会调用
        self.mcts = MCTS(self.game, self.nnet, self.args)

    def endGame(self):     # 可选
        pass
