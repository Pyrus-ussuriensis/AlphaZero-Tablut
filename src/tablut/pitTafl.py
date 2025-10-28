
# Note: Run this file from Arena directory (the one above /tafl)

from tablut import Arena
from tablut.models.MCTS import MCTS
from tablut.rules.TaflGame import TaflGame, display
from tablut.baselines.TaflPlayers import *
from tablut.models.Players import MCTSPlayer
from tablut.models.NNet import NNetWrapper as nn
from tablut.Args import args
#from tafl.keras.NNet import NNetWrapper as NNet

import numpy as np
from tablut.utils.utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = TaflGame("Tablut")

# all players
rp = RandomTaflPlayer(g).play
gp = GreedyTaflPlayer(g).play
hp = HumanTaflPlayer(g).play

from tablut.baselines.alphabeta_player import AlphaBetaTaflPlayer
from tablut.baselines.greedy_player import GreedyTaflPlayer
from tablut.baselines.random_player import RandomPlayer

# nnet players
#n1 = NNet(g)
#n1.load_checkpoint('./pretrained_models/tafl/keras/','6x100x25_best.pth.tar')
#args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
#mcts1 = MCTS(g, n1, args1)
#n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
a, b, c = AlphaBetaTaflPlayer(g,2), GreedyTaflPlayer(g), RandomPlayer(g)

#arena = Arena.Arena(a, b, g, display=display)

#print(arena.playGames(2, verbose=True))
nnet = nn(g)
pnet = nn(g)
nnet.load_checkpoint(args.checkpoint, args.load_folder_file)
pnet.load_checkpoint(args.checkpoint, args.load_folder_file)
pmcts_player = MCTSPlayer(g, pnet, args, temp=0)
nmcts_player = MCTSPlayer(g, nnet, args, temp=0)

n = 100
def test_model(a, b, c, g, n): # 让a和b,c在g上处理n次
    arena = Arena.Arena(a, b, g)
    print(arena.playGames(n, verbose=False))
    arena = Arena.Arena(a, c, g)
    print(arena.playGames(n, verbose=False))
    arena = Arena.Arena(b, c, g)
    print(arena.playGames(n, verbose=False))

test_model(nmcts_player,pmcts_player,c,g,n)
from tablut.baselines.Elo_Cal import Evaluate_Model_with_Alpha_Beta
#print(Evaluate_Model_with_Alpha_Beta(new_model=a, g=g))
#print(Evaluate_Model_with_Alpha_Beta(new_model=b, g=g))
#print(Evaluate_Model_with_Alpha_Beta(new_model=c, g=g))
#print(Evaluate_Model_with_Alpha_Beta(new_model=a, g=g, n=6))