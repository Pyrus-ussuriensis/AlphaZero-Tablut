import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import torch, random

import numpy as np
from tqdm import tqdm

from tablut.Arena import Arena
from tablut.models.MCTS import MCTS
from tablut.models.Players import MCTSPlayer
from tablut.baselines.Elo_Cal import Evaluate_Model_with_Alpha_Beta
from tablut.utils.log import logger, writer
from tablut.utils.ThreefoldRepetition import ThreefoldRepetition
from tablut.Args import args

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.start = 1 # 默认初始化为1，但是加载是会加载到上次
        self.meta = None

    # 控制训练，输出训练的记录用于训练
    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        threefold = ThreefoldRepetition(k=3)
        threefold.add_and_check(self.game.BoardRepresentation(board))
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            valids = self.game.getValidMoves(canonicalBoard, 1).astype(np.float32)
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            pi = pi*valids
            s = pi.sum()
            if s>0:
                pi = pi/s
            else:
                pi = valids/(valids.sum()+1e-8)

            #sym = self.game.getSymmetries(canonicalBoard, pi)
            #for b, p in sym: # board, pi
            #    trainExamples.append([b.astype(np.float32), self.curPlayer, p, None]) # 添加了从board到矩阵的转化，是否数据能够被网络处理
            img2d = np.array(canonicalBoard.getImage(), dtype=np.int8)
            trainExamples.append((img2d, self.curPlayer, np.asarray(pi, np.float32), canonicalBoard.time, canonicalBoard.size))
            #trainExamples.append([canonicalBoard.astype(np.float32), self.curPlayer, pi, None]) # 添加了从board到矩阵的转化，是否数据能够被网络处理


            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, np.random.choice(len(pi), p=pi))
            #action = np.random.choice(len(pi), p=pi)
            #board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            # 以1为基准，然后看是否和1相等，其检测的结果就是原对象值
            r = self.game.getGameEnded(board, 1)
            # 记录状态，计数
            cnt = threefold.add_and_check(self.game.BoardRepresentation(canonicalBoard))
            if cnt:
                r = args.draw

            if r != 0:
                return [(x[0], x[2], r * (x[1]), x[3], x[4]) for x in trainExamples]

    # 总的学习流程，先训练，然后进行评估
    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """


        for i in range(self.start, self.args.numIters + 1):
            # bookkeeping
            logger.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                logger.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            #pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            #nmcts = MCTS(self.game, self.nnet, self.args)
            pmcts_player = MCTSPlayer(self.game, self.pnet, self.args, temp=0)
            nmcts_player = MCTSPlayer(self.game, self.nnet, self.args, temp=0)

            logger.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(pmcts_player, nmcts_player, self.game)
            #arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
            #              lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            logger.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))

            # Tensorboard记录得分率和胜率
            score_rate = float(nwins+draws/2)/(self.args.arenaCompare)            
            writer.add_scalar("self/score_rate", score_rate, i)
            win_rate = float(nwins) / (pwins + nwins) if (pwins+nwins) > 0 else float('nan')
            writer.add_scalar("self/win_rate", win_rate, i)

            if pwins + nwins == 0 or win_rate < self.args.updateThreshold:
                logger.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                logger.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                Evaluate_Model_with_Alpha_Beta(new_model=nmcts_player, g=self.game, step=i, n=self.args.evaluate, write=True)
            self.save_iteration_checkpoints(i)

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        #modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        #examplesFile = modelFile + ".examples"
        folder = self.args.checkpoint
        if self.meta == None:
            self.load_iteration_checkpoints()
        examplesFile = os.path.join(folder, self.getCheckpointFile(self.meta['i']-1) + ".examples")
        if not os.path.isfile(examplesFile):
            logger.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            logger.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            logger.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True


    def save_iteration_checkpoints(self, i):
        parameters = {
            "i":i, # 当前轮数
            "writer_path":writer.log_dir,
        }
        torch.save(parameters, os.path.join(self.args.checkpoint, "resume.pt"))
    
    def load_iteration_checkpoints(self):
        meta = torch.load(os.path.join(self.args.checkpoint, "resume.pt"), map_location="cpu")
        self.start = meta["i"] + 1
        self.meta = meta
        return meta


