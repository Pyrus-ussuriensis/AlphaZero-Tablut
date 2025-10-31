from tablut.Coach import Coach
from tablut.rules.TaflGame import TaflGame as Game
from tablut.models.NNet import NNetWrapper as nn
from tablut.utils.utils import *
from tablut.utils.log import init_logging
from tablut.Args import *



if __name__=="__main__":
    g = Game("Tablut")
    #g = Game(6)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.checkpoint, args.load_folder_file)

    c = Coach(g, nnet, args, ab_data=0)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
        c.load_iteration_checkpoints()
    c.learn()
