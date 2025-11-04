# tools/run_viewer.py
from tablut.rules.TaflGame import TaflGame
from tablut.models.NNet import NNetWrapper as NNet
from tablut.models.MCTS import MCTS
from tablut.models.Players import MCTSPlayer
from tablut.ui.viewer import PygameObserver, HumanGUIPlayer
from tablut.Args import args
from tablut.Arena import Arena
from tablut.baselines.alphabeta_player import AlphaBetaTaflPlayer

def make_ai(game, ckpt, name):
    nnet = NNet(game)
    nnet.load_checkpoint(ckpt, name)
    return MCTSPlayer(game, nnet, args, temp=0, noise=False)

if __name__ == "__main__":
    game = TaflGame("Tablut")
    # 选择模式：
    mode = "human-vs-ai"  # or "ai-vs-ai"
    #mode = "ai-vs-ai"  # or "ai-vs-ai"
    checkpoint = "./store/6_best_2225_0/"
    checkpoint1 = "./store/6_20/"
    name = "best.pth.tar"

    #obs = PygameObserver(game, delay_ms=300, step_mode=False)
    obs = PygameObserver(game, delay_ms=200, step_mode=False,
                     record=True, out_path="./store/videos/replay_alphazero_self.gif", fps=30)


    if mode == "ai-vs-ai":
        p1 = make_ai(game, checkpoint, name)
        p2 = make_ai(game, checkpoint, name)  # 或者别的模型
        #p2 = AlphaBetaTaflPlayer(game,2)

    else:
        p1 = HumanGUIPlayer(obs)            # 人类执白
        p2 = make_ai(game, checkpoint, name="best.pth.tar")  # AI 执黑

    arena = Arena(p1, p2, game, on_step=obs.on_step, on_end=obs.on_end)
    arena.playGame(verbose=False)  # 单局；也可循环多局
