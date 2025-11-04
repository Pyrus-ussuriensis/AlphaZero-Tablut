from tablut.utils.log import writer
from tablut.Arena import Arena
from tablut.baselines.alphabeta_player import AlphaBetaTaflPlayer
def elo_vs_ab(W,D,L, base=1500):
    import math
    N=W+D+L
    s=(W+0.5*D)/N
    sp=(W+0.5*D+0.5)/(N+1)            # Haldane–Anscombe 平滑
    delta=400*math.log10(sp/(1-sp))   # 与 AB 的 Elo 差
    # 误差 频率学，delta-method 
    ex2 = (W + 0.25*D)/N              # E[X^2], X∈{1,0.5,0}
    var = ex2 - s*s
    se_s = (var/N)**0.5
    se_delta = (400/math.log(10))*se_s/(sp*(1-sp))
    ci = (delta-1.96*se_delta, delta+1.96*se_delta)
    return base+delta, delta, ci, s, sp # 基础加差，差，误差上下界，胜率，平滑胜率

def Evaluate_Model_with_Alpha_Beta(new_model, g, step=0, n=200, write=False, d=2):
    print(f"start calculating Elo scores base on {d} layers alpha-beta tree algorithm")
    arena = Arena(player1=new_model, player2=AlphaBetaTaflPlayer(game=g, depth=d), game=g)
    W, L, D = arena.playGames(num=n)
    elo, delta, (lo,hi), s, sp = elo_vs_ab(W,D,L, base=1500)
    if write:
        writer.add_scalar("eval/elo_vs_ab", elo, step)
        writer.add_scalar("eval/elo_delta", delta, step)
        writer.add_scalar("eval/elo_low", lo, step)
        writer.add_scalar("eval/elo_high", hi, step)
        writer.add_scalar("eval/score_score", s, step)
        writer.add_scalar("eval/score_score_smooth", sp, step)
        writer.add_scalar("eval/wins", W, step)
        writer.add_scalar("eval/draws", D, step)
        writer.add_scalar("eval/losses", L, step)
    print(f"the Elo score is {elo}")
    return elo
