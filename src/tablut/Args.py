from tablut.utils.utils import *


args = dotdict({
    # 迭代与数据
    'numIters': 120,                       # 总轮数
    'numEps': 384,                          # 每轮自博弈局数（首轮样本稳过 batch）
    #'numEps': 300,                          # 每轮自博弈局数（首轮样本稳过 batch）
    'numItersForTrainExamplesHistory': 20, # 保留最近20轮
    'maxlenOfQueue': 200_000,              # 经验窗口

    # MCTS
    'numMCTSSims': 128,    # 提升π质量
    #'numMCTSSims': 200,    # 提升π质量
    'cpuct': 1.5,
    'tempThreshold': 50,   # 前20手温度>0，其后=0

    # 评测/门控
    'arenaCompare': 64,    # 偶数，换边
    'evaluate': 32,       # 确认赛
    'updateThreshold': 0.54,

    # 存档
    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': 'best.pth.tar',
    'experiment': 1,

    # 根注噪（仅自博弈根）
    'dirichlet_alpha': 0.30,
    'noise_eps': 0.25,
    'limit': 100,
    'draw': -1e-6,

    
})
