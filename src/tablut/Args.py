from tablut.utils.utils import *


args = dotdict({
    # 迭代与数据
    'numIters': 20,                       # 总轮数
    #'numIters': 120,                       # 总轮数
    'numEps': 12,                          # 每轮自博弈局数（首轮样本稳过 batch）

    #'numEps': 160,                          # 每轮自博弈局数（首轮样本稳过 batch）
    #'numEps': 384,                          # 每轮自博弈局数（首轮样本稳过 batch）
    #'numEps': 300,                          # 每轮自博弈局数（首轮样本稳过 batch）
    'numItersForTrainExamplesHistory': 12, # 保留最近20轮
    'train_size': 32768,
    'step': 1000,
    'batch_size': 64,
    'maxlenOfQueue': 200_000,              # 经验窗口

    # MCTS
    'numMCTSSims': 32,    # 提升π质量

    #'numMCTSSims': 200,    # 提升π质量
    #'numMCTSSims': 128,    # 提升π质量
    #'numMCTSSims': 200,    # 提升π质量
    'cpuct': 1.5,
    'tempThreshold': 50,   # 前20手温度>0，其后=0

    # 评测/门控
    'arenaCompare': 64,    # 偶数，换边
    'evaluate': 32,       # 确认赛
    'updateThreshold': 0.54,

    # 存档
    'checkpoint': './experiment/5/',
    #'checkpoint': './experiment/3/',
    #'checkpoint': './experiment/4_test/',
    #'load_model': True,
    'load_model': False,
    'load_folder_file': 'best.pth.tar',
    'experiment': 5,
    #'experiment': 3,
    #'experiment': 4,

    # 根注噪（仅自博弈根）
    'dirichlet_alpha': 0.30,
    'noise_eps': 0.25,
    'limit': 100,
    'draw': -1e-6,
    'cpuct_c_base': 19652,
    'cpuct_c_init': 1.25,


    
})
