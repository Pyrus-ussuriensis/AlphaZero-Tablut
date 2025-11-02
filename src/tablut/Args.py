from tablut.utils.utils import *


args = dotdict({
    # 迭代与数据
    'numIters': 30,                       # 总轮数
    #'numIters': 120,                       # 总轮数
    #'numEps': 2,                          # 每轮自博弈局数（首轮样本稳过 batch）

    'numEps': 160,                          # 每轮自博弈局数（首轮样本稳过 batch）#####1
    #'numEps': 1,                          # 每轮自博弈局数（首轮样本稳过 batch）#####1
    #'numEps': 384,                          # 每轮自博弈局数（首轮样本稳过 batch）
    #'numEps': 300,                          # 每轮自博弈局数（首轮样本稳过 batch）
    'numItersForTrainExamplesHistory': 9, # 保留最近20轮
    #'train_size': 32768,
    'step': 1000,
    'batch_size': 64,
    'maxlenOfQueue': 200_000,              # 经验窗口

    # MCTS
    #'numMCTSSims': 3,    # 提升π质量
    'numMCTSSims': 200,    # 提升π质量#####2
    #'numMCTSSims': 128,    # 提升π质量
    #'numMCTSSims': 200,    # 提升π质量
    'cpuct': 1.5,
    #'tempThreshold': 50,   # 前20手温度>0，其后=0

    # 评测/门控
    'arenaCompare': 64,    # 偶数，换边#####3
    #'arenaCompare': 4,    # 偶数，换边#####3
    'evaluate': 32,       # 确认赛#####4
    #'evaluate': 2,       # 确认赛
    'updateThreshold': 0.54,

    # 存档
    'checkpoint': './experiment/6/',#####5
    #'checkpoint': './experiment/6/',#####5
    #'checkpoint': './experiment/3/',
    #'checkpoint': './experiment/4_test/',
    #'load_model': True,#####6
    'load_model': False,
    'load_folder_file': 'best.pth.tar',
    'experiment': 6,#####7
    #'experiment': 6,#####7
    #'experiment': 3,
    #'experiment': 4,

    # 根注噪（仅自博弈根）
    'dirichlet_alpha': 0.30,
    'noise_eps': 0.25,
    'limit': 100,
    'limit_progress': 60,
    'draw': -1e-6,
    'cpuct_c_base': 19652,
    'cpuct_c_init': 1.25,

    'ab_data': 1,


    
})
