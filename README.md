# Solve the Bannerlord version of the Tablut with AlphaZero's method
## Introduction
在开始进行DRL的学习时，我就想要在最后解决Bannerlord中的古代棋。其中我对Tablut，也叫板棋或古典象棋比较感兴趣，因为其他的棋随机性比较强不太会玩。最后决定利用AlphaZero来解决，然后发现了alphazero-general这个项目，决定基于这个项目实现，但后来才发现这个项目对于AlphaZero实现可以说是漏洞百出。具体这个原始项目存在的问题和修改我放在[这里](/docs/azg_problems.md)，对于整个项目每个文件和函数的功能实现介绍我放在[这里](/docs/Explaination.md)，对于我的参数设置的说明我放在了[这里](/docs/Parameters_Management.md)，对于整个项目逐步修改的日志我放在了[这里](/docs/project_logs.md)。
## Results Display
Bannerlord中Tablut的规则是这样：
1. 9&times;9棋盘，黑方（进攻方）16子对白方（防守方）8子。王从中宫起步；防守方把王走到任意边格即胜，进攻方吃掉王或让所有防守方无子可动即胜。
2. 所有棋子（包括王）直线走任意格，不可跳子；任何棋子都不能走到或穿过中心格（王座）。王一旦离开中心格，也不能再跨越它。
3. 基本吃法是两侧夹击。此外，可以把敌子"顶在中心格上"吃掉，也就是把对方夹在己子与中心格之间（这一点双方都适用）。
4. 王在王座上仍然按普通两侧夹杀被吃（并没有"必须四面合围"的特别条款）。
5. 王不在王座上即王座为空，黑子可把白兵夹在"己子和空王座"之间而吃掉。王仍在王座上：敌方不能把王座当作"墙"参与夹杀旁兵。

这是我执白和训练的模型对弈。 
![Demo](docs/demo_az_h.gif)
这是训练的模型执白和2层alphabeta树进行对弈。
![Demo](docs/demo_az_2ab.gif)
这是训练模型的自博奕。
![Demo](docs/demo_az_self.gif)

## Usage
### Environment and Configuration
```bash
conda env create -f docs/environment.yml
pip install -r docs/requirements.txt
pip install -e .
```
### Train
参数设置在src/tablut/Args.py中，具体docs/Explaination.md中提供了参数的功能介绍，docs/Parameters_Management.md提供了我设置参数的经验，运行mainTafl.py来开始训练。
### Compare
可以在pitTafl.py中对各个模型统一进行比较。
### GUI play
项目提供了通过打印棋盘可视化的基础实现，只要在Arena参数中提供显示函数，tablut.rules.TaflGame.display提供了一个，同时在playGames设置verbose为True。即可显示，这里如果要人机对弈，需要手动输入四个坐标，原x,y，目的x,y。其中视角会变化，即自己的棋子是白色的，而地方的是黑色的。
另外在ui中实现了更成熟的可视化，可以运行ui/run.py根据调节实现不同对象的对弈或者和模型下棋。
### other tools
#### checkpoints
项目提供了比较完整的恢复训练功能，在每一轮训练结束后会保存元数据，在中途会记录训练数据和网络权重，如果在参数中设置load_model为True，能够自动恢复记录了元数据的每一轮，同时checkpoint给出了存储的位置，load_model设置了加载的权重文件，如果开始没有进化，则没有最佳可以加载则需要更改名字到记录的temp权重。
#### Elo
项目提供了对于给定模型Elo分数计算的功能，在baselines中，默认以2层alphabeta树模型为1500分基准计算。
#### other agents
baselines中除了原项目提供的随机和贪心模型，我也实现了新的随机模型和贪心模型，同时我们使用了更好的启发式函数，也提供了一个alphabeta树模型。
#### alphabeta teacher data
由于我的硬件资源和时间资源受限，所以我使用了一种加速训练的方法，即在训练开始前提供2层alphabeta树自博奕得到的训练数据，如果你追求纯粹的自博奕训练，可以设置ab_data为0则相关代码都不会调用，如果使用其值为先获得几轮alphabeta产生的数据，同时在最佳模型性能没有超过老师前是不会删除老师提供的数据的。在资源和时间充足的情况下我推荐还是不要使用这个功能，其会有代价，后面会提到。
## Others
### Train Log Analysis
![alt text](Elo.png)
我的训练设计总共是30轮，但是在15，16局已经能够全胜alphabeta树，最后也仅仅训练到20轮。Elo分数设置是32局，这里分数存在波动，同时之后存在分数的下降。我的理解是因为我使用了1轮2层alphabeta树自博奕的数据。所以开始学习的比较快，但是相应的也有副作用，比如数据有巨大的区别，而老师数据的占比是1/i，逐步下降，同时在Elo分数超过2层alphabeta后会删除老师数据，造成的结果就应该是网络的能力是受到了alphabeta先验的影响，其能力应该有一定专门的优化，污染了自博奕进化的进程，所以在老师数据比例有巨大变化，甚至删除后网络的学习会有巨大影响，甚至会遗忘学习到的知识，导致能力下降。上面我提供的权重文件是手动保存的第二个全胜2层alphabeta树的权重，它在64轮比较中能够49胜15负第一个权重，但是演示都是用第一个做的。

## Upstream & Acknowledgments
本项目改造自 [Alpha Zero General](https://github.com/suragnair/alpha-zero-general)（MIT License）。该仓库提供了通用的自博弈强化学习框架（训练循环 `Coach.py`、搜索 `MCTS.py`、Othello 示例与教程）。向原作者与贡献者致谢。

### Citation
若引用上游工作，请使用其 README 中的 BibTeX：
@misc{thakoor2016learning,
  title={Learning to play othello without human knowledge},
  author={Thakoor, Shantanu and Nair, Surag and Jhunjhunwala, Megha},
  year={2016},
  publisher={Stanford University, Final Project Report}
}

## Attribution
This project is a fork of [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) (MIT License).  
Original work © Surag Nair and contributors (see upstream LICENSE).  
Modifications © 2025 LiZhiZheng. See [LICENSE](./LICENSE) for details.
