from tqdm import tqdm
from tablut.utils.log import logger
from tablut.utils.ThreefoldRepetition import ThreefoldRepetition



class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None, on_step=None, on_end=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.on_step = on_step
        self.on_end = on_end

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0

        for player in players[0], players[2]:
            if hasattr(player, "startGame"):
                player.startGame()

        # 不断跑，取动作，验证合理，跑一次 显示
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            '''
            board = self.game.getCanonicalForm(board, curPlayer)
            
            action = players[curPlayer + 1](board)
            #board = self.game.getCanonicalForm(board, curPlayer)
            valids = self.game.getValidMoves(board, 1)
            '''
            #valids = self.game.getValidMoves(board, 1)

            # --- 决策端：AI 用 canonical；人类用原始 ---
            cboard = self.game.getCanonicalForm(board, curPlayer)
            player_fn = players[curPlayer + 1]
            expects_canon = getattr(player_fn, "expects_canonical", True)
            action = player_fn(cboard) if expects_canon else player_fn(board)
            # --- 合法性校验：与上面所用棋盘保持一致 ---
            if expects_canon:
                valids = self.game.getValidMoves(cboard, 1)
            else:
                valids = self.game.getValidMoves(board, curPlayer)

            if valids[action] == 0:
                logger.error(f'Action {action} is not valid!')
                logger.debug(f'valids = {valids}')
                assert valids[action] > 0

            # Notifying the opponent for the move
            opponent = players[-curPlayer + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(board, action)
            
            # --- 可视化：在推进状态前，抛出“当前局面 + 动作” ---
            if callable(self.on_step):
                try:
                    self.on_step(board, curPlayer, action, it)
                except Exception:
                    pass

            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        for player in players[0], players[2]:
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        #return curPlayer * self.game.getGameEnded(board, curPlayer) # 总是以白方的视角记录结果
        result = curPlayer * self.game.getGameEnded(board, curPlayer)  # 以白方视角
        if callable(self.on_end):
            try:
                self.on_end(board, result, it)
            except Exception:
                pass
        return result

    # 按照给的盘数，跑一半，交换棋子，跑另一半，返回最终记录的值
    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        t1, t2, t3 = 0, 0, 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
                t1+=1
            elif gameResult == -1:
                twoWon += 1
                t2+=1
            else:
                draws += 1
                t3+=1
        print(f"1w,2w,3e {t1}, {t2}, {t3}") # 第一个赢，第二个赢，平局，因为循环而平局
        t1, t2, t3 = 0, 0, 0
        

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
                t1+=1
            elif gameResult == 1:
                twoWon += 1
                t2+=1
            else:
                draws += 1
                t3+=1
        print(f"1w,2w,3e {t1}, {t2}, {t3}") # 第一个赢，第二个赢，平局，因为循环而平局

        return oneWon, twoWon, draws
