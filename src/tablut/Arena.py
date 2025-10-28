from tqdm import tqdm
from tablut.utils.log import logger
from tablut.utils.ThreefoldRepetition import ThreefoldRepetition



class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
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
        threefold = ThreefoldRepetition(k=3)
        cnt = False
        threefold.add_and_check(self.game.BoardRepresentation(board))
        it = 0

        for player in players[0], players[2]:
            if hasattr(player, "startGame"):
                player.startGame()

        # 不断跑，取动作，验证合理，跑一次 显示

        while self.game.getGameEnded(board, curPlayer) == 0 and not cnt:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            board = self.game.getCanonicalForm(board, curPlayer)
            
            action = players[curPlayer + 1](board)
            #board = self.game.getCanonicalForm(board, curPlayer)
            valids = self.game.getValidMoves(board, 1)
            #valids = self.game.getValidMoves(board, 1)

            if valids[action] == 0:
                logger.error(f'Action {action} is not valid!')
                logger.debug(f'valids = {valids}')
                assert valids[action] > 0

            # Notifying the opponent for the move
            opponent = players[-curPlayer + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(board, action)

            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            cnt = threefold.add_and_check(self.game.BoardRepresentation(board))

        for player in players[0], players[2]:
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return curPlayer * self.game.getGameEnded(board, curPlayer) # 总是以白方的视角记录结果

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
        t1, t2, t3, t4 = 0, 0, 0, 0
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
                if gameResult == 0:
                    t4+=1
        print(f"1w,2w,3e,4c {t1}, {t2}, {t3}, {t4}") # 第一个赢，第二个赢，平局，因为循环而平局
        t1, t2, t3, t4 = 0, 0, 0, 0
        

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
                if gameResult == 0:
                    t4+=1
        print(f"1w,2w,3e,4c {t1}, {t2}, {t3}, {t4}")

        return oneWon, twoWon, draws
