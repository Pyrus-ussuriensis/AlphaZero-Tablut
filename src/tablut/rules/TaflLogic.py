import numpy as np
from tablut.rules.GameVariants import Tafl
from tablut.Args import args

class Board():


    def __init__(self, gv):
        self.size=gv.size  
        self.width=gv.size
        self.height=gv.size
        self.board=gv.board #[x,y,type]
        self.pieces=gv.pieces #[x,y,type]
        self.time=0
        self.done=0
        self._canon_flip = False        # 是否翻转（攻方视角）
        self._canon_flip_king = True    # 翻不翻王：True/False
        self.pieces_count = len(self.pieces)
        self.progress_count = 0

    def __str__(self):
        img = self.getImage()
        flat = ''.join(str(r) for v in img for r in v)
        return f"{self.getPlayerToMove()}|c{self.progress_count}|t{self.time}|d{self.done}|{flat}" # 增加了终局值，避免出现同样的局面因超时结束与没有超时的影响
        #return f"{self.getPlayerToMove()}|d{self.done}|{flat}" # 增加了终局值，避免出现同样的局面因超时结束与没有超时的影响

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return np.array(self.getImage())[index]

    def astype(self,t):
        return np.array(self.getImage()).astype(t)

    def getPBR(self): # get pure board representation
        img = self.getImage()
        flat = ''.join(str(r) for v in img for r in v)
        return f"{self.getPlayerToMove()}|{flat}" 


    def getCopy(self):
      gv=Tafl()
      gv.size=self.size
      gv.board=np.copy(np.array(self.board)).tolist()
      gv.pieces=np.copy(np.array(self.pieces)).tolist()
      b = Board(gv)
      b.time=self.time
      b.done=self.done
      b._canon_flip=self._canon_flip
      b._canon_flip_king=self._canon_flip_king
      b.pieces_count = self.pieces_count
      b.progress_count = self.progress_count
      return b


    def countDiff(self, color):
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        count = 0
        for p in self.pieces:
            if p[0] >= 0:
               if p[2]*color > 0:
                   count += 1
               else:
                   count -= 1
        return count

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        return self._getValidMoves(color)
     
    def has_legal_moves(self, color):
        vm = self._getValidMoves(color)
        if len(vm)>0: return True
        return False


    def execute_move(self, move, color):
        """Perform the given move on the board.
        color gives the color pf the piece to play (1=white,-1=black)
        """
        x1,y1,x2,y2 = move
        pieceno = self._getPieceNo(x1,y1)
        legal = self._isLegalMove(pieceno,x2,y2)
        if legal>=0:
           #print("Accepted move: ",move) 
           self._moveByPieceNo(pieceno,x2,y2)
        #else:
           #print("Illegal move:",move,legal)
   
    '''
    def getImage(self):
        image = [[0 for col in range(self.width)] for row in range(self.height)]
        for item in self.board:
            image[item[1]][item[0]] = item[2]*10
        for piece in self.pieces:
            if piece[0] >= 0: image[piece[1]][piece[0]] = piece[2] + image[piece[1]][piece[0]]
        return image
    '''

    # TaflLogic.py
    def getImage(self):
        image = [[0 for _ in range(self.width)] for _ in range(self.height)]
        # 地形：角/王座保持不变
        for x, y, typ in self.board:
            image[y][x] = typ * 10

        flip = bool(getattr(self, "_canon_flip", False))
        sign = -1 if flip else 1
        flip_king = bool(getattr(self, "_canon_flip_king", True))

        for x, y, typ in self.pieces:
            if x < 0: 
                continue
            v = typ
            if sign == -1:               # 攻方视角：符号取反
                if v in (-1, 1):
                    v = -v
                elif v == 2 and flip_king:
                    v = -2
            image[y][x] += v
        return image

    

    def getPlayerToMove(self):
        return -(self.time%2*2-1)


################## Internal methods ##################

    def _isLegalMove(self,pieceno,x2,y2):
        try:

            # 是否超过棋盘范围
            if x2 < 0 or y2 < 0 or x2 >= self.width or y2 >= self.height: return -1 # 原版有误
            
            piece = self.pieces[pieceno]
            x1=piece[0]
            y1=piece[1]
            # 棋子已经死亡
            if x1<0: return -2 #piece was captured
            # 没有走直线
            if x1 != x2 and y1 != y2: return -3 #must move in straight line
            # 没有移动
            if x1 == x2 and y1 == y2: return -4 #no move

            piecetype = piece[2]
            if (piecetype == -1 and self.time%2 == 0) or (piecetype != -1 and self.time%2 == 1): return -5 #wrong player # time指轮数

            # 目的地是王座，禁落
            for item in self.board: # 是否目的地是禁落点
                if item[0] == x2 and item[1] == y2 and item[2] > 0:
                    return -10 #forbidden space
                # 禁止经过王座
                if y1==y2 and y1 == item[1] and ((x1 < item[0] and x2 >= item[0]) or (x1 > item[0] and x2 <= item[0])): 
                    return -10
                if x1==x2 and x1 == item[0] and ((y1 < item[1] and y2 >= item[1]) or (y1 > item[1] and y2 <= item[1])): 
                    return -10
            # x,y两个方向是否路径中间有棋子
            for apiece in self.pieces:
                if y1==y2 and y1 == apiece[1] and ((x1 < apiece[0] and x2 >= apiece[0]) or (x1 > apiece[0] and x2 <= apiece[0])): return -20 #interposing piece
                if x1==x2 and x1 == apiece[0] and ((y1 < apiece[1] and y2 >= apiece[1]) or (y1 > apiece[1] and y2 <= apiece[1])): return -20 #interposing piece

            return 0 # legal move
        except Exception as ex:
            print("error in islegalmove ",ex,pieceno,x2,y2)
            raise

   
    def _getCaptures(self,pieceno,x2,y2):
        #Assumes was already checked for legal move
        captures=[]
        piece=self.pieces[pieceno]
        piecetype = piece[2]
        for apiece in self.pieces:
            # 如果是敌对棋子
            if piecetype*apiece[2] < 0:
                d1 = apiece[0]-x2 
                d2 = apiece[1]-y2
                # 如果挨着
                if (abs(d1)==1 and d2==0) or (abs(d2)==1 and d1==0): 
                    check = True
                    throne = self.width // 2
                    for bpiece in self.pieces:
                        # 如果没有检测过，同时王不在王座
                        if check and bpiece[2] == 2 and (bpiece[0] != throne or bpiece[1] != throne):
                            check = False
                            e1 = throne-apiece[0]
                            e2 = throne-apiece[1]
                            if d1==e1 and d2==e2:
                                captures.extend([apiece])

                        # 是己方的棋子，但不是自己
                        if piecetype*bpiece[2] > 0 and not(piece[0]==bpiece[0] and piece[1]==bpiece[1]):
                            e1 = bpiece[0]-apiece[0]
                            e2 = bpiece[1]-apiece[1]
                            if d1==e1 and d2==e2:
                                captures.extend([apiece])
        return captures

    # returns code for invalid mode (<0) or number of pieces captured
    def _moveByPieceNo(self,pieceno,x2,y2):
      
      legal = self._isLegalMove(pieceno,x2,y2)
      if legal != 0: return legal

      self.time = self.time + 1

      piece=self.pieces[pieceno]
      piece[0]=x2
      piece[1]=y2
      caps = self._getCaptures(pieceno,x2,y2)
      #print("Captures = ",caps)
      for c in caps:
          c[0]=-99

      self._has_progress()
      self.done = self._getWinLose()
      
      return len(caps)
        
    def _has_progress(self):
        count = 0
        for i in self.pieces:
            if i[0] >= 0:
                count+=1
        if count < self.pieces_count:
            self.pieces_count = count
            self.progress_count = 0
        else:
            self.progress_count += 1

    def _getWinLose(self): # 胜负结果以白的视角给
        DRAW = args.draw
        if self.time >= args.limit or self.progress_count >= args.limit_progress:
            return DRAW
        w = self.width - 1
        for apiece in self.pieces:
            if apiece[2]==2 and apiece[0] > -1:
                # 修改为到达边沿后取胜
                if apiece[0] in {0, w} or apiece[1] in {0, w}:
                    return 1 #white won
                return 0 # no winner
        return -1  #white lost
   
    def _getPieceNo(self,x,y):
        # 返回棋子序号
       for pieceno in range(len(self.pieces)):
           piece=self.pieces[pieceno]
           if piece[0]==x and piece[1]==y: return pieceno
       return -1    
   
    def _getValidMoves(self,player):
       moves=[]
       for pieceno in range(len(self.pieces)):
           piece=self.pieces[pieceno]
           if piece[2]*player > 0: # 相同阵营
              #print("checking pieceno ",pieceno,piece)
              for x in range(0,self.width): # 在两条直线上实验
                  if self._isLegalMove(pieceno,x,piece[1])>=0:moves.extend([[piece[0],piece[1],x,piece[1]]])
              for y in range(0,self.height):
                  if self._isLegalMove(pieceno,piece[0],y)>=0:moves.extend([[piece[0],piece[1],piece[0],y]])
       #print("moves ",moves)
       return moves


