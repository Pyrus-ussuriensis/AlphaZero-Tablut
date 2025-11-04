# tablut/ui/viewer_tafl.py
import pygame
import sys, math, time
from typing import List, Tuple

from tablut.rules.TaflGame import TaflGame
from tablut.rules.TaflLogic import Board
from tablut.utils.Digits import base2int, int2base
import os, time
import numpy as np
import imageio.v2 as imageio  # 用 get_writer/append_data

CELL = 80         # 单格像素
MARGIN = 40       # 画布边距
PIECE_PAD = 10
#BG = (242, 242, 242)
#GRID = (60, 60, 60)
#WHITE = (240, 240, 240)
#BLACK = (30, 30, 30)
#KING  = (220, 170, 60)
HINT  = (120, 180, 255)
SEL   = (0, 0, 128)
LAST  = (255, 110, 110)
THRONE= (180, 180, 180)
BG    = (230, 230, 230)
GRID  = (90, 90, 90)
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
KING  = (230, 180, 60)


class PygameObserver:
    """Arena 的 on_step/on_end 回调，它自己负责绘制、节流和人类点击输入"""
    def __init__(self, game: TaflGame, delay_ms=300, step_mode=False, record=False, out_path=None, fps=30):
        self.game = game
        self.n = game.n
        self.size = MARGIN*2 + self.n*CELL # 棋盘尺寸
        self.delay_ms = delay_ms # 显示延迟，便于观看
        self.step_mode = step_mode # 是否按一下空格显示一下
        self.record   = record # 是否记录视频
        self.out_path = out_path or f"replay_{self.n}x{self.n}_{int(time.time())}.mp4" # 视频输出路径
        self.fps      = fps # 帧率
        self._writer  = None # 视频对象
        if self.record:
            self._open_recorder()

        # 初始化导入的pygame模块
        pygame.init()
        # 创建窗口
        self.screen = pygame.display.set_mode((self.size, self.size))
        # 设置窗口标题，帧率，字体
        pygame.display.set_caption(f"Tafl Viewer ({self.n}x{self.n})")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 22)

        self.last_move = None # 最后一次移动记录          # (x1,y1,x2,y2)
        self.hints: List[Tuple[int,int]] = [] # 可选动作
        self.sel = None                 # (x,y) 已选起点
        self.human_wants_action = False # 是否等待人类输入
        self.human_action = None        # 选好的 action 索引
        self.running = True # 是否正在运行

    # 根据不同格式使用不同视频写入器
    def _open_recorder(self):
        ext = os.path.splitext(self.out_path)[1].lower()
        try:
            if ext in ('.mp4', '.mkv', ''):
                # mp4 推荐，体积小；macro_block_size=None 允许任意分辨率
                self._writer = imageio.get_writer(
                    self.out_path if ext else self.out_path + '.mp4',
                    fps=self.fps, codec='libx264', quality=8,
                    pixelformat='yuv420p', macro_block_size=None
                )
            elif ext == '.gif':
                self._writer = imageio.get_writer(self.out_path, fps=self.fps)
            else:
                # 默认 mp4
                self.out_path += '.mp4'
                self._writer = imageio.get_writer(
                    self.out_path, fps=self.fps, codec='libx264',
                    pixelformat='yuv420p', macro_block_size=None
                )
            print(f"[rec] Recording to: {self.out_path}  @ {self.fps} fps")
        except Exception as e:
            print(f"[rec] Failed to open writer: {e}")
            self._writer = None

    # 停止写入，保存
    def _close_recorder(self):
        if self._writer is not None:
            try:
                self._writer.close()
                print(f"[rec] Saved: {self.out_path}")
            except Exception as e:
                print(f"[rec] Close writer error: {e}")
            self._writer = None

    # 记录图像
    def _capture_if_recording(self):
        if self._writer is None:
            return
        # Pygame -> (H,W,3) uint8
        arr = pygame.surfarray.array3d(self.screen)      # (W,H,3)
        frame = np.transpose(arr, (1, 0, 2)).copy()      # -> (H,W,3)
        self._writer.append_data(frame)

    # 翻转记录器状态
    def toggle_record(self):
        if self._writer is None:
            self._open_recorder()
        else:
            self._close_recorder()

    # 保存这一帧为图片
    def snapshot_png(self, path=None):
        path = path or f"snap_{int(time.time())}.png"
        # 直接保存当前 Surface
        pygame.image.save(self.screen, path)
        print(f"[rec] Snapshot: {path}")

    # ---- 工具 ----
    # 像素坐标转化为棋盘坐标
    def _board_to_xy(self, mx, my):
        x = (mx - MARGIN) // CELL
        y = (my - MARGIN) // CELL
        if 0 <= x < self.n and 0 <= y < self.n:
            return int(x), int(self.n-1-y)  # y 轴翻转（上方为高 y）
        return None

    # 渲染文字
    def _draw_text(self, s, x, y, color=(0,0,0)):
        surf = self.font.render(s, True, color)
        self.screen.blit(surf, (x, y))

    '''
    def _draw_board(self, board: Board):
        self.screen.fill(BG)
        # 网格
        for i in range(self.n+1):
            sx = MARGIN + i*CELL
            sy = MARGIN + i*CELL
            pygame.draw.line(self.screen, GRID, (MARGIN, MARGIN+i*CELL), (MARGIN+self.n*CELL, MARGIN+i*CELL), 1)
            pygame.draw.line(self.screen, GRID, (MARGIN+i*CELL, MARGIN), (MARGIN+i*CELL, MARGIN+self.n*CELL), 1)
        # 王座(中心)底色
        cx = self.n//2; cy = self.n//2
        rx = MARGIN + cx*CELL; ry = MARGIN + (self.n-1-cy)*CELL
        pygame.draw.rect(self.screen, THRONE, (rx+1, ry+1, CELL-2, CELL-2), 0)

        img = board.getImage()  # 原始棋面
        # 画棋子
        for y in range(self.n):
            for x in range(self.n):
                v = img[y][x]
                if v == 0 or v % 10 == 0:
                    continue
                p = v % 10  # 去掉地形
                sx = MARGIN + x*CELL + PIECE_PAD
                sy = MARGIN + (self.n-1-y)*CELL + PIECE_PAD
                r = CELL - 2*PIECE_PAD
                if abs(p) == 2:  # 王
                    pygame.draw.circle(self.screen, KING, (sx+r//2, sy+r//2), r//2)
                    pygame.draw.circle(self.screen, BLACK, (sx+r//2, sy+r//2), r//2, 2)
                elif p > 0:      # 白子
                    pygame.draw.circle(self.screen, WHITE, (sx+r//2, sy+r//2), r//2)
                    pygame.draw.circle(self.screen, BLACK, (sx+r//2, sy+r//2), r//2, 2)
                else:            # 黑子
                    pygame.draw.circle(self.screen, BLACK, (sx+r//2, sy+r//2), r//2)
        # 上一步高亮
        if self.last_move:
            x1,y1,x2,y2 = self.last_move
            for (x,y) in [(x1,y1),(x2,y2)]:
                rx = MARGIN + x*CELL
                ry = MARGIN + (self.n-1-y)*CELL
                pygame.draw.rect(self.screen, LAST, (rx+2, ry+2, CELL-4, CELL-4), 3)
        # 合法目标高亮
        for (x,y) in self.hints:
            rx = MARGIN + x*CELL
            ry = MARGIN + (self.n-1-y)*CELL
            pygame.draw.rect(self.screen, HINT, (rx+4, ry+4, CELL-8, CELL-8), 2)
    def _draw_board(self, board: Board):
        self.screen.fill(BG)
        # 1) 画网格 …
        # 2) 画王座底色（用 board.board 或 getImage 的十位判断即可）
        for x, y, typ in board.board:
            if typ > 0:  # 王座/角等
                rx = MARGIN + x*CELL
                ry = MARGIN + (self.n-1-y)*CELL
                pygame.draw.rect(self.screen, THRONE, (rx+1, ry+1, CELL-2, CELL-2), 0)
    '''
    # 画棋盘，王座，棋子，特殊提示框
    def _draw_board(self, board):
        self.screen.fill(BG)

        # —— 1) 画网格（先画横再画竖，线条放在所有棋子之下）——
        for i in range(self.n + 1):
            y = MARGIN + i*CELL
            x = MARGIN + i*CELL
            pygame.draw.line(self.screen, GRID, (MARGIN, y), (MARGIN + self.n*CELL, y), 1)
            pygame.draw.line(self.screen, GRID, (x, MARGIN), (x, MARGIN + self.n*CELL), 1)
        
        # 王座(中心)底色
        cx = self.n//2; cy = self.n//2
        rx = MARGIN + cx*CELL; ry = MARGIN + (self.n-1-cy)*CELL
        pygame.draw.rect(self.screen, THRONE, (rx+1, ry+1, CELL-2, CELL-2), 0)


        # 3) 画棋子 —— 只看 board.pieces 的 typ（不会被地形干扰）
        for x, y, typ in board.pieces:
            if x < 0:            # 被吃掉的
                continue
            sx = MARGIN + x*CELL + PIECE_PAD
            sy = MARGIN + (self.n-1-y)*CELL + PIECE_PAD
            r  = CELL - 2*PIECE_PAD

            if typ == 2:         # 王
                fill = KING
            elif typ > 0:        # 白
                fill = WHITE
            else:                # 黑
                fill = BLACK

            pygame.draw.circle(self.screen, fill, (sx+r//2, sy+r//2), r//2)
            pygame.draw.circle(self.screen, BLACK, (sx+r//2, sy+r//2), r//2, 2)  # 外圈

        if self.last_move:
            x1, y1, x2, y2 = self.last_move
            for (x, y) in [(x1, y1), (x2, y2)]:
                rx = MARGIN + x*CELL
                ry = MARGIN + (self.n - 1 - y)*CELL
                pygame.draw.rect(self.screen, LAST, (rx+2, ry+2, CELL-4, CELL-4), 3)
        
        if self.sel:
            x, y = self.sel
            rx = MARGIN + x*CELL
            ry = MARGIN + (self.n - 1 - y)*CELL
            pygame.draw.rect(self.screen, SEL, (rx+2, ry+2, CELL-4, CELL-4), 3)

        for (x, y) in self.hints:
            rx = MARGIN + x*CELL
            ry = MARGIN + (self.n - 1 - y)*CELL
            pygame.draw.rect(self.screen, HINT, (rx+4, ry+4, CELL-8, CELL-8), 2)

    # 末端，画图，显示文字，刷新到屏幕记录一秒，停止记录，监听按键
    def on_end(self, board, result, it):
        self._draw_board(board)
        txt = f"Game Over. Result={result}  (Turns={it})   [Q] quit"
        self._draw_text(txt, 10, 30, (0,120,0))
        pygame.display.flip()
        #self._capture_if_recording()
        # 结尾多录 1 秒停留画面
        for _ in range(self.fps):
            self._capture_if_recording()
            self.clock.tick(self.fps)
        self._close_recorder()
        # 等待任意键退出
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit(0); self._close_recorder(); pygame.quit(); sys.exit(0)
                elif event.type == pygame.KEYDOWN: return


    # 写入文本情况信息
    def _draw_hud(self, curPlayer, it, delay_ms):
        self._draw_text(f"Turn: {it}   To move: {'White(+1)' if curPlayer==1 else 'Black(-1)'}", 10, 10)
        self._draw_text(f"Delay: {delay_ms} ms   [D/F to -/+ ]   Step mode: {'ON' if self.step_mode else 'OFF'} (SPACE)", 10, self.size-28)
        self._draw_text("[Mouse] select piece → target   [R] reset select   [Q] quit", 10, self.size-48)

    # ---- 人类输入：返回 action（原始棋面坐标系）----
    # 人类输入先初始化值，得到合法动作，监听按键和鼠标，根据按键和点击位置更新变量值
    # 更新画面，记录画面
    def get_human_action(self, board: Board):
        """阻塞等待：点击起点->终点；或输入坐标 i,j,k,l（按 I 切换）"""
        self.sel = None
        self.hints.clear()
        typing = False
        typed = ""

        legal = board.get_legal_moves(board.getPlayerToMove())  # [[x1,y1,x2,y2],...]
        from_map = {}
        for (x1,y1,x2,y2) in legal:
            from_map.setdefault((x1,y1), []).append((x2,y2))

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit(0)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit(); sys.exit(0)
                    if event.key == pygame.K_r:
                        self.sel = None; self.hints.clear()
                    if event.key == pygame.K_i:
                        typing = not typing; typed = ""
                    if typing:
                        if event.key == pygame.K_RETURN:
                            try:
                                parts = [int(x) for x in typed.replace(',', ' ').split()]
                                if len(parts) == 4:
                                    x1,y1,x2,y2 = parts
                                    return base2int([x1,y1,x2,y2], self.n)
                                typed = ""
                            except:
                                typed = ""
                        elif event.key == pygame.K_BACKSPACE:
                            typed = typed[:-1]
                        else:
                            ch = event.unicode
                            if ch.isprintable():
                                typed += ch
                    else:
                        # 速度 & 步进键无效（人类回合不需要）
                        pass
                elif event.type == pygame.MOUSEBUTTONDOWN and not typing:
                    pos = pygame.mouse.get_pos()
                    xy = self._board_to_xy(*pos)
                    if xy is None: continue
                    if self.sel is None:
                        if xy in from_map:
                            self.sel = xy
                            self.hints = from_map[xy]
                    else:
                        if xy in self.hints:
                            x1,y1 = self.sel
                            x2,y2 = xy
                            return base2int([x1,y1,x2,y2], self.n)
                        else:
                            # 重新选择
                            if xy in from_map:
                                self.sel = xy
                                self.hints = from_map[xy]
                            else:
                                self.sel = None
                                self.hints.clear()

            # 画面更新
            self._draw_board(board)
            self._draw_hud(board.getPlayerToMove(), 0, self.delay_ms)
            if typing:
                self._draw_text("Input x1 y1 x2 y2: " + typed, 10, 30, (180,0,0))
            pygame.display.flip()
            self._capture_if_recording()
            self.clock.tick(60)

    # ---- Arena 回调 ----
    # 根据棋盘画面，输入的动作，谁在下，监听键盘，更新画面，文本，刷新，记录，看是否等待。
    def on_step(self, board: Board, curPlayer: int, action: int, it: int):
        # AI 回合：节流或单步
        x1,y1,x2,y2 = int2base(action, self.n, 4)
        self.last_move = (x1,y1,x2,y2)
        self.hints.clear(); self.sel = None

        # 事件泵 + 热键
        waiting = True
        start = time.time()
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit(0)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit(); sys.exit(0)
                    elif event.key == pygame.K_SPACE:
                        self.step_mode = not self.step_mode  # 切换步骤模式
                    elif event.key == pygame.K_d:
                        self.delay_ms = max(0, self.delay_ms-100)
                    elif event.key == pygame.K_f:
                        self.delay_ms += 100

            self._draw_board(board)
            self._draw_hud(curPlayer, it, self.delay_ms)
            pygame.display.flip()
            self._capture_if_recording()
            self.clock.tick(60)

            if self.step_mode:
                # 等待按下 N 键推进一步
                keys = pygame.key.get_pressed()
                if keys[pygame.K_n]:
                    waiting = False
            else:
                if (time.time() - start) * 1000.0 >= self.delay_ms:
                    waiting = False

    '''
    def on_end(self, board: Board, result: int, it: int):
        self._draw_board(board)
        txt = f"Game Over. Result={result}  (Turns={it})   [Q] quit"
        self._draw_text(txt, 10, 30, (0,120,0))
        pygame.display.flip()
        # 等待退出或任意键
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit(0)
                elif event.type == pygame.KEYDOWN:
                    return
    '''

# ----- 人类玩家（原始棋盘） -----
# 获取观察，输入棋盘会调用观察得到人类动作
class HumanGUIPlayer:
    expects_canonical = False   # 关键：告诉 Arena 用原始棋盘调我

    def __init__(self, observer: PygameObserver):
        self.observer = observer

    def startGame(self): pass
    def endGame(self): pass

    def __call__(self, board: Board) -> int:
        # Arena 会把原始棋盘传进来（因为 expects_canonical=False）
        return self.observer.get_human_action(board)
