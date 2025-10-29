# Modified AlphaZero Model
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, channels=256):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=channels),
        )
    
    def forward(self, x):
        return F.relu(x + self.block(x))
    
class AlphaZeroNet_Tablut(nn.Module):
    def __init__(self, game, args, in_channels=6, channels=128):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        assert self.board_x == self.board_y
        self.size = self.board_x
        self.action_size = game.getActionSize()
        self.args = args
        self.out_channels = self.action_size
        self.in_channels = in_channels
        self.channels = channels
        self.rank = getattr(args, "policy_rank", 64)   # 低秩维度，可调 16/32

        super(AlphaZeroNet_Tablut, self).__init__()
        self.Block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
        )
        self.ResidualBlocks = nn.Sequential(*(ResidualBlock(in_channels=channels, channels=channels) for _ in range(8)))

        self.p_drop = nn.Dropout(p=0.10)

        self.ValueHead = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=self.board_x*self.board_y, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1),
            nn.Tanh(),
        )

        self.f_head = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=self.rank, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=self.rank),
            nn.ReLU(inplace=True), # 9*9*2
        )

        self.g_head = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=self.rank, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=self.rank),
            nn.ReLU(inplace=True), # 9*9*2
        )

    def forward(self, s):
        s = self.Block(s)
        s = self.ResidualBlocks(s)

        f = self.f_head(s)                     # (N,R,S,S)
        g = self.g_head(s)                     # (N,R,S,S)
        N = s.size(0)
        f = f.reshape(N, self.rank, self.board_x*self.board_y).transpose(1, 2)  # (N,S^2,R)  row = from
        g = g.reshape(N, self.rank, self.board_x*self.board_y) # (N,R,S^2)  col = to
        logits_pairs = torch.bmm(f, g) # (N,S^2,S^2) [from,to]
        logits_pairs = self.p_drop(logits_pairs)
        pi = logits_pairs.transpose(1, 2).contiguous().reshape(N, self.size**4)

        v = self.ValueHead(s)
        return F.log_softmax(pi, dim=1), v