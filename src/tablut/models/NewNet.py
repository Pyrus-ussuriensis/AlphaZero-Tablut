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
    
class AlphaZeroNet(nn.Module):
    def __init__(self, game, args, in_channels=6, channels=256):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.size = self.board_x
        self.action_size = game.getActionSize()
        self.args = args
        self.out_channels = self.action_size
        self.in_channels = in_channels
        self.channels = channels

        super(AlphaZeroNet, self).__init__()
        self.Block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
        )
        self.ResidualBlocks = nn.Sequential(*(ResidualBlock(in_channels=channels) for _ in range(19)))

        self.PolicyHead = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(inplace=True), # 9*9*2
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=self.board_x*self.board_y*2, out_features=self.out_channels),
        )

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


    def forward(self, s):
        s = self.Block(s)
        s = self.ResidualBlocks(s)
        pi = self.PolicyHead(s)
        v = self.ValueHead(s)
        return F.log_softmax(pi, dim=1), v