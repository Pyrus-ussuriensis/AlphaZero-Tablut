import sys
#sys.path.append('..')
from tablut.utils.utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TaflNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(TaflNNet, self).__init__()
        self.conv1 = nn.Conv2d(6, args.num_channels, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1, bias=False)

        '''
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)
        '''
        def GN(c): return nn.GroupNorm(8, c)
        self.bn1 = GN(args.num_channels)
        self.bn2 = GN(args.num_channels)
        self.bn3 = GN(args.num_channels)
        self.bn4 = GN(args.num_channels)
        C = args.num_channels


        # Value 头：GAP + MLP
        self.v_conv = nn.Conv2d(C, C//2, 1, bias=False)
        self.v_bn   = nn.GroupNorm(8, C//2)
        self.v_fc1  = nn.Linear((C//2), 128)
        self.v_fc2  = nn.Linear(128, 1)

        # Policy 头：双线性 from×to
        R = getattr(args, "policy_rank", 32)   # 低秩维度，可调 16/32
        self.f_head = nn.Conv2d(C, R, 1, bias=False)  # from 嵌入
        self.g_head = nn.Conv2d(C, R, 1, bias=False)  # to   嵌入

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        #s = s.view(-1, 6, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        #s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))

        # ---- policy 双线性头 ----
        f = self.f_head(s)                     # (N,R,S,S)
        g = self.g_head(s)                     # (N,R,S,S)
        N, R, S, _ = f.shape
        f = f.reshape(N, R, S*S).transpose(1, 2)   # (N, S^2, R)
        g = g.reshape(N, R, S*S)                   # (N, R, S^2)
        logits_pairs = torch.bmm(f, g)             # (N, S^2, S^2)
        pi_logits = logits_pairs.transpose(1, 2)    # (N, S^2, S^2) = [to, from]
        pi_logits = pi_logits.reshape(N, S*S*S*S)  # (N, 6561)
        #pi_logits = logits_pairs.reshape(N, S*S*S*S)  # (N, 6561)

        # ---- value 头 ----
        v = F.relu(self.v_bn(self.v_conv(s)))   # (N,C//2,S,S)
        v = v.mean(dim=(2,3))                   # GAP -> (N,C//2)
        v = F.relu(self.v_fc1(v))               # (N,128)
        v = torch.tanh(self.v_fc2(v))           # (N,1)

        return F.log_softmax(pi_logits, dim=1), v