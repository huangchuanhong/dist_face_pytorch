import torch.nn as nn
import torch

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Residual_Block(nn.Module):
    def __init__(self, in_c, hidden_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), cadinality=32):
        super(Residual_Block, self).__init__()
        self.conv = Conv_block(in_c, out_c=hidden_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_group = Conv_block(hidden_c, hidden_c, groups=hidden_c//cadinality, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(hidden_c, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.prelu = nn.PReLU(out_c)
        self.residual = residual
    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_group(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        output = self.prelu(output)
        return output


class Residual(nn.Module):
    def __init__(self, in_c, hidden_c, out_c, num_block, kernel=(3, 3), stride=(1, 1), padding=(1, 1), cadinality=32):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Residual_Block(in_c, hidden_c, out_c, residual=True, kernel=kernel, padding=padding, stride=stride, cadinality=cadinality))
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResFaceNext(nn.Module):
    def __init__(self, embedding_size):
        super(ResFaceNext, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv_23 = Residual_Block(64, 64, 256, kernel=(3, 3), stride=(1, 1), padding=(1, 1), cadinality=32)
        self.conv_3 = Residual(256, 64, 256, num_block=2, kernel=(3, 3), stride=(1, 1), padding=(1, 1), cadinality=32)
        self.conv_34 = Residual_Block(256, 128, 512, kernel=(3, 3), stride=(2, 2), padding=(1, 1), cadinality=32)
        self.conv_4 = Residual(512, 128, 512, num_block=3, kernel=(3, 3), stride=(1, 1), padding=(1, 1), cadinality=32)
        self.conv_45 = Residual_Block(512, 256, 1024, kernel=(3, 3), stride=(2, 2), padding=(1, 1), cadinality=32)
        self.conv_5 = Residual(1024, 256, 1024, num_block=5, kernel=(3, 3), stride=(1, 1), padding=(1, 1), cadinality=32)
        self.conv_56 = Residual_Block(1024, 512, 2048, kernel=(3, 3), stride=(2, 2), padding=(1, 1), cadinality=32)
        self.conv_6 = Residual(2048, 512, 2048, num_block=2, kernel=(3, 3), stride=(1, 1), padding=(1, 1), cadinality=32)
        self.conv_6_dw = Linear_block(2048, 2048, groups=2048, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(2048, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_56(out)
        out = self.conv_6(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)
        #return out

    def init_weights(self, pretrained=None):
        pass


