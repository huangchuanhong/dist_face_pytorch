import math
import torch
import torch.nn as nn
from ..registry import TOP_MODEL

@TOP_MODEL.register_module
class PairwiseCirclelossModel(nn.Module):
    def __init__(self, m=0.25, gamma=256, gpu=0):
        super(PairwiseCirclelossModel, self).__init__()
        self.device = gpu
        self.soft_plus = nn.Softplus()
        self.margin = torch.FloatTensor([m]).to("cuda:{}".format(self.device))
        self.gamma = torch.FloatTensor([gamma]).to("cuda:{}".format(self.device))

    def init_weights(self, pretrained=None):
        pass

    def _forward_train(self, feat, label):
        feat = feat.to("cuda:{}".format(self.device))
        label = label.to("cuda:{}".format(self.device))
        normed_feat = torch.nn.functional.normalize(feat)
        bs = label.size(0)
        mask = label.expand(bs, bs).t().eq(label.expand(bs, bs)).float()
        simi = torch.mm(normed_feat, normed_feat.transpose(0, 1))
        pos_mask = mask-torch.eye(bs).to("cuda:{}".format(self.device))
        neg_mask = 1 - mask
        sp = simi[pos_mask == 1].contiguous().view(bs, -1)
        sn = simi[neg_mask == 1].contiguous().view(bs, -1)
        alpha_p = (1 + self.margin - sp.detach()).clamp(min=0)
        alpha_n = (sn.detach() + self.margin).clamp(min=0)
        p_sub = torch.sub(1 - self.margin, sp)
        n_sub = torch.sub(sn, self.margin)
        loss = self.soft_plus(torch.logsumexp(alpha_p * p_sub * self.gamma, dim=1)
                              + torch.logsumexp(alpha_n * n_sub * self.gamma, dim=1))
        return loss, sp.view(-1), sn.view(-1)

    def forward(self, input, label, mode='train'):
        if mode == 'train':
            return self._forward_train(input, label)
        elif mode == 'val':
            return self._forward_val(input, label)

    def _forward_val(self, input, label):
        pass

if __name__ == '__main__':
    model = PairwiseCirclelossModel()
    feat = torch.tensor([[1,2], [3,4], [5,6], [7,8]], dtype=torch.float32)
    label = torch.tensor([1,2,1,2], dtype=torch.float32)
    print(model(feat, label))
