from dis import dis
import numpy as np
import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, x1, x2, y):
        dist = F.pairwise_distance(x1, x2)
        total_loss = (1-y) * torch.pow(dist, 2) + \
            y * torch.pow(torch.clamp_min_(self.margin - dist, 0), 2)
        loss = torch.mean(total_loss)
        return loss
    
    
if __name__ == "__main__":
    x1 = torch.randint(0, 5, (4, 3,3))
    x2 = torch.randint(0, 5, (4, 3,3))
    y = torch.randint(0, 2, (4,1))
    loss = ContrastiveLoss(0.2)
    print(loss(x1, x2, y))