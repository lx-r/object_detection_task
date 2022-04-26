import torch
import torch.nn as nn
import torch.nn.functional as f

class SiameseNetwork(nn.Module):
    """Custom Siamese Network
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, stride=3, padding=2), # 10
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.001, beta=.75, k=2), # TODO
            nn.MaxPool2d(4, stride=2), # 4
            nn.Dropout2d(p=.5),          
        ) # 12544
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )
    
    def forward_once(self, x):
        y = self.cnn(x)
        y = y.view(y.size()[0], -1)
        y = self.fc(y)
        return y

    def forward(self, x1, x2):
        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)
        return y1, y2
    
    
if __name__ == '__main__':
    t = torch.randn((1,1,28,28))
    net = SiameseNetwork()
    y1, y2 = net(t, t)
    print(y1, y2)
        