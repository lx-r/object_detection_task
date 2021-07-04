import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

LR = 0.1
BATCH_SIZE = 20
EPOCH = 10

# 生成数据
x = torch.unsqueeze(torch.linspace(-1,1,1500),dim=1)
y = x.pow(3) + 0.1*torch.normal(torch.zeros(*x.size()))

# 画出数据
plt.scatter(x.numpy(), y.numpy())
plt.show()

# 把数据转成tensor类型
torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(1,20) 
        self.predict = torch.nn.Linear(20,1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net_SGD = Net()
net_Momentum = Net()
net_RMSProp = Net()
net_AdaGrad = Net()
net_Adam = Net()

nets = [net_SGD, net_Momentum, net_RMSProp, net_AdaGrad, net_Adam]

opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_AdaGrad = torch.optim.Adagrad(net_AdaGrad.parameters(), lr=LR)
opt_RMSProp = torch.optim.RMSprop(net_RMSProp.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9,0.99))

optimizers = [opt_SGD, opt_Momentum, opt_RMSProp, opt_AdaGrad, opt_Adam]

loss_func = torch.nn.MSELoss()

losses_his = [[], [], [], [], []]

# 训练

for epoch in range(EPOCH):
    print("Epoch: ", epoch)
    for step,(batch_x,batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        for net, opt, l_his, in zip(nets, optimizers, losses_his):
            output = net(b_x)
            loss = loss_func(output, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.item())

labels = ['SGD', 'Momentum', 'AdaGrad', 'RMSProp', 'Adam']
for i,l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc = "best")
plt.xlabel('Step')
plt.ylabel('Loss')
plt.ylim((0,0.2))
plt.show()