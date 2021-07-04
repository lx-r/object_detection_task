import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper Parameters
torch.manual_seed(1)

input_size = 784
hidden_size = 500
num_classes = 10
num_epoches = 5
batch_size = 64
lr = 0.001

# Load MNIST Data
train_dataset = dsets.MNIST(
    root="./data",
    train=True,
    transform=transforms.ToTensor(),
    download=True
    )

test_dataset = dsets.MNIST(
    root="./data",
    train=False,
    transform=transforms.ToTensor()
)

# Data Loader, 设置batch_size, shuffle=True训练时需打乱数据
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Create Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1000)
        self.fc3 = nn.Linear(1000,num_classes)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


net = Net(input_size,hidden_size,num_classes)
print("=====>>>>> fc model",net)

losses = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=lr)

# training
for epoch in range(num_epoches):
    for i,(images,labels) in enumerate(train_loader):
        # convert torch tensor to variable
        images = Variable(images.view(-1,28*28))
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad() 
        outputs = net(images)
        loss = losses(outputs, labels)
        loss.backward()
        optimizer.step()

        if(i+1)%100 == 0:
            print("Epoch [%d/%d], Step[%d/%d], Loss: %.4f"%(epoch+1,num_epoches,
            i+1,len(train_dataset)//batch_size,loss.item()))

# Test the model
correct = 0
total = 0
for img, labels in test_loader:
    img = Variable(img.view(-1,28*28))
    outputs = net(img)
    _, predicted = torch.max(outputs.data,1)
    total += labels.size(0)
    correct += (predicted==labels).sum()
print("Accuray of the nn of the 1000 test images:%d %%"%(correct/total*100))
