# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class CutNet(torch.nn.Module):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    """

    def __init__(self):
        super(CutNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 3, 1, 1)

    def forward(self, input):
        output = self.conv1(input)
        return output


class PoolNet(torch.nn.Module):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
    """

    def __init__(self):
        super(PoolNet, self).__init__()
        self.mp1 = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        output = self.mp1(x)
        return output


def model_test(name):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(56),
        torchvision.transforms.ToTensor()
    ])
    dataset = torchvision.datasets.MNIST("data", train=False, transform=transforms,
                                         download=False)

    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    summer = SummaryWriter("logs")
    for idx, data in enumerate(dataloader):
        imgs, labels = data
        print(f"shape of images: {imgs.shape}")
        summer.add_images("inputs", imgs, idx)
        net = CutNet()
        output = net(imgs)
        summer.add_images("conv_outputs", output, idx)
        mp_net = PoolNet()
        output = mp_net(imgs)
        summer.add_images("max_outputs", output, idx)

    x = torch.tensor([1., 2, 3, 4, 4, 3, 2, 1])
    x = x.reshape(-1, 1, 2, 4)
    print(x)
    mp_net = PoolNet()
    print("mpooled result: ", mp_net(x))
    kernel = torch.tensor([1, 2, 2., 1]).reshape(1, 1, 2, 2)
    print(f"kernel: {kernel}")
    y = torch.nn.functional.conv2d(x, kernel, stride=2)
    print(f"Conv2d Result: {y}")
    summer.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model_test('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
