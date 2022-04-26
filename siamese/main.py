import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import yaml

from Dataset import Dataset
from Loss import ContrastiveLoss
from utils import imshow, show_plot
from net import SiameseNetwork

def visualize_example_data(dataset):
    # Viewing the sample of images and to check whether its loading properly
    vis_dataloader = DataLoader(dataset, shuffle=True, batch_size=8)
    dataiter = iter(vis_dataloader)
    example_batch = next(dataiter)
    print(example_batch[0].shape)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    imshow(torchvision.utils.make_grid(concatenated))
    print(example_batch[2].numpy())


def load_config():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg


def train(model, opts, transform):
    
    dataset = Dataset(opts["data_dir"], opts["mode"], transform)
    # visualize_example_data(dataset)
    dataloader = DataLoader(dataset, shuffle=True,
                            num_workers=4, batch_size=opts["batch"])
    loss = []
    counter = []
    iteration_number = 0
    criterion = ContrastiveLoss(0.4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-3)
     
    for epoch in range(opts["epoches"]):
        for i, data in enumerate(dataloader, 0):
            img0, img1, label = data
            optimizer.zero_grad()
            y0, y1 = model(img0, img1)
            loss_cur = criterion(y0, y1, label)
            loss_cur.backward()
            optimizer.step()
            if i % opts["print_freq"] == 0:
                print(f"[{i}]iteration => loss: {loss_cur.item()}")
        print(f"[{epoch}] current loss: {loss_cur.item()}")
        iteration_number += len(dataloader) / opts["batch"]
        counter.append(iteration_number)
        loss.append(loss_cur.item())
        if epoch % opts["save_freq"] == 0:
            torch.save(model.state_dict(), f"model_{epoch}.pt")
    show_plot(counter, loss)       
    return model
     
def test(model, opts, transform):
    test_dataset = Dataset(opts["data_dir"], "test", transform)
    test_dataloader = DataLoader(test_dataset)
    
    model.eval()
    for i, data in enumerate(test_dataloader):
        x0, x1, label = data
        concat = torch.cat((x0, x1), dim=0)
        y1, y2 = model(x0, x1)
        dist = F.pairwise_distance(y1, y2)
        if label == torch.FloatTensor([[0]]):
            label = "Different Signature"
        else:
            label = "Same Signature"
        
        imshow(torchvision.utils.make_grid(concat))
        print(f"Predicted Distance:\t{dist.item()}")
        print(f"Actual Label:\t{label}")
        if i > 100:
            break
        

def main(opts):
    transform = transforms.Compose([
        # transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    
    net = SiameseNetwork()
    model = train(net, opts, transform)
    test(model, opts, transform)
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    
    
if __name__ == "__main__":
    opts = load_config()
    main(opts)