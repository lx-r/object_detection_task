import glob 
import os
import cv2
import random
import shutil

def sample_data():
    # glob.glob("/media/lx/data/dataset/data/train/*.png")
    dp = "/media/lx/data/dataset/data/train/"
    data_list = range(10000)
    # data_list.sort()
    print(data_list)
    with open("/media/lx/data/dataset/data/train_labels.txt") as f:
        labels = f.read().rstrip("\n").split()

    img_path = "dataset/images"
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    data_images = list(zip(data_list, labels))
    random.shuffle(data_images)
    datas = random.sample(data_images, 1000)

    for i, (p, l) in enumerate(datas):
        shutil.copy(os.path.join(dp, f"{p}.png"), os.path.join(img_path, f"{l}_{p}.png"))
    
def get_data_pair():
    datas = os.listdir("dataset/images")
    random.shuffle(datas)
    partion = int(.9*len(datas))
    train_data = datas[:partion]
    test_data = datas[partion:]
    with open("dataset/train.csv", "w") as f:
        for i in range(8000):
            pairs = random.sample(train_data, 2)
            cat = 1 if pairs[0][0] == pairs[1][0] else 0
            f.write(f"images/{pairs[0]},images/{pairs[1]},{cat}\n")
            
    with open("dataset/test.csv", "w") as f:
        for i in range(800):
            pairs = random.sample(test_data, 2)
            cat = 1 if pairs[0][0] == pairs[1][0] else 0
            f.write(f"images/{pairs[0]},images/{pairs[1]},{cat}\n")
            
if __name__ == "__main__":
    get_data_pair()