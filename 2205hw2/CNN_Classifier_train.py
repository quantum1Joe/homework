import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms 
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import time
from models import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyDataset(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        self.transforms = transform
        self.df = pd.read_csv(csv_file, header=None, skiprows=1)
        self.classes = sorted(self.df[1].unique())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        vid, label = self.df.iloc[index, :]
        img_list = os.listdir(os.path.join(self.root, f"{vid}"))
        print(f"Trying to access: {os.path.join(self.root, f'{vid}')}")    
        img_list = sorted(img_list)
        img_path = os.path.join(self.root, f"{vid}", img_list[int(len(img_list)/2)])
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.classes.index(label)
        return img, label

# You can add data augmentation here
transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])


trainval_dataset = MyDataset("D:/code/homework/hw2/video_frames_30fpv_320p", "trainval.csv", transform)
train_data, val_data = train_test_split(trainval_dataset, test_size=0.2, random_state=0)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

net = Net().to(device)

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()


for epoch in range(50):
    # TODO: Metrics variables ...
    running_loss_train = 0.0
    correct_train = 0
    total_train = 0
    net.train()

    start_time = time.time()

        # TODO: Training code ...
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)


        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # 反向传播并优化
        loss.backward()
        optimizer.step()
        # 统计信息
        running_loss_train += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # TODO: Validation code ...  
    running_loss_val = 0.0
    correct_val = 0
    total_val = 0
    net.eval()  # 设置模型为评估模式  

    # TODO: save best model
    torch.save(net.state_dict(), 'model_best.pth')
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            # 前向传播
            val_outputs = net(val_inputs)
            val_loss = criterion(val_outputs, val_labels)

            # 统计信息
            running_loss_val += val_loss.item()
            _, val_predicted = torch.max(val_outputs.data, 1)
            total_val += val_labels.size(0)
            correct_val += (val_predicted == val_labels).sum().item()

    end_time = time.time()

    # save last model
    torch.save(net.state_dict(), 'model_last.pth')

    # print metrics log
    print('[Epoch %d] Loss (train/val): %.3f/%.3f' % (epoch + 1, running_loss_train/len(train_loader), running_loss_val/len(val_loader)),
        ' Acc (train/val): %.2f%%/%.2f%%' % (100 * correct_train/total_train, 100 * correct_val/total_val),
        ' Epoch Time: %.2f' % (end_time - start_time))
