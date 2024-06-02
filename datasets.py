import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

def load_data(data_dir, sub_dirs, image_size=(128, 128)):
    x, t = [], []
    cnt = 0
    for dir in sub_dirs:
        folder_dir = os.path.join(data_dir, dir)
        for file_name in os.scandir(folder_dir):
            img = Image.open(file_name.path).resize(image_size)
            npimg = np.array(img) / 255.0
            x.append(npimg)
            t.append(cnt)
        cnt += 1
    x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
    t = torch.tensor(t, dtype=torch.int64)
    return TensorDataset(x, t)

def get_dataloaders(data_dir='/content/drive/MyDrive/AI/datasets2/', sub_dirs=['photo/', 'midjourney/', 'stablediffusion/', 'dalle3/'], batch_size=5):
    dataset = load_data(data_dir, sub_dirs)
    train_size = int(0.75 * len(dataset))
    val_size = test_size = (len(dataset) - train_size) // 2
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
