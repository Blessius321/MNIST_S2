import torch
import numpy as np
device = torch.device('cpu')

class MNIST_Dataset(torch.utils.data.Dataset):
  # 784 tab-delim pixel values (0-255) then label (0-9)
  def __init__(self, src_file, transform = None):
    all_xy = np.loadtxt(src_file, usecols=range(785),
      delimiter="\t", comments="#", dtype=np.float32)

    self.transform = transform
    tmp_x = all_xy[:, 0:784]  # all rows, cols [0,783]
    tmp_x /= 255
    tmp_x = tmp_x.reshape(-1, 1, 28, 28)
    tmp_y = all_xy[:, 784]

    self.x_data = \
      torch.tensor(tmp_x, dtype=torch.float32).to(device)
    self.y_data = \
      torch.tensor(tmp_y, dtype=torch.int64).to(device)
     

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    lbl = self.y_data[idx]  # no use labels
    pixels = self.x_data[idx]

    if self.transform:
      pixels = self.transform(pixels) 
    return (pixels, lbl)