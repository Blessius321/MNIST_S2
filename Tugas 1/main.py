# mnist_cnn.py
# PyTorch 1.10.0-CPU Anaconda3-2020.02  Python 3.7.6
# Windows 10 

# reads MNIST data from text file rather than using
# built-in black box Dataset from torchvision

import numpy as np
import torch as T
from mnistloader import MNIST_Dataset
from CNN import Net 
from torchvision import transforms
from time import time
from tqdm import tqdm
import csv

file = open("trainlog.csv", 'a')
writer = csv.writer(file, dialect='excel')

device = T.device('cpu')
trainTxt = "train.txt"
testTxt = "test.txt"
bat_size = 100
compose = transforms.Compose([
  transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2), 
  transforms.Normalize(mean=(0.1307), std=(0.3079)),
])

def loadData(file, test = False):
  print(f"loading data from {file}")
  ds = MNIST_Dataset(file, transform = compose)
  if not test:
    return (T.utils.data.DataLoader(ds, batch_size=bat_size, shuffle=True), ds)
  else:
    return (T.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=True), ds)

def accuracy(model, file):
  (ldr, ds) = loadData(file, test=True)
  n_correct = 0
  for data in ldr:
    (pixels, labels) = data
    with T.no_grad():
      oupts = model(pixels)
    (_, predicteds) = T.max(oupts, 1)
    n_correct += (predicteds == labels).sum().item()

  acc = (n_correct * 1.0) / len(ds)
  return acc

def train(net, file, max_epochs = 15, lrn_rate = 0.005):
  (train_ldr, train_ds) = loadData(file)
  ep_log_interval = 1
  since = time()

  loss_func = T.nn.CrossEntropyLoss()  # does log-softmax()
  optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)

  print("\nStarting training")
  net.train()  # set mode
  for epoch in range(0, max_epochs):
    ep_loss = 0  # for one full epoch
    loop = tqdm(train_ldr)
    for (batch_idx, batch) in enumerate(loop):
      loop.set_description(f"Epoch [{epoch+1}/{max_epochs}]")
      (X, y) = batch  # X = pixels, y = target labels
      optimizer.zero_grad()
      oupt = net(X)
      loss_val = loss_func(oupt, y)  # a tensor
      ep_loss += loss_val.item()  # accumulate
      loss_val.backward()  # compute grads
      optimizer.step()     # update weights
      loop.set_postfix(loss=ep_loss)
    writer.writerow([epoch, ep_loss])

  print(f"\nDone training in {time() - since}s")

  print("\nComputing model accuracy")
  net.eval()
  acc_train = accuracy(net, trainTxt)  # all at once
  print("Accuracy on training data = %0.4f" % acc_train)

  net.eval()
  acc_test = accuracy(net, testTxt)  # all at once
  print("Accuracy on test data = %0.4f" % acc_test) 

def main():
  np.random.seed(1)
  T.manual_seed(1)

  net = Net().to(device)
  train(net, trainTxt, max_epochs=32) #train and evaluate on train and test set
  
  print("\nSaving trained model state")
  fn = "./Models/mnist_withpreprocessing.pt"
  T.save(net.state_dict(), fn)  
  file.close()


if __name__ == "__main__":
  main()
