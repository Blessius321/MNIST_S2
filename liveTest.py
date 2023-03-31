import torch
import matplotlib.pyplot as plt
from CNN import Net
from random import randint
from mnistloader import MNIST_Dataset
from torchvision import transforms

device = torch.device('cpu')

def loadData(file, transform = None):
    print(f"loading data from {file}") 
    ds = MNIST_Dataset(file, transform=transform)
    return (torch.utils.data.DataLoader(ds, batch_size=1, shuffle = False), ds)

def showImg(image):
    image = image.permute(1,2,0)
    plt.figure()
    plt.imshow(image.numpy(), cmap=plt.get_cmap('gray_r'))
    plt.show()

def main():
    net = Net().to(device)
    net.load_state_dict(torch.load("Models/mnist_withpreprocessing.pt"))

    compose = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2), 
        transforms.Normalize(mean=(0.1307), std=(0.3079)),
        ])

    (ldr, ds) = loadData("full_test.txt", transform=compose)
    net.eval()
    for i in range(0,10):
        id = randint(0, len(ds))
        pixels = ds[id][0]
        labels = ds[id][1]
        input = pixels.unsqueeze(0)
        
        output = net(input)
        (_, pred) = torch.max(output, 1)
        print(f"Prediksi NN: {pred}, Ground Truth: {labels}")
        showImg(pixels)

    return

if (__name__ == '__main__'):
    main()