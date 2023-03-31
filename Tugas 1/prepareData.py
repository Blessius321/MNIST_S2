import numpy as np
import matplotlib.pyplot as plt

def convert(img_file, label_file, txt_file, n_images):
  print("\nOpening binary pixels and labels files ")
  lbl_f = open(label_file, "rb")   # labels (digits)
  img_f = open(img_file, "rb")     # pixel values
  print("Opening destination text file ")
  txt_f = open(txt_file, "w")      # output to write to

  print("Discarding binary pixel and label headers ")
  img_f.read(16)   # discard header info
  lbl_f.read(8)    # discard header info

  print("\nReading binary files, writing to text file ")
  print("Format: 784 pixels then labels, tab delimited ")
  for i in range(n_images):   # number requested 
    lbl = ord(lbl_f.read(1))  # Unicode, one byte
    for j in range(784):  # get 784 pixel vals
      val = ord(img_f.read(1))
      txt_f.write(str(val) + "\t") 
    txt_f.write(str(lbl) + "\n")
  img_f.close(); txt_f.close(); lbl_f.close()
  print("\nDone ")

def display_from_file(txt_file, idx):
  all_data = np.loadtxt(txt_file, delimiter="\t",
    usecols=range(0,785), dtype=np.int64)

  x_data = all_data[:,0:784]  # all rows, 784 cols
  y_data = all_data[:,784]    # all rows, last col

  label = y_data[idx]
  print("digit = ", str(label), "\n")

  pixels = x_data[idx]
  pixels = pixels.reshape((28,28))
  for i in range(28):
    for j in range(28):
      # print("%.2X" % pixels[i,j], end="")
      print("%3d" % pixels[i,j], end="")
      print(" ", end="")
    print("")

  plt.tight_layout()
  plt.imshow(pixels, cmap=plt.get_cmap('gray_r'))
  plt.show()  

nTrain = 30000
nTest = 10000

train = "train-images-idx3-ubyte/train-images.idx3-ubyte"
train_label = "train-labels-idx1-ubyte/train-labels.idx1-ubyte"

test = "t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"
test_label = "t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"

# print("making train set")
# convert(train, train_label, "train.txt", nTrain)
# display_from_file("./train.txt", idx= 10)

print("making test set")
convert(test, test_label, "full_test.txt", 10000)
display_from_file("./full_test.txt", idx= 9)

