import csv

file = open("trainlog.csv", 'a')
writer = csv.writer(file, dialect='excel')

writer.writerow(["Epoch", "loss"])