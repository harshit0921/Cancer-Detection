import numpy as np
import torch
from PIL import Image
import pandas as pd
import sys
import os
import math

script_location = os.path.dirname(__file__)

#Read csv
data = pd.read_csv(os.path.join(script_location, 'SKIN_CANCER_metadata.csv'))

##Read an image to get resolution
image_directory = os.path.join(script_location, 'HAM10000')
file_path = os.path.join(image_directory, data['Name'][0] + '.jpg')
im = Image.open(file_path, 'r')
pixels = im.size[0] * im.size[1]
batches = 100
batch_size = int(math.ceil(len(data['Name'])/batches))

#Create array to store pixel values of a batch (N x d)
images = torch.zeros(batch_size, pixels+4)
save_file_directory = os.path.join(script_location, 'batches')
#For each batch
for j in range(batches):
    if ((len(data['Name']) - batch_size*j) >= batch_size):
        N = batch_size
        images = torch.zeros(batch_size, pixels+4)
    else:
        N = int(len(data['Name']) % batch_size)
        images = torch.zeros(int(len(data['Name']) % batch_size), pixels+4)
    #For each image in a batch
    for i in range(N):
        file_path = os.path.join(image_directory, data['Name'][batches*j + i] + '.jpg')
        #directory = '/Users/shivamodeka/Desktop/Machine-Learning-Algorithms/HAM10000/' + data['Name'][batches*j + i] + '.jpg'
        im = Image.open(file_path, 'r')
        pix_val = list(im.getdata())
        #Flatten the image and convert to grayscale values from RGB
        pix_val_flat = [sum(sets)/3 for sets in pix_val]
        pix_val_flat.append(data['Age'][batches*j + i])
        pix_val_flat.append(data['Male'][batches*j + i])
        pix_val_flat.append(data['Female'][batches*j + i])
        pix_val_flat.append(data['Label'][batches*j + i])
        images[i, :] = torch.tensor(pix_val_flat)
    print("Saving file " + str(j))
    #Save the batch file
    np.savetxt(os.path.join(save_file_directory, 'batch' + str(j) + '.csv'), images.numpy(), delimiter = ",")

