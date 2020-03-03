import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import datetime
import os
import numpy as np
from autoencoder import ConvAutoEncoder
import torchvision, torch
from torchvision.transforms import transforms
import tensorflow as tf

def training(traindata_dir, testdata_dir, log_dir='log', epochs=50, batch_size=16):

    os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(traindata_dir):
        print(traindata_dir + 'not found!')
        exit()
    if not os.path.exists(testdata_dir):
        print(testdata_dir + 'not found!')
        exit()
    tranforms = transforms.Compose([transforms.Resize(128,128), transforms.Grayscale(), transforms.Normalize()])
    trainfiles = torchvision.datasets.ImageFolder(traindata_dir, transform=transforms)
    testfiles = torchvision.datasets.ImageFolder(testdata_dir, transform=transforms)

    train_loader = torch.utils.data.DataLoader(trainfiles, batch_size=32, num_workers=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testfiles, batch_size=1, num_workers=0, shuffle=False)
    print(len(train_loader))
    print(len(test_loader))
    exit()
    #Reshape for training
    X_in = X_in[:,:,:]
    X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
    X_ou = X_ou[:,:,:]
    X_ou = X_ou.reshape(X_ou.shape[0],X_ou.shape[1],X_ou.shape[2],1)

    X_train, X_test, y_train, y_test = train_test_split(X_in, X_ou, test_size=0.10, random_state=42)

    net = ConvAutoEncoder(weights_path=log_dir + '/weight')
    #generator_nn.summary()
    net.fit(X_train, y_train, X_test, y_test)
    net.save_weights()

training(traindata_dir='../mnist_png/training/', testdata_dir='../mnist_png/testing/' )