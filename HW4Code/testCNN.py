import numpy as np
import os
import utils
import time
import digitFeatures
import linearModel
from CNNModel import CNN
from torch import optim
import torch.nn as nn
import torch
import pdb
import random

# There are three versions of MNIST dataset
dataTypes = ['digits-normal.mat', 'digits-scaled.mat', 'digits-jitter.mat']

# Accuracy placeholder
accuracy = np.zeros(len(dataTypes))
trainSet = 1
testSet = 2

iterations = 1000
batchSize = [40]#[40, 50, 100, 200]

for i in range(len(dataTypes)):
    dataType = dataTypes[i]

    #Load data
    path = os.path.join('..', 'data', dataType)
    data = utils.loadmat(path)
    print('+++ Loading dataset: {} ({} images)'.format(dataType, data['x'].shape[2]))

    # Organize into numImages x numChannels x width x height
    x = data['x'].transpose([2,0,1])
    x = np.reshape(x,[x.shape[0], 1, x.shape[1], x.shape[2]])
    y = data['y']
    # Convert data into torch tensors
    x = torch.tensor(x).float()
    y = torch.tensor(y).long() # Labels are categorical

    # Define the model
    model = CNN()
    model.train()
    #print(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    # Start training
    xTrain = x[data['set']==trainSet,:,:,:]
    yTrain = y[data['set']==trainSet]
    
    # Loop over training data in some batches
    # Implement this
    for epoch in range(iterations):
        batch = batchSize[random.randint(0, len(batchSize) - 1)]
        totalTrain = int(xTrain.shape[0]/batch)
        count = 0
        for k in range(0, xTrain.shape[0], batch):
          count = count + 1
          #Forward Prpogation
          optimizer.zero_grad()
          #Using permutations without replacement
          indices = torch.randperm(xTrain.shape[0])[k:k + batch]
          xBatch, yBatch = xTrain[indices], yTrain[indices]
          output = model(xBatch)
          loss = criterion(output, yBatch)
          
          #Backward Pass
          loss.backward()
          optimizer.step()
          #Logging
          if epoch%10 == 0:
            print("Epoch:{} Batch:{}/{}..............Loss={}".format(epoch+1, count, totalTrain, loss.item()))          

    # Test model
    xTest = x[data['set']==testSet,:,:,:]
    yTest = y[data['set']==testSet]

    yPred = np.zeros(yTest.shape[0])
    model.eval() # Set this to evaluation mode
    
    # Loop over xTest and compute labels (implement this)
    yPredict = []
    with torch.no_grad():
      testOutput = model(xTest)
      predict, index = torch.max(testOutput.data, 1)
      yPredict = list(index)
    
    yPred = np.array(yPredict)
    
    # Map it back to numpy to use our functions
    yTest = yTest.numpy()
    (acc, conf) = utils.evaluateLabels(yTest, yPred, False)
    print('Accuracy [testSet={}] {:.2f} %\n'.format(testSet, acc*100))
    accuracy[i] = acc

# Print the results in a table
print('+++ Accuracy Table [trainSet={}, testSet={}]'.format(trainSet, testSet))
print('--------------------------------------------------')
print('dataset\t\t\t', end="")
print('{}\t'.format('cnn'), end="")
print()
print('--------------------------------------------------')
for i in range(len(dataTypes)):
    print('{}\t'.format(dataTypes[i]), end="")
    print('{:.2f}\t'.format(accuracy[i]*100))

# Once you have optimized the hyperparameters, you can report test accuracy
# by setting testSet=3. Do not optimize your hyperparameters on the
# test set. That would be cheating.