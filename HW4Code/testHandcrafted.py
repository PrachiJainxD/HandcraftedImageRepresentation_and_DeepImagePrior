import numpy as np
import os
import utils
import time

import digitFeatures
import linearModel

# There are three versions of MNIST dataset
dataTypes = ['digits-normal.mat', 'digits-scaled.mat', 'digits-jitter.mat']

# You have to implement three types of features
featureTypes = ['pixel', 'hog', 'lbp']

# Accuracy placeholder
accuracy = np.zeros((len(dataTypes), len(featureTypes)))
trainSet = 1
testSet = 3

for i in range(len(dataTypes)):
    dataType = dataTypes[i]

    #Load data
    path = os.path.join('C:/Users/14136/OneDrive/Desktop/670 Computer vision/hw4_release/data', dataType)
    data = utils.loadmat(path)
    print('+++ Loading dataset: {} ({} images)'.format(dataType, data['x'].shape[2]))

    # To montage the digits in the val set use
    #utils.montageDigits(data['x'][:, :, data['set']==2])

    for j in range(len(featureTypes)):
        featureType = featureTypes[j]

        # Extract features
        tic = time.time()
        features = digitFeatures.getFeature(data['x'], featureType)
        
        print('{:.2f}s to extract {} features ({} dim).'.format(time.time()-tic, featureType, features.shape[0]))

        # Train model
        tic = time.time()
        
        #output, bestParam = linearModel.KFoldCV(features[:, data['set']==trainSet], data['y'][data['set']==trainSet])
        #print(bestParam) #{'maxiter': 2500, 'lambda': 0.1, 'eta': 0.05}
        
        model = linearModel.train(features[:, data['set']==trainSet], data['y'][data['set']==trainSet])
        print('{:.2f}s to train model.'.format(time.time()-tic))

        # Test the model
        tic = time.time()
        ypred = linearModel.predict(model, features[:, data['set']==testSet])
        print('{:.2f}s to test model.'.format(time.time()-tic))
        y = data['y'][data['set']==testSet]

        # Measure Accuracy and Display Confusion Matrix
        (acc, conf) = utils.evaluateLabels(y, ypred, False)
        print('Accuracy [testSet={}] {:.2f} %\n'.format(testSet, acc*100))
        accuracy[i, j] = acc


# Print the results in a table
print('+++ Accuracy Table [trainSet={}, testSet={}]'.format(trainSet, testSet))
print('--------------------------------------------------')
print('dataset\t\t\t', end="")
for j in range(len(featureTypes)):
    print('{}\t'.format(featureTypes[j]), end="")

print()
print('--------------------------------------------------')
for i in range(len(dataTypes)):
    print('{}\t'.format(dataTypes[i]), end="")
    for j in range(len(featureTypes)):
        print('{:.2f}\t'.format(accuracy[i, j]*100), end="")
    print()

# Once you have optimized the hyperparameters, you can report test accuracy
# by setting testSet=3. Do not optimize your hyperparameters on the
# test set. That would be cheating!