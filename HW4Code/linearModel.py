import numpy as np
import utils

def softmax(z):  
  exp = np.exp(z - np.max(z))
  for i in range(len(z)):
    exp[i]/=np.sum(exp[i])
  return exp
  
def oneHotEncoding(y, classLabel):
    #Assigning one to relevant class labels each belonging to respective digit
    yEncode = np.zeros((len(y), classLabel))
    yEncode[np.arange(len(y)), y] = 1
    return yEncode

def KFoldCV(x,y):
    classLabels = np.unique(y)
    classID = []
    classSamples = []
    
    for i in range(max(classLabels)+1):
        indices = np.where(y==i)[0]
        classID.append(indices)
        classSamples.append(len(indices))
    
    trainFold = []
    valFold = []
    for fold in range(5):
        tempTrain = []
        tempVal = []
        for i in range(max(classLabels)+1):
            stepSize = int(classSamples[i]/5)
            start = fold * (stepSize)
            valClassID = classID[i][start:start + stepSize]
            trainClassID = list((set(classID[i]) - set(valClassID)))
            tempVal.extend(valClassID)
            tempTrain.extend(trainClassID)
        trainFold.append(tempTrain)
        valFold.append(tempVal)

    lambdaS = [0.1, 0.01]
    maxiterS = [2000, 2500, 3000]
    etaS = [0.1, 0.01, 0.05]
    output = {'maxiter': [],'lambda': [], 'eta': [], 'accuracy':[]}
    
    for m in maxiterS:
        for l in lambdaS:
            for e in etaS:
                modelParams = {'maxiter': m, 'lambda': l, 'eta': e}
                output['maxiter'].append(m)
                output['lambda'].append(l)                
                output['eta'].append(e)
                avgAccuracy= 0
                for fold in range(5):
                    xTrain = np.apply_along_axis(lambda cv: cv[trainFold[fold]], 1, x)
                    yTrain = np.apply_along_axis(lambda cv: cv[trainFold[fold]], 0, y)
                    xTest = np.apply_along_axis(lambda cv: cv[valFold[fold]], 1, x)
                    yTest = np.apply_along_axis(lambda cv: cv[valFold[fold]], 0, y)
                    model = train(xTrain, yTrain, param = modelParams)
                    ypred = predict(model, xTest)
                    (acc, conf) = utils.evaluateLabels(yTest, ypred, False)
                    avgAccuracy += acc
                avgAccuracy = np.mean(avgAccuracy)
                output['accuracy'].append(avgAccuracy)
                print('5 Fold Cross Validation Accuracy ={}'.format(avgAccuracy))

    indices = np.argmax(np.array(output['accuracy']))
    bestParam = {'maxiter':output['maxiter'][indices],'lambda': output['lambda'][indices],'eta': output['eta'][indices]}
    return output, bestParam
    
def train(x, y, param = None):
    if param is None:
        param = {}       
        param['lambda'] = 0.1      # Regularization term
        param['maxiter'] = 2500    # Number of iterations
        param['eta'] = 0.05        # Learning rate
        param['addBias'] = True    # Weather to add bias to features
    else:
        param['lambda'] = param['lambda']
        param['maxiter'] = param['maxiter']
        param['eta'] = param['eta']
        param['addBias'] = True
    return multiclassLRTrain(x, y, param)

def predict(model, x):
    #Returns highest probability class
    yHat = softmax(x.T@model['w'].T + model['b'])
    return np.argmax(yHat, axis=1)

def multiclassLRTrain(x, y, param):
    classLabels = np.unique(y)
    numClass = classLabels.shape[0]
    numFeats = x.shape[0]
    numData = x.shape[1] 

    # Randomly initializes a model
    model = {}
    model['w'] = np.random.randn(numClass,numFeats) * 0.01
    if  param['addBias'] == True:
        model['b'] = np.random.randn(numClass) * 0.01
    else:
        model['b'] = np.zeros(numClass)
        
    model['classLabels'] = classLabels  
    w = model['w'] 
    b = model['b']
    lossEpoch = []
    
    for epoch in range(param['maxiter']):    
        z = x.T@w.T + b
        dW = (1/numData) * np.dot(x, softmax(z) - oneHotEncoding(y, numClass)).T + (2*param['lambda']/numData)*np.linalg.norm(w)
        dB = (1/numData) * np.sum(softmax(z) - oneHotEncoding(y, numClass))
        
        w = w - param['eta'] * dW
        b = b - param['eta'] * dB
      
        loss = -(np.log(softmax(z)[np.arange(len(y)), y])).mean() + ((1/numData)*param['lambda'] * np.square(np.linalg.norm(w)))        
        lossEpoch.append(loss)
        #print('Epoch:{}.......................Loss:{:.4f}'.format(epoch+1, loss))
    
    model['w'] = w
    model['b'] = b
    return model    

    