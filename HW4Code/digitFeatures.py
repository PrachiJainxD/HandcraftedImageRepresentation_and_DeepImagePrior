import numpy as np

#Helper Functions
def computeL2Norm(features):    
    sumF = 0
    for i in range(len(features)):
        square = np.square(features[i])
        sumF += square
    features = features/(np.sqrt(sumF)+1e-6)
    return features 
    
def calculateGradient(img, template):
    ts = template.size #Number of elements in the template
    
    #New padded array to hold the resultant gradient image
    newImg = np.zeros((img.shape[0]+ts-1, img.shape[1]+ts-1))
    newImg[np.uint16((ts-1)/2.0):img.shape[0]+np.uint16((ts-1)/2.0), np.uint16((ts-1)/2.0):img.shape[1]+np.uint16((ts-1)/2.0)] = img
    result = np.zeros((newImg.shape))
    
    for r in np.uint16(np.arange((ts-1)/2.0, img.shape[0]+(ts-1)/2.0)):
        for c in np.uint16(np.arange((ts-1)/2.0, img.shape[1]+(ts-1)/2.0)):
            currentRegion = newImg[r-np.uint16((ts-1)/2.0):r+np.uint16((ts-1)/2.0)+1, c-np.uint16((ts-1)/2.0):c+np.uint16((ts-1)/2.0)+1]
            currentResult = currentRegion * template
            score = np.sum(currentResult)
            result[r, c] = score
    
    #Result of the same size as the original image after removing the padding.
    resultImg = result[np.uint16((ts-1)/2.0):result.shape[0]-np.uint16((ts-1)/2.0), np.uint16((ts-1)/2.0):result.shape[1]-np.uint16((ts-1)/2.0)]
    return resultImg
    
def gradientMagnitude(horizontalGradient, verticalGradient):
    gradMagnitude = np.sqrt(np.power(horizontalGradient, 2) + np.power(verticalGradient, 2))
    return gradMagnitude

def gradientDirection(horizontalGradient, verticalGradient):
    gradDirection = np.arctan(verticalGradient/(horizontalGradient + 1e-8))
    gradDirection = np.rad2deg(gradDirection)
    return gradDirection   
    
def getFeature(x, featureType):
    if featureType == 'pixel':
        features = pixelFeatures(x)

    elif featureType == 'hog':
        features = hogFeatures(x)

    elif featureType == 'lbp':
        features = lbpFeatures(x)

    return features
    
def pixelFeatures(x):
    '''
    Takes x images as input and returns their pixel values as a feature
    Output is an array of size 784 x N where N is the number of images in x
    '''
    features = np.reshape(x, (784, x.shape[2])) 
    
    #Chosen method works best - outperforms accuracy as mentioned in HW4
    method = 0
    
    if method == 0:
    #Performing standardization so that the raw pixel values when fed into the network does not 
    #slows down the learning due to the erratic input nature.    
        meanF = features.mean().astype(np.float32)
        stdF = features.std().astype(np.float32)
        features = (features - meanF)/(stdF)
    
    elif method == 1:
    #Square-root scaling
        features = np.sqrt(abs(features))
    
    elif method == 2:
    #L2-normalization
        features = computeL2Norm(features) 
    return features

def hogFeatures(x):
    '''
    Histogram of oriented gradients (HoG)
    Takes x images as input and returns their HoG features
    '''
    #Define Parameters
    binSize = 4
    numOri = 8
    horizontalMask = np.array([-1, 0, 1])
    verticalMask = np.array([[-1],[0],[1]]) 

    numFeats = x.shape[0]
    numImages = x.shape[2]
    
    #Store features of all images
    features = []
        
    # For all images
    for image in range(numImages):
    
        y = x[:,:,image]
        
        #According to HOG paper of Dalal and Triggs
        #STEP 1: Logarithm of the intensity values(Non-linear Mapping)
        #Additional step for more invariance
        y = np.log(y + 1)
        
        #STEP 2: Compute Gradients Gx, Gy, magnitude and angle
        
        #Calculate Horizontal and Vertical Gradients
        Gx = calculateGradient(y, horizontalMask)
        Gy = calculateGradient(y, verticalMask)
                
        # Getting Magnitude and Direction of H and V Gradients
        # Angle lies between [-pi/2, pi/2]
        
        GMagnitude = gradientMagnitude(Gx, Gy)
        GDirection = gradientDirection(Gx, Gy)
       
        # Store features of each image, then concatenate it with other image features to form a single matrix
        blockFeature = []
        
        #STEP 3: Weighted Vote into spatial and orientation cells
        # For the grid of 7 x 7 => numFeats//binSize x numFeats//binSize => 28//4 x 28//4
        for i in range(numFeats//binSize):
            for j in range(numFeats//binSize):
                
                # For each cell get the magnitudes and directions for that cell
                magPatch = GMagnitude[binSize*i:binSize*i+binSize, binSize*j:binSize*j+binSize]
                angPatch = GDirection[binSize*i:binSize*i+binSize, binSize*j:binSize*j+binSize]

                # Store histogram for each cell here
                histogram  = np.zeros([8,1])
                
                # Binning Process (Bi-Linear Interpolation)
                for p in range(binSize):
                    for q in range(binSize):
                        mag = magPatch[p,q]
                        angle = angPatch[p,q]
                        
                        #Assigning each pixel to the nearest orientation in numOri 
                        #With vote proportional to the gradient magnitude
                        
                        if angle>=-90 and angle<-67.5:
                            histogram[0] = histogram[0] + mag *(-67.5 - angle)/22.5
                            histogram[1] = histogram[1] + mag *(angle + 90)/22.5
                        elif angle>=-67.5 and angle<-45:
                            histogram[1] = histogram[1] + mag * (-45 - angle)/22.5 
                            histogram[2] = histogram[2] + mag * (angle + 67.5)/22.5
                        elif angle>=-45 and angle<-22.5:
                            histogram[2] = histogram[2] + mag * (-22.5 - angle)/22.5 
                            histogram[3] = histogram[3] + mag * (angle + 45)/22.5
                        elif angle>=-22.5 and angle<0:
                            histogram[3] = histogram[3] + mag * (0 - angle)/22.5 
                            histogram[4] = histogram[4] + mag * (angle + 22.5)/22.5
                        elif angle>=0 and angle<22.5:
                            histogram[4] = histogram[4] + mag * (22.5 - angle)/22.5 
                            histogram[5] = histogram[5] + mag *(angle - 0)/22.5
                        elif angle>=22.5 and angle<45:
                            histogram[5] = histogram[5] + mag *(45-angle)/22.5
                            histogram[6] = histogram[6] + mag *(angle-22.5)/22.5
                        elif angle>=45 and angle<67.5:
                            histogram[6] = histogram[6] + mag *(67.5 - angle)/22.5 
                            histogram[7] = histogram[7] + mag *(angle - 45)/22.5
                        elif angle>=67.5 and angle<90:
                            histogram[7]= histogram[7] + mag *(90 - angle)/22.5
                            histogram[0]= histogram[0] + mag *(angle - 67.5)/22.5

                #Concatenate histograms
                blockFeature.append(histogram)
        
        #STEP 4: Local contrast normalization  of feature vector over overlapping spatial blocks 
        #Additional step for more invariance
        blockFeature = np.array(blockFeature)
        #Chosen method works best - outperforms accuracy as mentioned in HW4
        method = 1 
    
        if method == 0:
        #Performing standardization so that the raw pixel values when fed into the network does not 
        #slows down the learning due to the erratic input nature.    
            meanF = blockFeature.mean().astype(np.float32)
            stdF = blockFeature.std().astype(np.float32)
            blockFeature = (blockFeature - meanF)/(stdF)
    
        elif method == 1:
        #Square-root scaling
            blockFeature = np.sqrt(abs(blockFeature))
    
        elif method == 2:
        #L2-normalization
            blockFeature = computeL2Norm(blockFeature) 
    
        #Concatenate HoG features of each image
        features.append(blockFeature.flatten())    
    
    featuresT = np.array(features).T
    return featuresT         

def lbpFeatures(x):
    '''
    Takes x images as input and returns a local binary pattern (LBP) histogram for each image
    '''    
    height = x.shape[0]
    width = x.shape[1]
    channel = x.shape[2]
    patchSize = 3    
    
    localBinaryPatternMap = np.array([[1, 2, 4],[8, 0, 16], [32, 64, 128]])    
    localBinaryPattern = np.zeros((256, channel))
    
    #Each 3x3 patch is mapped to a number between 0 and 255 as follows: 
    #Consider the pixels in a 3x3 patch. 
    #Subtract from each pixel the value of the center pixel and assign them a bit value of 
    #1 if the value of the pixel is greater than zero, and a bit value of 0 otherwise. 
    #This gives us a 8-bit representation (excluding the center which is always 0) of every 3x3 patch, 
    #corresponding to an integer between 0-255.
    for n in range(channel):
        for h in range(height-patchSize+1):
            for w in range(width-patchSize+1):        
                localBinaryPattern[np.sum(localBinaryPatternMap * np.array(x[h:h+3,w:w+3,n] > x[h+1,w+1,n])), n] += 1

    features = localBinaryPattern         
    #Chosen method works best - outperforms accuracy as mentioned in HW4
    method = 1
    
    if method == 0:
    #Performing standardization so that the raw pixel values when fed into the network does not 
    #slows down the learning due to the erratic input nature.    
        meanF = features.mean().astype(np.float32)
        stdF = features.std().astype(np.float32)
        features = (features - meanF)/(stdF)
    
    elif method == 1:
    #Square-root scaling
        features = np.sqrt(abs(features))
    
    elif method == 2:
    #L2-normalization
        features = computeL2Norm(features) 
    
    return features
    
def zeroFeatures(x):
    return np.zeros((10, x.shape[2]))    