#Author : Golam Mostaeen | gom766 | golammostaeen@gmail.com


#ALL MY IMPORTS
import pandas as pd
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import operator
from collections import Counter
import itertools
#from sklearn.metrics import accuracy_score # you can't use sklearn in the final assignment.
#from sklearn.metrics import confusion_matrix # you can't use sklearn in the final assignment, write your own code.
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets






# Loading and preprocessing the dataset
df=pd.read_csv('knnDataSet.csv')
df['L'] = df['L'].astype('category')
df['TL'] = df['TL'].astype('category')
trainSet = df.dropna()
testSet = df.drop(trainSet.index)
#del trainSet['TL'], trainSet['Unnamed: 0']
#del testSet['L'], testSet['Unnamed: 0']
del trainSet['Unnamed: 0'], testSet['Unnamed: 0']
trainSet=trainSet.values
testSet=testSet.values



#np.random.shuffle(testSet)








###______________________________scatterPlotTheProvidedData________________________________

#This function scatter plots the supplied (train and test) data.
def scatterPlotTheProvidedData():
    plt.figure()
    plt.scatter(testSet[:,0], testSet[:,1], c=testSet[:,2], s=70, alpha=0.5, marker='x')
    plt.scatter(trainSet[:,0], trainSet[:,1], c=trainSet[:,2], s=80, alpha=1)
    plt.title('True Plot')














###______________________________knn________________________________________________________

# My KNN algorithm implementation as per the guideline of the Assignment.
# Parameters:
    #trainSet - Training Samples
    #xTest - Test Samples
    #K - The Number of Neighbours to look for
    #feedback_classification - Adds predicted test data to train sample if set to true
#Returns:
    #The corresponding predictions for each of the supplied test data samples.
def knn(trainSet, xTest, K, feedback_classification=False):
    predictions=[]
    # loop over all test samples
    for x in range(len(xTest)):
        # loop over all training samples
            # find the distance between test and train samples
        allDistancesToTrainSamples = []
        numberOfTrainSamples = len(trainSet)
        for i in range(numberOfTrainSamples):
            d = euclidDist(trainSet[i,0:2], xTest[x,0:2])
            allDistancesToTrainSamples.append((trainSet[i,3], d))


        # sort the distance and choose k nearest neighbour
        #allDistancesToTrainSamples.sort()
        allDistancesToTrainSamples.sort(key = operator.itemgetter(1))
        kNearestNeighbours = allDistancesToTrainSamples[:K]


        # assign the majority class to the current test sample
        kNearestNeighboursClasseLabels = [neighbour[0] for neighbour in kNearestNeighbours]
        count = Counter(kNearestNeighboursClasseLabels)

        #print(count.most_common()[0][0])
        predictedClassLevel = count.most_common()[0][0];
        predictions.append(predictedClassLevel)

        # if feedback is true then add the test sample to the
        # set with the predicted label
        if (feedback_classification == True):
            trainSet = np.append(trainSet, [[xTest[x,0],xTest[x,1],xTest[x,2],predictedClassLevel]], axis=0)


    # return the predicted labels of xTest
    return  predictions
 #   pass


















#___________________________________euclidDist_______________________________________

#This function calculates the euclid distance
#parameters:
    #a - The First point in array format
    #b - The Second point in array format
#returns:
    #floating point distance between those points

def euclidDist(a,b):
    return np.sqrt(np.sum((a-b)**2))



















#_________________________________________calculateAccuracy__________________________

#This function calculates accuracy.
#parameters:
    #testSet - the testSet list containing true labels
    #predictions - the corresponding prediction list of the test sets
#returns:
    #float value in percentage denoting the accuracy of the algorithm
def calculateAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][2] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0




















#_____________________________________________plot_confusion_matrix__________________________
#This function plots the confusion matrix
#parameters:
    #cm - the confusion matrix array
    #classes - class names in array to plot
    #normalize - normalize the conf matrix if set to true
    #title - Title of the conf matrix
    #cmap - color map of the plotting
#returns:
    #plots the confusion matrix provided.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')













#______________________________________plotDecisionBoundary____________________
#This function plots the decision boundary of the algorithm. Please change the K value as
#commented below for expected decision boundary.

# Create color maps
def plotDecisionBoundary():
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', "#F1C40F", '#17A589'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#9A7D0A', '#0E6251'])
    h = 0.01
    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.


        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = trainSet[:, 0].min() - 1, trainSet[:, 0].max() + 1
        y_min, y_max = trainSet[:, 1].min() - 1, trainSet[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = knn(trainSet,np.c_[xx.ravel(), yy.ravel()],1)#Call to implemented KNN. Change the last K value for expected Decision boundary.
        Z = np.asarray(Z)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)


         # Plot also the training points
        plt.scatter(testSet[:, 0], testSet[:, 1], c=testSet[:, 2], cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())













#_________________________________________________calculateConfusionMatrix____________________________
#This function calculates the confusion Matrix
#parameters:
    #testSet - the test set of the problem
    #tmp - the prediction list generated by call call to knn function
#returns:
    #the confusion matrix for the supplied test set and predictions.
def calculateConfusionMatrix(testSet, tmp):
    myConf = np.zeros((5,5))
    for c in range(len(testSet)):
        myConf[int(testSet[c,2]-1)][int(tmp[c]-1)] += 1
    return  myConf;













############################################################# DIFFERENT CALL TO ABOVE FUNCTIONS FOR ALGORITHM TESTINGS #################################
######################################## PLEASE CALL FUNCTIONS AS NEEDED FOR REQUIRED OUTPUTS #########################################################


#Scatter Plot the provided data
scatterPlotTheProvidedData()

#Calling the KNN function
tmp = knn(trainSet,testSet,1,False)

#printing the algorithm Accuracy.
print('Accuracy: ' , calculateAccuracy(testSet,tmp) , ' %')


#Printing the Confusion Matrix
cm = calculateConfusionMatrix(testSet, tmp)
print(cm)



#Plotting the confusion matrices both with normalization and normalization.
plt.figure()
plot_confusion_matrix(cm, classes=['0','1','2','3','4'],
                      title='Confusion matrix, K=1, Feedback and Shuffled Unlabeled Nodes(without Normalization )')
plt.figure()
plot_confusion_matrix(cm, classes=['0','1','2','3','4'],normalize=True,
                      title='Confusion matrix, K=1, Feedback and Shuffled Unlabeled Nodes(with Normalization)')



#Plotting the test data as per the algorithm prediction.
plt.figure()
plt.scatter(testSet[:,0], testSet[:,1], c=np.asarray(tmp), s=70, alpha=0.5, marker='x')
plt.scatter(trainSet[:,0], trainSet[:,1], c=trainSet[:,2], s=80, alpha=1)
plt.title('Predicted Plot')


#Plotting the decision boundary
#Please uncomment the following call for decision boundary plot.
#plotDecisionBoundary()



#showing all the plotting finally.
plt.show()




