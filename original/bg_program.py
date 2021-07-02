import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier #meta-estimator that fits a classifier then fits more copies of it with adjusted weights
from sklearn.tree import DecisionTreeClassifier #the clasifier that will be fitted multiple times by Ada
from sklearn.metrics import confusion_matrix #shows the success of the algorithm
from sklearn.model_selection import train_test_split #function to create a training and testing dataset randomly
from sklearn.metrics import classification_report 

from sklearn.metrics import roc_curve   #the things to make the roc curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot 

#code from dataorganise.py, organises the data from the raw text files into a dataset
data1 = pd.read_csv("signalRawData.txt", sep=" ", header=None) #read in data
data1.columns = ["n9", "inner_hit", "dt_prev_us"]               #name the columns
df1 = pd.DataFrame(data1)                               #put into a data frame

data2 = pd.read_csv("li9RawData.txt", sep=" ", header=None) #same for each background source
data2.columns = ["n9", "inner_hit", "dt_prev_us"]
df2 = pd.DataFrame(data2)

data3 = pd.read_csv("n17RawData.txt", sep=" ", header=None)
data3.columns = ["n9", "inner_hit", "dt_prev_us"]
df3 = pd.DataFrame(data3)

data4 = pd.read_csv("geoRawData.txt", sep=" ", header=None)
data4.columns = ["n9", "inner_hit", "dt_prev_us"]
df4 = pd.DataFrame(data4)

df1['label'] = 1
df2['label'] = 0
df3['label'] = 0
df4['label'] = 0 #labels each frame as signal or background so the success of the program can be determined

#to save time for editing the program, take a sample of the first 10000 events 
df1 = df1.head(5000)
df2 = df2.head(5000)
df3 = df3.head(5000)
df4 = df4.head(5000)


frames = [df1, df2, df3, df4] #array of the frames i am merging
merged_df = pd.concat(frames) #concat appends the columns of each dataframe 

merged_df = merged_df.drop(merged_df[merged_df.n9 <= 8].index) #apply a cut


#now writing the adaboost program to sort the background from the signal
#need to split the data into X(containing the data for all files) and Y
#(containing the labels of each row)

#make y first by copying the label column and making into its own dataframe
label_column = merged_df[['label']]
y = label_column.copy()
y = y.to_numpy() #converts y from a dataframe to a numpy array
y = y.flatten()  #flattens to a 1d array to use in the classifier

#need X to be data without the labels so need to drop label column

X = merged_df.drop(['label'], axis=1)

#split the data into training and testing
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 1)


#construct and fit the model to the training set 
classifier = AdaBoostClassifier(
				DecisionTreeClassifier(max_depth=1), 
				n_estimators=200
)
#max depth says each tree is one decision with two options
#n_estimators says the number of these trees

classifier.fit(train_X, train_y) #fits the classifier to the training data

#predict the label based on the fit
predictions = classifier.predict(test_X)

#predicting the probabilty of what an event will be classified as
pred_prob = classifier.predict_proba(test_X)

#evaluate success with a confusion matrix
#print(confusion_matrix(test_y, predictions))

#create a classification report to show how precise each category was
#print(classification_report(test_y, predictions))

#predicted_signal = predictions[:,1] #puts all of the times it predicted signal (1) into a dataframe

#pred_prob_signal = pred_prob[:,1]  #puts all of the probabilities of signal into an array

#make the roc curve numbers
#false_pr, true_pr, thresholds = roc_curve(test_y, pred_prob_signal)
#plot the curve
#pyplot.plot(false_pr, true_pr, marker='.', label='AdaBoost')
#axis labels
#pyplot.xlabel('False Positive Rate')
#pyplot.ylabel('True Positive Rate')
#pyplot.legend()
#pyplot.show()

#pred_prob_background = pred_prob[:,0] #puts all of the probabilities of background into a data frame

