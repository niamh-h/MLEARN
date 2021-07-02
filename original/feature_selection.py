import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier #meta-estimator that fits a classifier then fits more copies of it with adjusted weights
from sklearn.tree import DecisionTreeClassifier #the clasifier that will be fitted multiple times by Ada
from sklearn.model_selection import train_test_split #function to create a training and testing dataset randomly
from sklearn import metrics
from matplotlib import pyplot
from sklearn.inspection import permutation_importance

#code from dataorganise.py, organises the data from the raw text files into a dataset
data1 = pd.read_csv("newsignalRawData.txt", sep=" ", header=None) #read in data
data1.columns = ["n9", "inner_hit", "dt_prev_us", "beta_one", "beta_two", "beta_three", "beta_four", "beta_five", "beta_six"]  #name the columns
df1 = pd.DataFrame(data1)                               #put into a data frame

data2 = pd.read_csv("newli9RawData.txt", sep=" ", header=None) #same for each background source
data2.columns = ["n9", "inner_hit", "dt_prev_us", "beta_one", "beta_two", "beta_three", "beta_four", "beta_five", "beta_six"]
df2 = pd.DataFrame(data2)

data3 = pd.read_csv("newn17RawData.txt", sep=" ", header=None)
data3.columns = ["n9", "inner_hit", "dt_prev_us", "beta_one", "beta_two", "beta_three", "beta_four", "beta_five", "beta_six"]
df3 = pd.DataFrame(data3)

data4 = pd.read_csv("newworldRawData.txt", sep=" ", header=None)
data4.columns = ["n9", "inner_hit", "dt_prev_us", "beta_one", "beta_two", "beta_three", "beta_four", "beta_five", "beta_six"]
df4 = pd.DataFrame(data4)

data5 = pd.read_csv("newsmall_reactorRawData.txt", sep=" ", header=None)
data5.columns = ["n9", "inner_hit", "dt_prev_us", "beta_one", "beta_two", "beta_three", "beta_four", "beta_five", "beta_six"]
df5 = pd.DataFrame(data5)

#data6 = pd.read_csv("newgeoRawData.txt", sep=" ", header=None)
#data6.columns = ["n9", "inner_hit", "dt_prev_us", "beta_one", "beta_two", "beta_three", "beta_four", "beta_five", "beta_six"]
#df6 = pd.DataFrame(data6)

#code from bg_program.py runs the adaboost classifier
df1['label'] = 1
df2['label'] = 0
df3['label'] = 0
df4['label'] = 0
df5['label'] = 0

df1 = df1.head(5000)
df2 = df2.head(5000)
df3 = df3.head(5000)
df4 = df4.head(5000)
df5 = df5.head(5000)

frames = [df1, df2, df3, df4, df5] 
merged_df = pd.concat(frames) 
print(merged_df)

label_column = merged_df[['label']]
y = label_column.copy()
y = y.to_numpy() #converts y from a dataframe to a numpy array
y = y.flatten()  #flattens to a 1d array to use in the classifier

X = merged_df.drop(['label'], axis=1)

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 1)

#construct and fit the adaboost classifier to the training set - THIS IS THE OPTIMSED VERSION OF ADABOOST
ada_classifier = AdaBoostClassifier(
                                DecisionTreeClassifier(max_depth=1),
                                n_estimators=100, learning_rate=0.1
)

ada_classifier.fit(train_X, train_y)

ada_predictions = ada_classifier.predict(test_X)

ada_pred_prob = ada_classifier.predict_proba(test_X)

#finding the importance scores for each feature
importance = ada_classifier.feature_importances_

importance2 = permutation_importance(ada_classifier, test_X, test_y,n_repeats=10, random_state=0)

#summarise them
print(importance)

print(importance2.importances_mean)


