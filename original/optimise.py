import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier #meta-estimator that fits a classifier then fits more copies of it with adjusted weights
from sklearn.tree import DecisionTreeClassifier #the clasifier that will be fitted multiple times by Ada
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split #function to create a training and testing dataset randomly
from sklearn.model_selection import RepeatedStratifiedKFold 
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.inspection import permutation_importance

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
#df1 = df1.head(5000)
#df2 = df2.head(5000)
#df3 = df3.head(5000)
#df4 = df4.head(5000)

frames = [df1, df2, df3, df4] #array of the frames i am merging
merged_df = pd.concat(frames) #concat appends the columns of each dataframe 

merged_df = merged_df.drop(merged_df[merged_df.n9 <= 8].index) #apply a cut
print(merged_df)

#make y first by copying the label column and making into its own dataframe
label_column = merged_df[['label']]
y = label_column.copy()
y = y.to_numpy() #converts y from a dataframe to a numpy array
y = y.flatten()  #flattens to a 1d array to use in the classifier

#need X to be data without the labels so need to drop label column

X = merged_df.drop(['n9','label'], axis=1)
print(X)
print(y)

#split the data into training and testing
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 1)


#AdaBoost
#ada_classifier = AdaBoostClassifier(
#                                DecisionTreeClassifier(max_depth=1),
#                               n_estimators=200
#)
#max depth says each tree is one decision with two options
#n_estimators says the number of these trees

#ada_classifier.fit(train_X, train_y) #fits the classifier to the training data

#predict the label based on the fit
#ada_predictions = ada_classifier.predict(test_X)

#predicting the probabilty of what an event will be classified as
#ada_pred_prob = ada_classifier.predict_proba(test_X)

#gradient boosting classifier

#grad_classifier = GradientBoostingClassifier(
#                                n_estimators=200, learning_rate=1.0,
#                                max_depth=1, random_state=0)

#grad_classifier.fit(train_X, train_y)

#grad_classifier.fit(train_X, train_y)

#grad_predictions = grad_classifier.predict(test_X)
#grad_pred_prob = grad_classifier.predict_proba(test_X)
#evaluate success with a confusion matrix
#print(confusion_matrix(test_y, predictions))

#create a classification report to show how precise each category was
#print(classification_report(test_y, ada_predictions))
#print(classification_report(test_y, grad_predictions))

#ada_pp_signal = ada_pred_prob[:,1]  #puts all of the probabilities of signal into an array

#new adaboosting with optimised parameters:
ada_op_classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), 
					n_estimators=100, learning_rate=0.1
)

ada_op_classifier.fit(train_X, train_y)
ada_op_predictions = ada_op_classifier.predict(test_X)
ada_op_predprob = ada_op_classifier.predict_proba(test_X)
adaop_signal = ada_op_predprob[:,1]
#grad_pp_signal = grad_pred_prob[:,1]
#plot confusion matrix
print('Confusion Matrix: ' ,confusion_matrix(test_y, ada_op_predictions))
#disp = plot_confusion_matrix(ada_op_classifier, test_X, test_y)
#disp.figure_.suptitle("Confusion Matrix")
#plt.show()
#plt.savefig("Confusion_matrix_oldfulldata_optimised")
print('Classification Report: ' , classification_report(test_y, ada_op_predictions))
print('Classifier score: ' , ada_op_classifier.score(test_X, test_y, sample_weight=None))

importance = permutation_importance(ada_op_classifier, test_X, test_y, n_repeats=30, random_state=0)
print('Permutation importances: ', importance.importances_mean)

#make the roc curve numbers
#ada_false_pr, ada_true_pr, ada_thresholds = roc_curve(test_y, ada_pp_signal)
#grad_false_pr, grad_true_pr, grad_thresholds = roc_curve(test_y, grad_pp_signal)
#adaop_falsepr, adaop_truepr, adaop_thresholds = roc_curve(test_y, adaop_signal)
#plot the curve

#plt.plot(ada_false_pr, ada_true_pr, marker='.', label='AdaBoost')
#pyplot.plot(grad_false_pr, grad_true_pr, linestyle='-', label='Gradient Boosting')
#plt.plot(adaop_falsepr, adaop_truepr, linestyle='-', label='Optimitsed AdaBoost')

#axis labels
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.legend()
#plt.show()

#Grid search code
#grid = dict()
#grid['n_estimators'] = [10, 50, 100, 200, 500]
#grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]

#define the evaluation procedure
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#define the grid search procedure
#grid_search = GridSearchCV(estimator=ada_classifier, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

#execute the grid search
#grid_result = grid_search.fit(test_X, test_y)

#summarise the best score and config
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#summarise all scores that were evaluated
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means,stds, params):
#	print("%f (%f) with: %r" % (mean, stdev, param))

 
