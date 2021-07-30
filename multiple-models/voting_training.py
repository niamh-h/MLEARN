import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, roc_curve
from collections import Counter 
#import the models
li9_clf = joblib.load('li9_classifier.sav')
world_clf = joblib.load('world_classifier.sav')
n17_clf = joblib.load('n17_classifier.sav')
neu_clf = joblib.load('neu_classifier.sav')
geo_clf = joblib.load('geo_classifier.sav')
#load the data
data1 = pd.read_csv("bigfilename.txt", sep=" ", header=None)
data1.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
		"beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df1 = pd.DataFrame(data1)

data2 = pd.read_csv("li9filename.txt", sep=" ", header=None)
data2.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
		"beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df2 = pd.DataFrame(data2)

data3 = pd.read_csv("n17filename.txt", sep=" ", header=None)
data3.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
		"beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df3 = pd.DataFrame(data3)

data4 = pd.read_csv("smallfilename.txt", sep=" ", header=None)
data4.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
		"beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df4 = pd.DataFrame(data4)

data5 = pd.read_csv("worldfilename.txt", sep=" ", header=None)
data5.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
		"beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df5 = pd.DataFrame(data5)

data6 = pd.read_csv("neufilename.txt", sep=" ", header=None)
data6.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
		"beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df6 = pd.DataFrame(data6)

data7 = pd.read_csv("geofilename.txt", sep=" ", header=None)
data7.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
                "beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df7 = pd.DataFrame(data7)
df1['label'] = 1
df2['label'] = 0
df3['label'] = 0
df4['label'] = 1
df5['label'] = 0
df6['label'] = 0
df7['label'] = 0

frames = [df2,df3,df4,df5, df6, df7]
X = df1.append(frames, ignore_index=True)
#X = X.drop(X[X.dt_prev_us < 0].index) #optional cuts
#X = X.drop(X[X.closestPMT < 0].index)
X.index = range(len(X))
#create the target values
label = X[['label']]
label = label.to_numpy()
label = label.flatten()
X = X.drop(['label'], axis=1)
train_X, test_X, train_y, test_y = train_test_split(X, label)
print('Training: ', Counter(train_y))
print('Testing: ', Counter(test_y))

vot_clf = VotingClassifier(estimators=[('geo', geo_clf), ('neu', neu_clf), ('li9', li9_clf),
			('n17', n17_clf),('world',world_clf)],voting='soft')
vot_clf.fit(train_X, train_y)

pred = vot_clf.predict(test_X)
prob = vot_clf.predict_proba(test_X)
print(confusion_matrix(test_y, pred))
print(classification_report(test_y, pred))
disp = plot_confusion_matrix(vot_clf, test_X, test_y)
disp.figure_.suptitle("Voting Classifier (training)")
plt.show()
prob_signal = prob[:,1]
falsepr, true_pr, thresh = roc_curve(test_y, prob_signal, pos_label=1)
plt.plot(falsepr, true_pr, marker='.', label='VotingClassifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('Correlated Background Classifier (training)')
plt.show()

filename = 'voting_classifier.sav'
joblib.dump(vot_clf, filename)

