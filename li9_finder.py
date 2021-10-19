import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score, confusion_matrix, auc
import matplotlib.pyplot as plt
import joblib
import sys

data1 = pd.read_csv("/path/to/heysham_2_data.txt", sep=" ", header=None)
data1.columns = ["n100", "n100_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev", "drPrevr"]
df1 = pd.DataFrame(data1)
df1['label'] = 1

data2 = pd.read_csv("/path/to/li9_data.txt", sep=" ", header=None)
data2.columns = ["n100", "n100_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev",  "beta_three",
                "beta_three_prev","beta_four", "beta_four_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev", "drPrevr"]
df2 = pd.DataFrame(data2)
df2['label'] = 0

data3 = pd.read_csv("/path/to/n17_data.txt", sep=" ", header=None)
data3.columns = ["n100", "n100_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev",  "drPrevr"]
df3= pd.DataFrame(data3)
df3['label'] = 0

data4 = pd.read_csv("/path/to/torness_data.txt", sep=" ", header=None)
data4.columns = ["n100", "n100_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev",  "drPrevr"]
df4 = pd.DataFrame(data4)
df4['label'] = 0

data5 = pd.read_csv("/path/to/world_data.txt", sep=" ", header=None)
data5.columns = ["n100", "n100_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT" , "closestPMT_prev",  "drPrevr"]]
df5 = pd.DataFrame(data5)
df5['label'] = 0

data7 = pd.read_csv("/path/to/geoneutrinos_data.txt", sep=" ", header=None)
data7.columns = ["n100", "n100_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev", "drPrevr"]
df7 = pd.DataFrame(data7)
df7['label'] = 0

#li9 model
frames = [df1, df3, df4, df5, df7]
X = df2.append(frames, ignore_index=True)
ydf = li9[['label']]
y = ydf.to_numpy()
y = y.flatten()
X = X.drop(['label'], axis=1)
train_X, test_X, train_y, test_y = train_test_split(X, y, stratify = y)
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                                n_estimators=100, learning_rate=0.1)
clf.fit(train_X, train_y)
pred = clf.predict(test_X)
prob = clf.predict_proba(test_X)
score = clf.decision_function(test_X)
cm = confusion_matrix(test_y, pred, labels=clf.classes_)
print (cm)
ConfusionMatrixDisplay.from_predictions(test_y, pred)
plt.title('Li9 finder confusion matrix')
plt.savefig('/path/to/file.png')
#plt.show()
print(classification_report(test_li9lab, li9_predictions))

test_X.loc[:,'classifier'] = pred
rows = test_X.index
ydf.index = range(len(ydf))
labels = ydf.iloc[rows,:]
test_X.loc[:,'label'] = labels
test_X.loc[:,'scores'] = score
signal = prob[:,1]
fpr, tpr, _ = roc_curve(labels, signal)
auc = auc(fpr, tpr)
plt.plot(fpr, tpr, marker=',', label= 'Lithium-9 (area = {:.2f}'.format(auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Lithium-9 finder ROC curve')
plt.savefig('/path/to/file.png')
#plt.show()

ts = test_X.loc[(test_X.classifier==0) & (test_X.label==0)]
tb = test_X.loc[(test_X.classifier==1) & (test_X.label==1)]
fs = test_X.loc[(test_X.classifier==0) & (test_X.label==1)]
fb = test_X.loc[(test_X.classifier==1) & (test_X.label==0)]
#plot the events by their decision function score
plt.hist(ts.scores.values.flatten(), bins=50, label='True Other', alpha=.5)
plt.hist(tb.scores.values.flatten(), bins=50, label='True Lithium-9', alpha=.5)
plt.hist(fs.scores.values.flatten(), bins=50, label='False Other', alpha=.5)
plt.hist(fb.scores.values.flatten(), bins=50, label='False Lithium-9', alpha=.5)
plt.yscale('log')
plt.title('Lithium-9 Finder confidence scores')
plt.legend(loc='best')
plt.xlabel('Decision scores')
plt.ylabel('Frequency (log)')
plt.savefig("/path/to/file")
#plt.show()

test_X.to_csv('/path/to/file.csv')
filename = '/path/to/file.sav'
joblib.dump(clf, filename)
