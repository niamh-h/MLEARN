import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import plot_roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, auc
import sys
#load the model
clf = joblib.load('/path/to/model.sav')

#load data
data1 = pd.read_csv("/path/to/heysham_2_validation_data.txt", sep=" ", header=None)
data1.columns = ["n100", "n100_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", 'closestPMT_prev', 'drPrevr']
df1 = pd.DataFrame(data1)
df1['label'] = 0
df1['source'] = 1

data2 = pd.read_csv('/path/to/li9_validation_data.txt', sep= " ", header=None)
data2.columns = ['n100', 'n100_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT', 'closestPMT_prev', 'drPrevr']
df2 = pd.DataFrame(data2)
df2['label'] = 0
df2['source'] = 2

data3 = pd.read_csv('/path/to/n17_validation_data.txt', sep= " ", header=None)
data3.columns = ['n100', 'n100_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT', 'closestPMT_prev', 'drPrevr']
df3 = pd.DataFrame(data3)
df3['label'] = 0
df3['source'] = 3

data4 = pd.read_csv('/path/to/world_validation_data.txt', sep= " ", header=None)
data4.columns = ['n100', 'n100_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT', 'closestPMT_prev', 'drPrevr']
df4 = pd.DataFrame(data4)
df4['label'] = 0
df4['source'] = 4

data5 = pd.read_csv("/path/to/torness_validation_data.txt", sep=" ", header=None)
data5.columns = ["n100", "n100_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", 'closestPMT_prev', 'drPrevr']
df5 = pd.DataFrame(data5)
df5['label'] = 0
df5['source'] = 5

data6 = pd.read_csv('/path/to/neutrons_validation_data.txt', sep= " ", header=None)
data6.columns = ['n100', 'n100_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT', 'closestPMT_prev', 'drPrevr']
df6 = pd.DataFrame(data6)
df6['label'] = 1
df6['source'] = 6

data7 = pd.read_csv('/path/to/geoneutrinos_validation_data.txt', sep= " ", header=None)
data7.columns = ['n100', 'n100_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT', 'closestPMT_prev', 'drPrevr']
df7 = pd.DataFrame(data7)
df7['label'] = 0
df7['source'] = 7

#fast neutron validation, for lithium-9 validation replace all fn with li9, comment out df6 and label df2 as 1
frames = [df1, df2, df3, df4, df5, df7]
X = df6.append(frames, ignore_index=True)
y = X[['label']]
y = y.to_numpy()
y = y.flatten()
source = X[['source']]
source = source.to_numpy()
source = source.flatten()
X = X.drop(['label', 'source'], axis=1)
pred = clf.predict(X)
prob = clf.predict_proba(X)
scores = clf.decision_function(X)
confusion_matrix(y, pred)
cm = classification_report(y, pred, labels=clf.classes_))
print(cm)
ConfusionMatrixDisplay.from_predictions(y, pred)
plt.title('Fast Neutron Finder Validation Confusion Matrix')
plt.savefig('/path/to/file.png')
#plt.show()
signal=prob[;,1]
fpr, tpr, _ = roc_curve(y, signal)
auc = auc(fpr, tpr)
plt.plot(fpr, tpr, marker=',', label='Fast Neurons (area = {:.2f})'.format(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Fast Neutron Finder Validation ROC')
plt.savefig('/path/to/file.png')
#plt.show()

X.loc[:,'classifier'] = pred
X.loc[:,'source'] = source
X.loc[:,'label'] = y
X.loc[:,'scores'] = scores
X.loc[:,'prob_fn'] = prob[:,1] #or prob_li9
X.loc[:,'prob_other'] = prob[:,0]
print(X)
X.to_csv('/path/to/file.csv')

