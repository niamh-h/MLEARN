import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import plot_roc_curve, confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score, auc
import sys
#load the model
clf = joblib.load('/mnt/c/Users/Niamh/Documents/SummerJob/data/16m_gdwbls/hf_uncal_clf/fnfinder_16wbls.sav')

#load validation data
data2 = pd.read_csv('/path/to/li9/txt/file', sep= " ", header=None)
data2.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT']
df2 = pd.DataFrame(data2)
df2['label'] = 0
df2['source'] = 3 #label the specific source of each data point

data3 = pd.read_csv('/path/to/n17/txt/file', sep= " ", header=None)
data3.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT']
df3 = pd.DataFrame(data3)
df3['label'] = 0
df3['source'] = 4 

data5 = pd.read_csv('/path/to/world/txt/file', sep= " ", header=None)
data5.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT']
df5 = pd.DataFrame(data5)
df5['label'] = 0
df5['source'] = 5

datat = pd.read_csv("/path/to/tornessfull/txt/file", sep=" ", header=None)
datat.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
dft = pd.DataFrame(datat)
dft['label'] = 0
dft['source'] = 5

datah = pd.read_csv("/path/to/heyshamfull/txt/file", sep=" ", header=None)
datah.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
dfh = pd.DataFrame(datah)
dfh['label'] = 0
dfh['source'] = 1 #1 always = signal, so this will change

data6 = pd.read_csv('/path/to/fn/txt/file', sep= " ", header=None)
data6.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT']
df6 = pd.DataFrame(data6)
df6['label'] = 1
df6['source'] = 6

data7 = pd.read_csv('/path/to/geo/txt/file', sep= " ", header=None)
data7.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT']
df7 = pd.DataFrame(data7)
df7['label'] = 0
df7['source'] = 7

data8 = pd.read_csv("/path/to/heysham2/txt/file", sep=" ", header=None)
data8.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df8 = pd.DataFrame(data8)
df8['label'] = 0
df8['source'] = 1 #either heysham full OR heysham 2 should be used at once

frames = [df2, df3, df5, dft, df7, dfh]
X = df6.append(frames, ignore_index=True)
y = X[['label']]
y = y.to_numpy()
y = y.flatten()
source = X[['source']]
source = source.to_numpy()
source = source.flatten()
X = X.drop(['label', 'source'], axis=1)
#apply the model
pred = clf.predict(X)
prob = clf.predict_proba(X)
scores = clf.decision_function(X)
#analyse the model
print(confusion_matrix(y, pred))
print(classification_report(y, pred))
disp = plot_confusion_matrix(clf, X, y)
disp.figure_.suptitle('Confusion Matrix title')
plt.savefig('/path/to/file')
plt.show()

plot_roc_curve(clf, X, y, pos_label=0, marker=',', label='Fast Neutron')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('ROC Curve title')
plt.savefig('/path/to/file')
plt.show()

X.loc[:,'classifier'] = pred
X.loc[:,'source'] = source
X.loc[:,'label'] = y
X.loc[:,'scores'] = scores
X.loc[:,'prob_fn'] = prob[:,1]
X.loc[:,'prob_other'] = prob[:,0]

X.to_csv('/path/to/file')

