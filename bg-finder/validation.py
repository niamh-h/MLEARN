import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import plot_roc_curve, confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score, auc
import sys
#load the model
clf = joblib.load('/mnt/c/Users/Niamh/Documents/SummerJob/data/16m_gdwbls/h2_uncal_clf/li9_finder/li9finder_16wbls.sav')

#load data
data2 = pd.read_csv('/mnt/c/Users/Niamh/Documents/SummerJob/data/16m_gdwbls/h2_uncal_clf/h2_uncal_clf/li9_clfData.txt', sep= " ", header=None)
data2.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT', 'closestPMT_prev', 'drPrevr']
df2 = pd.DataFrame(data2)
df2['label'] = 1
df2['source'] = 3 

data3 = pd.read_csv('/mnt/c/Users/Niamh/Documents/SummerJob/data/16m_gdwbls/h2_uncal_clf/h2_uncal_clf/n17_clfData.txt', sep= " ", header=None)
data3.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT', 'closestPMT_prev', 'drPrevr']
df3 = pd.DataFrame(data3)
df3['label'] = 0
df3['source'] = 4 

data5 = pd.read_csv('/mnt/c/Users/Niamh/Documents/SummerJob/data/16m_gdwbls/h2_uncal_clf/h2_uncal_clf/world_clfData.txt', sep= " ", header=None)
data5.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT', 'closestPMT_prev', 'drPrevr']
df5 = pd.DataFrame(data5)
df5['label'] = 0
df5['source'] = 5

datat = pd.read_csv("/mnt/c/Users/Niamh/Documents/SummerJob/data/16m_gdwbls/h2_uncal_clf/h2_uncal_clf/tornessfull_clfData.txt", sep=" ", header=None)
datat.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", 'closestPMT_prev', 'drPrevr']
dft = pd.DataFrame(datat)
dft['label'] = 0
dft['source'] = 5

#datah = pd.read_csv("/mnt/c/Users/Niamh/Documents/SummerJob/data/16m_gdwbls/hf_uncal_clf/hf_uncal_clf/heyshamfull_clfData16.txt", sep=" ", header=None)
#datah.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
#                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
#dfh = pd.DataFrame(datah)
#dfh['label'] = 0
#dfh['source'] = 1

#data6 = pd.read_csv('/mnt/c/Users/Niamh/Documents/SummerJob/data/22m_gdwbls/eval/fn_evalData.txt', sep= " ", header=None)
#data6.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
#                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT', 'closestPMT_prev', 'drPrevr']
#df6 = pd.DataFrame(data6)
#df6['label'] = 1
#df6['source'] = 6

data7 = pd.read_csv('/mnt/c/Users/Niamh/Documents/SummerJob/data/16m_gdwbls/h2_uncal_clf/h2_uncal_clf/geo_clfData.txt', sep= " ", header=None)
data7.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT', 'closestPMT_prev', 'drPrevr']
df7 = pd.DataFrame(data7)
df7['label'] = 0
df7['source'] = 7

data8 = pd.read_csv("/mnt/c/Users/Niamh/Documents/SummerJob/data/16m_gdwbls/h2_uncal_clf/h2_uncal_clf/heysham2_clfData.txt", sep=" ", header=None)
data8.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", 'closestPMT_prev', 'drPrevr']
df8 = pd.DataFrame(data8)
df8['label'] = 0
df8['source'] = 1

frames = [df3, df5, dft, df7, df8]
X = df2.append(frames, ignore_index=True)
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
print(confusion_matrix(y, pred))
print(classification_report(y, pred))
disp = plot_confusion_matrix(clf, X, y)
disp.figure_.suptitle('Lithium-9 Finder, Heysham 2 Signal (final validation, 16m_gdwbls)')
plt.savefig('/mnt/c/Users/Niamh/Documents/SummerJob/figures/16m_gdwbls/h2_uncal/li9_finder/cm_clf16wbls.png')
plt.show()

plot_roc_curve(clf, X, y, pos_label=0, marker=',', label='Lithium-9')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Lithium-9 finder, Heysham 2 signal (final validation, 16_gdwbls)')
plt.savefig('/mnt/c/Users/Niamh/Documents/SummerJob/figures/16m_gdwbls/h2_uncal/li9_finder/roc_clf16wbls.png')
plt.show()

X.loc[:,'classifier'] = pred
X.loc[:,'source'] = source
X.loc[:,'label'] = y
X.loc[:,'scores'] = scores
X.loc[:,'prob_fn'] = prob[:,1]
X.loc[:,'prob_other'] = prob[:,0]
print(X)
X.to_csv('/mnt/c/Users/Niamh/Documents/SummerJob/data/16m_gdwbls/h2_uncal_clf/li9_finder/16wbls_clfclfdata.csv')

