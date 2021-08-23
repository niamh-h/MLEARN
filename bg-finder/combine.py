import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import plot_roc_curve, confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score, auc
import sys
#load the models
li9_clf = joblib.load('/mnt/c/Users/Niamh/Documents/SummerJob/data_v2/16m_wbls/li9finder_16wbls.sav')
fn_clf = joblib.load('/mnt/c/Users/Niamh/Documents/SummerJob/data_v2/16m_wbls/fnfinder_16wbls.sav')
#load data
data2 = pd.read_csv('/mnt/c/Users/Niamh/Documents/SummerJob/data_v2/16m_wbls/clf_2/li9_clf.txt', sep= " ", header=None)
data2.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT', 'closestPMT_prev', 'drPrevr', 'mc_energy']
df2 = pd.DataFrame(data2)
df2['label'] = 0
df2['source'] = 2

data3 = pd.read_csv('/mnt/c/Users/Niamh/Documents/SummerJob/data_v2/16m_wbls/clf_2/n17_clf.txt', sep= " ", header=None)
data3.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT', 'closestPMT_prev', 'drPrevr', 'mc_energy']
df3 = pd.DataFrame(data3)
df3['label'] = 0
df3['source'] = 3

data5 = pd.read_csv('/mnt/c/Users/Niamh/Documents/SummerJob/data_v2/16m_wbls/clf_2/world_clf.txt', sep= " ", header=None)
data5.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT', 'closestPMT_prev', 'drPrevr', 'mc_energy']
df5 = pd.DataFrame(data5)
df5['label'] = 0
df5['source'] = 4

datat = pd.read_csv("/mnt/c/Users/Niamh/Documents/SummerJob/data_v2/16m_wbls/clf_2/tornessfull_clf.txt", sep=" ", header=None)
datat.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", 'closestPMT_prev', 'drPrevr', 'mc_energy']
dft = pd.DataFrame(datat)
dft['label'] = 0
dft['source'] = 5

#datah = pd.read_csv("/mnt/c/Users/Niamh/Documents/SummerJob/data/16m_gdwbls/hf_uncal_clf/hf_uncal_clf/heyshamfull_clfData16.txt", sep=" ", header=None)
#datah.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
#                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_pr>
#dfh = pd.DataFrame(datah)
#dfh['label'] = 0
#dfh['source'] = 1

data6 = pd.read_csv('/mnt/c/Users/Niamh/Documents/SummerJob/data_v2/16m_wbls/clf_2/fn_clf.txt', sep= " ", header=None)
data6.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT', 'closestPMT_prev', 'drPrevr', 'mc_energy']
df6 = pd.DataFrame(data6)
df6['label'] = 1
df6['source'] = 6

data7 = pd.read_csv('/mnt/c/Users/Niamh/Documents/SummerJob/data_v2/16m_wbls/clf_2/geo_clf.txt', sep= " ", header=None)
data7.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_two', 'beta_two_prev', 'beta_three',
                'beta_three_prev', 'beta_four', 'beta_four_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT', 'closestPMT_prev', 'drPrevr', 'mc_energy']
df7 = pd.DataFrame(data7)
df7['label'] = 0
df7['source'] = 7

data8 = pd.read_csv("/mnt/c/Users/Niamh/Documents/SummerJob/data_v2/16m_wbls/clf_2/heysham2_clf.txt", sep=" ", header=None)
data8.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", 'closestPMT_prev', 'drPrevr', 'mc_energy']
df8 = pd.DataFrame(data8)
df8['label'] = 0
df8['source'] = 1

fn_frames = [df2, df3, df5, dft, df7, df8]
fn = df6.append(fn_frames, ignore_index=True)
fn_lab = fn[['label']]
fn_lab = fn_lab.to_numpy()
fn_lab = fn_lab.flatten()
fn_source = fn[['source']]
fn_source = fn_source.to_numpy()
fn_source = fn_source.flatten()
fn_en = fn[['mc_energy']]
fn_en = fn_en.to_numpy()
fn_en = fn_en.flatten()
fn = fn.drop(['label', 'source', 'mc_energy'], axis=1)
fn_pred = fn_clf.predict(fn)
fn_prob = fn_clf.predict_proba(fn)
fn_scores = fn_clf.decision_function(fn)
print(confusion_matrix(fn_lab, fn_pred))
print(classification_report(fn_lab, fn_pred))
disp = plot_confusion_matrix(fn_clf, fn, fn_lab)
disp.figure_.suptitle('Combined: Fast Neutron Finder, Heysham 2 Signal (16m, gdwbls)')
#plt.savefig('/mnt/c/Users/Niamh/Documents/SummerJob/figures_v2/16m_wbls/fncm_clf16wbls_h2.png')
#plt.show()

plot_roc_curve(fn_clf, fn, fn_lab, pos_label=0, marker=',', label='Fast Neutron')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Combined: Fast Neutron finder, Heysham 2 signal (16m, gdwbls)')
#plt.savefig('/mnt/c/Users/Niamh/Documents/SummerJob/figures_v2/16m_wbls/fnroc_clf16wbls_h2.png')
#plt.show()

fn.loc[:,'fn_classifier'] = fn_pred
fn.loc[:,'source'] = fn_source
fn.loc[:,'label'] = fn_lab
fn.loc[:,'fn_scores'] = fn_scores
fn.loc[:,'prob_fn'] = fn_prob[:,1]
fn.loc[:,'prob_otherfn'] = fn_prob[:,0]
fn.loc[:,'mc_energy'] = fn_en
print(fn)
fn.to_csv('/mnt/c/Users/Niamh/Documents/SummerJob/data_v2/16m_wbls/clf_2/fn_clf16wbls_h2.csv')

li9 = fn.loc[fn.fn_classifier==0] #keep only what fn_finder leaves
li9.label.replace(to_replace=1,value=0,inplace=True) #replace labels as 0
li9.loc[li9.source==3, 'label'] = 1 #relabel li9 as 1
print(li9)
print(li9.loc[li9.source==3])

li9_lab = li9['label']
li9_lab = li9_lab.to_numpy()
li9_lab = li9_lab.flatten()
li9_source = li9['source']
li9_source = li9_source.to_numpy()
li9_source = li9_source.flatten()
li9_en = li9['mc_energy']
li9_en = li9_en.to_numpy()
li9_en = li9_en.flatten()
li9 = li9.drop(['label', 'source', 'fn_classifier', 'fn_scores', 'prob_fn', 'prob_otherfn', 'mc_energy'], axis=1)
print(li9)

li9_pred = li9_clf.predict(li9)
li9_prob = li9_clf.predict_proba(li9)
li9_scores = li9_clf.decision_function(li9)
print(confusion_matrix(li9_lab, li9_pred))
print(classification_report(li9_lab, li9_pred))
disp = plot_confusion_matrix(li9_clf, li9, li9_lab)
disp.figure_.suptitle('Combined: Lithium-9 Finder, Heysham 2 Signal (16m, gdwbls)')
#plt.savefig('/mnt/c/Users/Niamh/Documents/SummerJob/figures_v2/16m_wbls/li9cm_clf16wbls_h2.png')
#plt.show()

plot_roc_curve(li9_clf, li9, li9_lab, pos_label=0, marker=',', label='Lithium-9')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Combined: Lithium-9 finder, Heysham 2 signal (16m, gdwbls)')
#plt.savefig('/mnt/c/Users/Niamh/Documents/SummerJob/figures_v2/16m_wbls/li9roc_clf16wbls_h2.png')
#plt.show()

li9.loc[:,'li9_classifier'] = li9_pred
li9.loc[:,'source'] = li9_source
li9.loc[:,'label'] = li9_lab
li9.loc[:,'li9_scores'] = li9_scores
li9.loc[:,'prob_li9'] = li9_prob[:,1]
li9.loc[:,'prob_otherli9'] = li9_prob[:,0]
li9.loc[:,'mc_energy'] = li9_en
print(li9)
li9.to_csv('/mnt/c/Users/Niamh/Documents/SummerJob/data_v2/16m_wbls/clf_2/li9_clf16wbls_h2.csv')

