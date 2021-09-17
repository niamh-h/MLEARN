import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_confusion_matrix, roc_curve, roc_auc_score, confusion_matrix, auc
from matplotlib import pyplot as plt
import joblib
import sys
#organise the data from the text files into a dataset
data2 = pd.read_csv("/path/to/li9_data.txt", sep=" ", header=None)
data2.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev", "drPrevr"]
df2 = pd.DataFrame(data2)
df2['label'] = 0

data3 = pd.read_csv("/path/to/n17_data.txt", sep=" ", header=None)
data3.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev",  "beta_three",
                "beta_three_prev","beta_four", "beta_four_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev", "drPrevr"]
df3 = pd.DataFrame(data3)
df3['label'] = 0

data5 = pd.read_csv("/path/to/world_data.txt", sep=" ", header=None)
data5.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev", "drPrevr"]
df5 = pd.DataFrame(data5)
df5['label'] = 0
datat = pd.read_csv("path/to/torness_data.txt", sep=" ", header=None)
datat.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev", "drPrevr"]
dft = pd.DataFrame(datat)
dft['label'] = 0

#datah = pd.read_csv("/path/to/heyshamfull_data.txt", sep=" ", header=None)
#datah.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
#                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev", "drPrevr"]
#dfh = pd.DataFrame(datah)
#dfh['label'] = 0

data6 = pd.read_csv("/path/to/neutrons_data.txt", sep=" ", header=None)
data6.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev", "drPrevr"]
df6 = pd.DataFrame(data6)
df6['label'] = 1

data7 = pd.read_csv("/path/to/geoneutrinos_data.txt", sep=" ", header=None)
data7.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev", "drPrevr"]
df7 = pd.DataFrame(data7)
df7['label'] = 0

data8 = pd.read_csv("/path/to/heysham2_data.txt", sep=" ", header=None)
data8.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev", "drPrevr"]
df8 = pd.DataFrame(data8)
df8['label'] = 0

#neutron model
frames = [df3, df2, df5, dft, df7, df8]
neu = df6.append(frames, ignore_index=True)
neu_labelcolumn = neu[['label']]
neu_label = neu_labelcolumn.to_numpy()
neu_label = neu_label.flatten()
neu = neu.drop(['label'], axis=1)
train_neu, test_neu, train_neulab, test_neulab = train_test_split(neu, neu_label, random_state=None, stratify=neu_label)
neu_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                                n_estimators=100, learning_rate=0.1)
neu_clf.fit(train_neu, train_neulab)
neu_predictions = neu_clf.predict(test_neu)
neu_prob = neu_clf.predict_proba(test_neu)
neu_score = neu_clf.decision_function(test_neu)
print(confusion_matrix(test_neulab, neu_predictions))
disp = plot_confusion_matrix(neu_clf, test_neu, test_neulab)
disp.figure_.suptitle("Fast Neutron finder, Heysham 2 signal (training, 22m_gdwbls)")
plt.savefig('/path/to/file.png')
plt.show()
print(classification_report(test_neulab, neu_predictions))
test_neu.loc[:,'classifier'] = neu_predictions
neu_rows = test_neu.index
neu_labelcolumn.index = range(len(neu_labelcolumn))
neu_test_labels = neu_labelcolumn.iloc[neu_rows,:]
test_neu.loc[:,'label'] = neu_test_labels
test_neu.loc[:,'scores'] = neu_score
neu_signal=neu_prob[:,1]
neu_fpr, neu_tpr, ___ = roc_curve(neu_test_labels, neu_signal, pos_label=1)
neu_auc = auc(neu_fpr, neu_tpr)
plt.plot(neu_fpr, neu_tpr, marker=',', label='Fast Neutrons (area = {:.2f})'.format(neu_auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Fast Neutron finder, Heysham 2 Signal (training, 22m_gdwbls)')
plt.savefig('/path/to/file.png')
plt.show()

test_neu.to_csv('/path/to/file.csv')
filename = '/path/to/file.sav'
joblib.dump(neu_clf, filename)
