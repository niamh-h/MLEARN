import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix, roc_curve, roc_auc_score, confusion_matrix, auc
import matplotlib.pyplot as plt
import joblib
import sys

data2 = pd.read_csv("/path/to/li9_data.txt", sep=" ", header=None)
data2.columns = ["n100", "n100_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev", "drPrevr"]
df2 = pd.DataFrame(data2)
df2['label'] = 1

data3 = pd.read_csv("/path/to/n17_data.txt", sep=" ", header=None)
data3.columns = ["n100", "n100_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev",  "beta_three",
                "beta_three_prev","beta_four", "beta_four_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev", "drPrevr"]
df3 = pd.DataFrame(data3)
df3['label'] = 0

data5 = pd.read_csv("/path/to/world_data.txt", sep=" ", header=None)
data5.columns = ["n100", "n100_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev",  "drPrevr"]
df5 = pd.DataFrame(data5)
df5['label'] = 0
datat = pd.read_csv("/path/to/torness_data.txt", sep=" ", header=None)
datat.columns = ["n100", "n100_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev",  "drPrevr"]
dft = pd.DataFrame(datat)
dft['label'] = 0
#datah = pd.read_csv("//path/to/heyshamfull_data.txt", sep=" ", header=None)
#datah.columns = ["n100", "n100_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
#                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT" , "closestPMT_prev",  "drPrevr"]]
#dfh = pd.DataFrame(datah)
#dfh['label'] = 0

data7 = pd.read_csv("/path/to/geoneutrinos_data.txt", sep=" ", header=None)
data7.columns = ["n100", "n100_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev", "drPrevr"]
df7 = pd.DataFrame(data7)
df7['label'] = 0

data8 = pd.read_csv("/path/to/heysham2_data.txt", sep=" ", header=None)
data8.columns = ["n100", "n100_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT", "closestPMT_prev", "drPrevr"]
df8 = pd.DataFrame(data8)
df8['label'] = 0

#li9 model
frames = [df3, df5, dft, df7, df8]
li9 = df2.append(frames, ignore_index=True)
print(li9)
li9_labelcolumn = li9[['label']]
li9_label = li9_labelcolumn.to_numpy()
li9_label = li9_label.flatten()
li9 = li9.drop(['label'], axis=1)
train_li9, test_li9, train_li9lab, test_li9lab = train_test_split(li9, li9_label, random_state=None, stratify=li9_label)
li9_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                                n_estimators=100, learning_rate=0.1)
li9_clf.fit(train_li9, train_li9lab)
li9_predictions = li9_clf.predict(test_li9)
li9_prob = li9_clf.predict_proba(test_li9)
li9_score = li9_clf.decision_function(test_li9)
print(confusion_matrix(test_li9lab, li9_predictions))
disp = plot_confusion_matrix(li9_clf, test_li9, test_li9lab)
disp.figure_.suptitle("Lithium-9 Finder (training, 16m_gdwbls)")
plt.savefig('/path/to/file.png')
plt.show()
print(classification_report(test_li9lab, li9_predictions))
test_li9.loc[:,'classifier'] = li9_predictions
li9_rows = test_li9.index
li9_labelcolumn.index = range(len(li9_labelcolumn))
li9_test_labels = li9_labelcolumn.iloc[li9_rows,:]
test_li9.loc[:,'label'] = li9_test_labels
test_li9.loc[:,'scores'] = li9_score
li9_signal = li9_prob[:,1]
li9_fpr, li9_tpr, _ = roc_curve(li9_test_labels, li9_signal, pos_label=1)
li9_auc = auc(li9_fpr, li9_tpr)
plt.plot(li9_fpr, li9_tpr, marker=',', label= 'Lithium-9 (area = {:.2f}'.format(li9_auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Lithium-9 finder, Heysham 2 Sginal (training, 16m_gdwbls)')
plt.savefig('/path/to/file.png')
plt.show()

test_li9.to_csv('/path/to/file.csv')
filename = '/path/to/file.sav'
joblib.dump(li9_clf, filename)
