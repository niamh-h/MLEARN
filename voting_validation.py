import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score, auc
import joblib
import sys
from sklearn.ensemble import VotingClassifier

#import the model
vot_clf = joblib.load('voting_classifier.sav')
#load the validation data
data1 = pd.read_csv("bigvalfilename.txt", sep=" ", header=None)
data1.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
                "beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df1 = pd.DataFrame(data1)

data2 = pd.read_csv("li9valfilename.txt", sep=" ", header=None)
data2.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
                "beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df2 = pd.DataFrame(data2)

data3 = pd.read_csv("n17valfilename.txt", sep=" ", header=None)
data3.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
                "beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df3 = pd.DataFrame(data3)

data4 = pd.read_csv("smallvalfilename.txt", sep=" ", header=None)
data4.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
                "beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df4 = pd.DataFrame(data4)

data5 = pd.read_csv("worldvalfilename.txt", sep=" ", header=None)
data5.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
                "beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df5 = pd.DataFrame(data5)

data6 = pd.read_csv("neuvalfilename.txt", sep=" ", header=None)
data6.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
                "beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df6 = pd.DataFrame(data6)

data7 = pd.read_csv("geovalfilename", sep=" ", header=None)
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

df1['source'] = 1 #1=big
df4['source'] = 2 #2=small
df2['source'] = 3 #3=li9
df3['source'] = 4 #4=n17
df5['source'] = 5 #5=world
df6['source'] = 6 #6=neutrons
df7['source'] = 7 #7=geoneutrinos

frames = [df2, df3, df4, df5, df6, df7]
data = df1.append(frames, ignore_index=True)

#create the target values
label = data[['label']]
label = label.to_numpy()
label = label.flatten()
source = data[['source']]
source = source.to_numpy()
source = source.flatten()
data = data.drop(['label', 'source'], axis=1)
#feed into voting classifier
pred = vot_clf.predict(data)
prob = vot_clf.predict_proba(data)

print('combined run\n', confusion_matrix(label, pred))
print(classification_report(label, pred))
disp = plot_confusion_matrix(vot_clf, data, label)
disp.figure_.suptitle("Voting Classifier (validation)")
plt.show()
prob_signal = prob[:,1]
prob_bg = prob[:,0]
falsepr, true_pr, thresh = roc_curve(label, prob_signal, pos_label=1)
plt.plot(falsepr, true_pr, marker='.', label='VotingClassifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('Correlated Backgrounds Classifier (validation)')
plt.show()

data.loc[:,'classifier'] = pred
data.loc[:,'label'] = label
data.loc[:,'source'] = source
data.loc[:,'prob_signal'] = prob_signal
data.loc[:,'prob_bg'] = prob_bg
print(data)
data.to_csv('votingclassifiedVALdata.csv')

