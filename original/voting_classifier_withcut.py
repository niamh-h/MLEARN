import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score, auc
import joblib
import sys
from sklearn.ensemble import VotingClassifier

#import the models
li9_bin = joblib.load('li9_classifier.sav')
world_bin = joblib.load('world_classifier.sav')
n17_bin = joblib.load('n17_classifier.sav')
neu_bin = joblib.load('neutron_classifier.sav')
sin_bin = joblib.load('trained_AdaBoost_sim1_nt_variables.sav')

#load the validation data
data1 = pd.read_csv("tarasignalVALData.txt", sep=" ", header=None)
data1.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
                "beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df1 = pd.DataFrame(data1)

data2 = pd.read_csv("tarali9VALData.txt", sep=" ", header=None)
data2.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
                "beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df2 = pd.DataFrame(data2)

data3 = pd.read_csv("taran17VALData.txt", sep=" ", header=None)
data3.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
                "beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df3 = pd.DataFrame(data3)

data4 = pd.read_csv("tarasmallVALData.txt", sep=" ", header=None)
data4.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
                "beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df4 = pd.DataFrame(data4)

data5 = pd.read_csv("taraworldVALData.txt", sep=" ", header=None)
data5.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
                "beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df5 = pd.DataFrame(data5)

data6 = pd.read_csv("taraneutronsVALData.txt", sep=" ", header=None)
data6.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
                "beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df6 = pd.DataFrame(data6)

data7 = pd.read_csv("sim2_singles_nt_variables.txt", sep=" ", header=None) #taras training data
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
df7['source'] = 7 #7=uncorrelated 

frames = [df2, df3, df4, df5, df6, df7]
data = df1.append(frames, ignore_index=True)
print(data)

#create the target values
label = data[['label']]
label = label.to_numpy()
label = label.flatten()
source = data[['source']]
source = source.to_numpy()
source = source.flatten()
data = data.drop(['label', 'source'], axis=1)
print(data)
print(label)
print(source)

#feed data through tara's model first
sin_pred = sin_bin.predict(data)
sin_prob = sin_bin.predict_proba(data)
sin_score = sin_bin.decision_function(data)

data.loc[:,'label'] = label
data.loc[:,'source'] = source
data.loc[:,'classifier'] = sin_pred
data.loc[:,'score'] = sin_score

print(data)
ts = data.loc[(data.classifier==1) & (data.label==1)]
tb = data.loc[(data.classifier==0) & (data.label==0)]
fs = data.loc[(data.classifier==1) & (data.label==0)]
fb = data.loc[(data.classifier==0) & (data.label==1)]

plt.hist(ts.score.values.flatten(), bins=100,
                label='True Signal', alpha=.5)
plt.hist(tb.score.values.flatten(), bins=100,
                label='True Background', alpha=.5)
plt.hist(fs.score.values.flatten(), bins=100,
                label='False Signal', alpha=.5)
plt.hist(fb.score.values.flatten(), bins=100,
                label='False Background', alpha=.5)
plt.yscale('log')
plt.title('Singles model on All sources')
plt.legend(loc='best')
plt.xlabel('Decision scores')
plt.ylabel('Frequency (log)')
plt.show()

data = data.drop(data[data.score < 0.1].index)
data.index = range(len(data))

label = data[['label']]
label = label.to_numpy()
label = label.flatten()
source = data[['source']]
source = source.to_numpy()
source = source.flatten()
data = data.drop(['label', 'source', 'classifier', 'score'], axis=1)

#combining with voting classifier
vot_clf = joblib.load('voting_classifier.sav')
pred = vot_clf.predict(data)
prob = vot_clf.predict_proba(data)

print('combined run\n', confusion_matrix(label, pred))
print(classification_report(label, pred))
disp = plot_confusion_matrix(vot_clf, data, label)
disp.figure_.suptitle("All Backgrounds Classifier (validation)")
plt.show()
prob_signal = prob[:,1]
prob_bg = prob[:,0]
falsepr, true_pr, thresh = roc_curve(label, prob_signal, pos_label=1)
plt.plot(falsepr, true_pr, marker='.', label='VotingClassifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('All Backgrounds Classifier (validation)')
plt.show()

data.loc[:,'classifier'] = pred
data.loc[:,'label'] = label
data.loc[:,'source'] = source
data.loc[:,'prob_signal'] = prob_signal
data.loc[:,'prob_bg'] = prob_bg
print(data)
data.to_csv('taravoting_classifiedVALdata_withcut.csv')

