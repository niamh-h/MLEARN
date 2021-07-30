import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score, auc
import sys
from sklearn.inspection import permutation_importance
#load the models
li9_clf = joblib.load('li9_classifier.sav')
world_clf = joblib.load('world_classifier.sav')
n17_clf = joblib.load('n17_classifier.sav')
neu_clf = joblib.load('neutron_classifier.sav')
geo_clf = joblib.load('geo_classifier.sav')
#load the validation data
data1 = pd.read_csv('bigfilename.txt', sep= " ", header=None)
data1.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_three',
		'beta_three_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT']
df1 = pd.DataFrame(data1)
df1['label'] = 1

data2 = pd.read_csv('li9filename.txt', sep= " ", header=None)
data2.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_three',
		'beta_three_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT']
df2 = pd.DataFrame(data2)
df2['label'] = 0

data3 = pd.read_csv('n17filename.txt', sep= " ", header=None)
data3.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_three',
		'beta_three_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT']
df3 = pd.DataFrame(data3)
df3['label'] = 0

data4 = pd.read_csv('smallfilename.txt', sep= " ", header=None)
data4.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_three',
		'beta_three_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT']
df4 = pd.DataFrame(data4)
df4['label'] = 1

data5 = pd.read_csv('worldfilename.txt', sep= " ", header=None)
data5.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_three',
		'beta_three_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT']
df5 = pd.DataFrame(data5)
df5['label'] = 0

data6 = pd.read_csv('neutronsfilename.txt', sep= " ", header=None)
data6.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_three',
		'beta_three_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT']
df6 = pd.DataFrame(data6)
df6['label'] = 0

data7 = pd.read_csv('geoneutrinosfilename.txt', sep= " ", header=None)
data7.columns = ['n9', 'n9_prev', 'dt_prev_us', 'inner_hit', 'inner_hit_prev', 'beta_one', 'beta_one_prev', 'beta_three',
                'beta_three_prev', 'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'good_pos', 'good_pos_prev', 'closestPMT']
df7 = pd.DataFrame(data7)
df7['label'] = 0
#run a validation test for each model
#lithium9 test
framesli9 = [df1, df4, df2]
li9 = pd.concat(framesli9)
li9.index = range(len(li9))
li9_label = li9[['label']]
li9_label = li9_label.to_numpy()
li9_label = li9_label.flatten()
li9 = li9.drop(['label'], axis=1)
li9_predictions = li9_clf.predict(li9)
li9_prob = li9_clf.predict_proba(li9)
li9_score = li9_clf.decision_function(li9)
li9_perm = permutation_importance(li9_clf, li9, li9_label, n_jobs=1, random_state=0) 
print('Lithium permutation importances: ', li9_perm.importances_mean, ' +/-', li9_perm.importances_std)
print(confusion_matrix(li9_label,li9_predictions))
print(classification_report(li9_label, li9_predictions))
disp = plot_confusion_matrix(li9_clf, li9, li9_label)
disp.figure_.suptitle("Lithium-9 Background Classifier (validation)")
plt.show()
#nitrogen17 test 
framesn17 = [df1, df4, df3]
n17 = pd.concat(framesn17)
n17.index = range(len(n17))
n17_label = n17[['label']]
n17_label = n17_label.to_numpy()
n17_label = n17_label.flatten()
n17 = n17.drop(['label'], axis=1)
n17_predictions = n17_clf.predict(n17)
n17_prob = n17_clf.predict_proba(n17)
n17_score = n17_clf.decision_function(n17)
n17_perm = permutation_importance(n17_clf, n17, n17_label, n_jobs=1, random_state=0)
print('Nitrogen permutation importances: ', n17_perm.importances_mean, ' +/-', n17_perm.importances_std)
print(confusion_matrix(n17_label, n17_predictions))
print(classification_report(n17_label, n17_predictions))
disp = plot_confusion_matrix(n17_clf, n17, n17_label)
disp.figure_.suptitle("Nitrogen-17 Background Classsifier (validation)")
plt.show()
#neutron test
framesneu = [df1, df4 , df6]
neu = pd.concat(framesneu)
neu.index = range(len(neu))
neu_label = neu[['label']]
neu_label = neu_label.to_numpy()
neu_label = neu_label.flatten()
neu = neu.drop(['label'], axis=1)
neu_predictions = neu_clf.predict(neu)
neu_prob = neu_clf.predict_proba(neu)
neu_score = neu_clf.decision_function(neu)
neu_perm = permutation_importance(neu_clf, neu, neu_label, n_jobs=1, random_state=0)
print('Fast Neutrons permutation importances: ', neu_perm.importances_mean, ' +/-', neu_perm.importances_std)
print(confusion_matrix(neu_label, neu_predictions))
print(classification_report(neu_label, neu_predictions))
disp = plot_confusion_matrix(neu_clf, neu, neu_label)
disp.figure_.suptitle("Fast Neutrons Background Classifier (validation)")
plt.show()
#geoneutrinos reactor test
framesgeo = [df1, df4]
geo = pd.concat(framesgeo)
geo.index = range(len(geo))
geo_label = geo[['label']]
geo_label = geo_label.to_numpy()
geo_label = geo_label.flatten()
geo = small.drop(['label'], axis=1)
geo_predictions = geo_clf.predict(geo)
geo_prob = geo_clf.predict_proba(geo)
geo_score = geo_clf.decision_function(geo)
geo_perm = permutation_importance(geo_clf, geo, geo_label, n_jobs=1, random_state=0)
print('Geoneutrinos permutation importances: ', geo_perm.importances_mean, ' +/-',geo_perm.importances_std)
print(confusion_matrix(geo_label, geo_predictions))
disp = plot_confusion_matrix(geo_clf, geo, geo_label)
disp.figure_.suptitle("Confusion Matrix, Geoneutrinos Background")
plt.show()
#world test
framesworld = [df1, df4, df5]
world = pd.concat(framesworld)
world.index = range(len(world))
world_label = world[['label']]
world_label = world_label.to_numpy()
world_label = world_label.flatten()
world = world.drop(['label'], axis=1)
world_predictions = world_clf.predict(world)
world_prob = world_clf.predict_proba(world)
world_score = world_clf.decision_function(world)
world_perm = permutation_importance(world_clf, world, world_label, n_jobs=1, random_state=0)
print('World permutation importances: ', world_perm.importances_mean, ' +/-', world_perm.importances_std)
print(confusion_matrix(world_label, world_predictions))
print(classification_report(world_label, world_predictions))
disp = plot_confusion_matrix(world_clf, world, world_label)
disp.figure_.suptitle("Boulby World Background Classifier (validation)")
plt.show()
sys.exit()
#analyse the success of each binary classifier
li9_signal = li9_prob[:,1]
n17_signal = n17_prob[:,1]
neu_signal = neu_prob[:,1]
geo_signal = geo_prob[:,1]
world_signal = world_prob[:,1]

li9_fpr, li9_tpr, _ = roc_curve(li9_label, li9_signal, pos_label=1)
li9_auc = auc(li9_fpr, li9_tpr)
n17_fpr, n17_tpr, __ = roc_curve(n17_label, n17_signal, pos_label=1)
n17_auc = auc(n17_fpr, n17_tpr)
neu_fpr, neu_tpr, ___ = roc_curve(neu_label, neu_signal, pos_label=1)
neu_auc = auc(neu_fpr, neu_tpr)
geo_fpr, geo_tpr, ____ = roc_curve(geo_label, geo_signal, pos_label=1)
geo_auc = auc(geo_fpr, geo_tpr)
world_fpr, world_tpr, _____ = roc_curve(world_label, world_signal, pos_label=1)
world_auc = auc(world_fpr, world_tpr)

plt.plot(li9_fpr, li9_tpr, marker=',', label='Lithium-9 (area = {:.2f})'.format(li9_auc))
plt.plot(n17_fpr, n17_tpr, marker=',', label='Nitrogen-17 (area = {:.2f})'.format(n17_auc))
plt.plot(neu_fpr, neu_tpr, marker=',', label='Fast Neutrons (area = {:.2f})'.format(neu_auc))
plt.plot(geo_fpr, geo_tpr, marker=',', label='Geoneutrinos (area = {:.2f})'.format(geo_auc))
plt.plot(world_fpr, world_tpr, marker=',', label='World Background (area = {:.2f})'.format(world_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('ROC, Correlated Backgrounds Binary Classification (validation)')
plt.show()
#add the label/score/prediction to each dataframe and save to separate csv files
li9.loc[:,'classifier'] = li9_predictions
li9.loc[:,'label'] = li9_label
li9.loc[:,'scores'] = li9_score
li9.to_csv('li9_classifiedvaldata.csv')
n17.loc[:,'classifier'] = n17_predictions
n17.loc[:,'label'] = n17_label
n17.loc[:,'scores'] = n17_score
n17.to_csv('n17_classifiedvaldata.csv')
neu.loc[:,'classifier'] = neu_predictions
neu.loc[:,'label'] = neu_label
neu.loc[:,'scores'] = neu_score
neu.to_csv('neu_classifiedvaldata.csv')
geo.loc[:,'classifier'] = geo_predictions
geo.loc[:,'label'] = geo_label
geo.loc[:,'scores'] = geo_score
geo.to_csv('geo_classifiedvaldata.csv')
world.loc[:,'classifier'] = world_predictions
world.loc[:,'label'] = world_label
world.loc[:,'scores'] = world_score
world.to_csv('world_classifiedvaldata.csv')
