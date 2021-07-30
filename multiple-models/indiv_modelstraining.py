import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_confusion_matrix, roc_curve, roc_auc_score, confusion_matrix, auc
from matplotlib import pyplot as plt
import joblib
#organise the data from the text files into a dataset
data1 = pd.read_csv("big_firstclfdata.txt", sep=" ", header=None)
data1.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "good_pos", "good_pos_prev", "closestPMT"]#"beta_one", "beta_one_prev", "beta_three",
		#"beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df1 = pd.DataFrame(data1)
df1['label'] = 1 #signal=1, bg=0

data2 = pd.read_csv("li9_firstclfdata.txt", sep=" ", header=None)
data2.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "good_pos", "good_pos_prev", "closestPMT"]#"beta_one", "beta_one_prev", "beta_three",
#		"beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df2 = pd.DataFrame(data2)
df2['label'] = 0

data3 = pd.read_csv("n17_firstclfdata.txt", sep=" ", header=None)
data3.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "good_pos", "good_pos_prev", "closestPMT"]#"beta_one", "beta_one_prev", "beta_three",
#		"beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df3 = pd.DataFrame(data3)
df3['label'] = 0 

data4 = pd.read_csv("small_firstclfdata.txt", sep=" ", header=None)
data4.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "good_pos", "good_pos_prev", "closestPMT"]#"beta_one", "beta_one_prev", "beta_three",
#		"beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df4 = pd.DataFrame(data4)
df4['label'] = 1

data5 = pd.read_csv("world_firstclfdata.txt", sep=" ", header=None)
data5.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "good_pos", "good_pos_prev", "closestPMT"]#"beta_one", "beta_one_prev", "beta_three",
#		"beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df5 = pd.DataFrame(data5)
df5['label'] = 0

#data6 = pd.read_csv("neu_firstclfdata.txt", sep=" ", header=None)
#data6.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_three",
#		"beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
#df6 = pd.DataFrame(data6)
#df6['label'] = 0

data7 = pd.read_csv("geo_firstclfdata.txt", sep=" ", header=None)
data7.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "good_pos", "good_pos_prev", "closestPMT"]#"beta_one", "beta_one_prev", "beta_three",
              #  "beta_three_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df7 = pd.DataFrame(data7)
df7['label'] = 0

signal = df1.append(df4, ignore_index=True)

#lithium model
li9 = signal.append(df2, ignore_index=True)
li9_labelcolumn = li9[['label']]
li9_label = li9_labelcolumn.to_numpy()
li9_label = li9_label.flatten()
li9 = li9.drop(['label'], axis=1)

#split the data into training and testing
train_li9, test_li9, train_li9lab, test_li9lab = train_test_split(li9, li9_label, random_state=None)

#construct and fit the adaboost classifier to the training set 
li9_clf  = AdaBoostClassifier(
			DecisionTreeClassifier(max_depth=2), 
			n_estimators=100, learning_rate=0.1
)
li9_clf.fit(train_li9, train_li9lab)

#predict the label and probabilities based on the fit
li9_predictions = li9_clf.predict(test_li9)
li9_prob = li9_clf.predict_proba(test_li9)
li9_score = li9_clf.decision_function(test_li9)
#evaluate success with a confusion matrix
print(confusion_matrix(test_li9lab, li9_predictions))
disp = plot_confusion_matrix(li9_clf, test_li9, test_li9lab)
disp.figure_.suptitle("Lithium-9 Background Classifier (training)")
plt.show()

#create a classification report
print(classification_report(test_li9lab, li9_predictions))
#print(clf.score(train_li9,train_li9lab,sample_weight=None))

#add labelled data to dataframe
test_li9.loc[:,'classifier'] = li9_predictions

#get the rows of the test data
li9_rows = test_li9.index

#order the label column
li9_labelcolumn.index = range(len(li9_labelcolumn))

#get the labels for the test data
li9_test_labels = li9_labelcolumn.iloc[li9_rows,:]

#add labels to dataframe
test_li9.loc[:,'label'] = li9_test_labels

#add scores to dataframe and output to csv
test_li9.loc[:,'scores'] = li9_score
test_li9.to_csv('li9_firstclassifiedtestdata.csv')

#save the model
filename = 'li9_firstclassifier.sav'
joblib.dump(li9_clf, filename)

#repeat for bg sources
#nitrogen
n17 = signal.append(df3, ignore_index=True)
n17_labelcolumn = n17[['label']]
n17_label = n17_labelcolumn.to_numpy()
n17_label = n17_label.flatten()
n17 = n17.drop(['label'], axis=1)
train_n17, test_n17, train_n17lab, test_n17lab = train_test_split(n17, n17_label, random_state=None)
n17_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
				n_estimators=100, learning_rate=0.1)
n17_clf.fit(train_n17, train_n17lab)
n17_predictions = n17_clf.predict(test_n17)
n17_prob = n17_clf.predict_proba(test_n17)
n17_score = n17_clf.decision_function(test_n17)
print(confusion_matrix(test_n17lab, n17_predictions))
disp = plot_confusion_matrix(n17_clf, test_n17, test_n17lab)
disp.figure_.suptitle("Nitrogen-17 Background Classifier (training)")
plt.show()
print(classification_report(test_n17lab, n17_predictions))
test_n17.loc[:,'classifier'] = n17_predictions
n17_rows = test_n17.index
n17_labelcolumn.index = range(len(n17_labelcolumn))
n17_test_labels = n17_labelcolumn.iloc[n17_rows,:]
test_n17.loc[:,'label'] = n17_test_labels
test_n17.loc[:,'scores'] = n17_score
test_n17.to_csv('n17_firstclassifiedtestdata.csv')
filename = 'n17_firstclassifier.sav'
joblib.dump(n17_clf, filename)

#neutrons
#neu = signal.append(df6, ignore_index=True)
#neu_labelcolumn = neu[['label']]
#neu_label = neu_labelcolumn.to_numpy()
#neu_label = neu_label.flatten()
#neu = neu.drop(['label'], axis=1)
#train_neu, test_neu, train_neulab, test_neulab = train_test_split(neu, neu_label, random_state=None)
#neu_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
 #                               n_estimators=100, learning_rate=0.1)
#neu_clf.fit(train_neu, train_neulab)
#neu_predictions = neu_clf.predict(test_neu)
#neu_prob = neu_clf.predict_proba(test_neu)
#neu_score = neu_clf.decision_function(test_neu)
#print(confusion_matrix(test_neulab, neu_predictions))
#disp = plot_confusion_matrix(neu_clf, test_neu, test_neulab)
#disp.figure_.suptitle("Fast Neutrons Background Classifier (training)")
#plt.show()
#print(classification_report(test_neulab, neu_predictions))
#test_neu.loc[:,'classifier'] = neu_predictions
#neu_rows = test_neu.index
#neu_labelcolumn.index = range(len(neu_labelcolumn))
#neu_test_labels = neu_labelcolumn.iloc[neu_rows,:]
#test_neu.loc[:,'label'] = neu_test_labels
#test_neu.loc[:,'scores'] = neu_score
#test_neu.to_csv('neu_classifiedtestdata.csv')
#filename = 'neu_classifier.sav'
#joblib.dump(neu_clf, filename)

#world 
world = signal.append(df5, ignore_index=True)
world_labelcolumn = world[['label']]
world_label = world_labelcolumn.to_numpy()
world_label = world_label.flatten()
world = world.drop(['label'], axis=1)
train_world, test_world, train_worldlab, test_worldlab = train_test_split(world, world_label, random_state=None)
world_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                                n_estimators=100, learning_rate=0.1)
world_clf.fit(train_world, train_worldlab)
world_predictions = world_clf.predict(test_world)
world_prob = world_clf.predict_proba(test_world)
world_score = world_clf.decision_function(test_world)
print(confusion_matrix(test_worldlab, world_predictions))
disp = plot_confusion_matrix(world_clf, test_world, test_worldlab)
disp.figure_.suptitle("Worldwide Reactor Background Classifier (training)")
plt.show()
print(classification_report(test_worldlab, world_predictions))
test_world.loc[:,'classifier'] = world_predictions
world_rows = test_world.index
world_labelcolumn.index = range(len(world_labelcolumn))
world_test_labels = world_labelcolumn.iloc[world_rows,:]
test_world.loc[:,'label'] = world_test_labels
test_world.loc[:,'scores'] = world_score
test_world.to_csv('world_firstclassifiedtestdata.csv')
filename = 'world_firstclassifier.sav'
joblib.dump(world_clf, filename)

#geoneutrinos
geo = signal.append(df7, ignore_index=True)
geo_labelcolumn = geo[['label']]
geo_label = geo_labelcolumn.to_numpy()
geo_label = geo_label.flatten()
geo = geo.drop(['label'], axis=1)
train_geo, test_geo, train_geolab, test_geolab = train_test_split(geo, geo_label, random_state=None)
geo_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                                n_estimators=100, learning_rate=0.1)
geo_clf.fit(train_geo, train_geolab)
geo_predictions = geo_clf.predict(test_geo)
geo_prob = geo_clf.predict_proba(test_geo)
geo_score = geo_clf.decision_function(test_geo)
print(confusion_matrix(test_geolab, geo_predictions))
disp = plot_confusion_matrix(geo_clf, test_geo, test_geolab)
disp.figure_.suptitle("Geoneutrino Background Classifier (training)")
plt.show()
print(classification_report(test_geolab, geo_predictions))
test_geo.loc[:,'classifier'] = geo_predictions
geo_rows = test_geo.index
geo_labelcolumn.index = range(len(geo_labelcolumn))
geo_test_labels = geo_labelcolumn.iloc[geo_rows,:]
test_geo.loc[:,'label'] = geo_test_labels
test_geo.loc[:,'scores'] = geo_score
test_geo.to_csv('geo_firstclassifiedtestdata.csv')
filename = 'geo_firstclassifier.sav'
joblib.dump(geo_clf, filename)
#create an roc curve for the signal classifications by each model
li9_signal =li9_prob[:,1]
n17_signal = n17_prob[:,1]
#neu_signal = neu_prob[:,1]
world_signal = world_prob[:,1]
geo_signal = geo_prob[:,1]

li9_fpr, li9_tpr, _ = roc_curve(li9_test_labels, li9_signal, pos_label=1)
li9_auc = auc(li9_fpr, li9_tpr)
n17_fpr, n17_tpr, __ = roc_curve(n17_test_labels, n17_signal, pos_label=1)
n17_auc = auc(n17_fpr, n17_tpr)
#neu_fpr, neu_tpr, ___ = roc_curve(neu_test_labels, neu_signal, pos_label=1)
#neu_auc = auc(neu_fpr, neu_tpr)
geo_fpr, geo_tpr, ____ = roc_curve(geo_test_labels, geo_signal, pos_label=1)
geo_auc = auc(geo_fpr, geo_tpr)
world_fpr, world_tpr, _____ = roc_curve(world_test_labels, world_signal, pos_label=1)
world_auc = auc(world_fpr, world_tpr)

plt.plot(li9_fpr, li9_tpr, marker=',', label='Lithium-9 (area = {:.2f})'.format(li9_auc))
plt.plot(n17_fpr, n17_tpr, marker=',', label='Nitrogen-17 (area = {:.2f})'.format(n17_auc))
#plt.plot(neu_fpr, neu_tpr, marker=',', label='Fast Neutrons (area = {:.2f})'.format(neu_auc))
plt.plot(geo_fpr, geo_tpr, marker=',', label='Geoneutrinos (area = {:.2f})'.format(geo_auc))
plt.plot(world_fpr, world_tpr, marker=',', label='Worldwide reactors (area = {:.2f})'.format(world_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Correlated Backgrounds Binary Classification (training)')
plt.show()

