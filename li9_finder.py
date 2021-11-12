import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score, confusion_matrix, auc
import matplotlib.pyplot as plt
import joblib
import sys
import os
import glob
import readline
import uproot

input = input('Enter the path to the training data directory.\n End the file path with "/*.root".\n')
location = os.path.join(input)
filenames = glob.glob(location)
d = {}
for f in filenames:
  with uproot.open(f) as f1:
    data = f1["data"]
    b, a = f.rsplit('/',1)
    c, e = a.split('_',1)
    d[c] = data.arrays(['n100', 'n100_prev', 'inner_hit', 'inner_hit_prev', 'dt_prev_us', 'beta_one', 'beta_one_prev',
                         'beta_two', 'beta_two_prev', 'beta_three', 'beta_three_prev', 'beta_four', 'beta_four_prev',
                         'beta_five', 'beta_five_prev', 'beta_six', 'beta_six_prev', 'drPrevr'], '(n100>0)&(n100_prev>0)',library='pd')
for x, y in d.items():
  if (x == 'li9'):
    y['label'] = 1
  else:
    y['label'] = 0

#li9 model
X = pd.concat(d.values(), ignore_index=True, keys=d.keys())
ydf = X[['label']]
y = ydf.to_numpy()
y = y.flatten()
X = X.drop(['label'], axis=1)
print('Data:')
print(X)
print('Labels:')
print(y)
train_X, test_X, train_y, test_y = train_test_split(X, y, stratify = y)
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                                n_estimators=100, learning_rate=0.1)
print('Fitting your classifier...')
clf.fit(train_X.values, train_y)
pred = clf.predict(test_X.values)
prob = clf.predict_proba(test_X.values)
score = clf.decision_function(test_X.values)

print('Saving your classifier...')
clf_file = 'li9_finder.sav'
joblib.dump(clf, clf_file)

cm = confusion_matrix(test_y, pred, labels=clf.classes_)
print('Confusion Matrix:')
print (cm)
print ('Classification Report:')
print(classification_report(test_y,pred))
print ('Creating Figures...')
ConfusionMatrixDisplay.from_predictions(test_y, pred)
plt.title('Li9 finder, training')
plt.savefig('cm_li9finder_training.pdf')
#plt.show()
plt.clf()

test_X.loc[:,'classifier'] = pred
rows = test_X.index
ydf.index = range(len(ydf))
labels = ydf.iloc[rows,:]
test_X.loc[:,'label'] = labels
test_X.loc[:,'scores'] = score
signal = prob[:,1]
fpr, tpr, _ = roc_curve(labels, signal)
auc = auc(fpr, tpr)
plt.plot(fpr, tpr, marker=',', label= 'Lithium-9 (area = {:.2f}'.format(auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Lithium-9 finder ROC curve')
plt.savefig('roc_li9finder_training.pdf')
#plt.show()
plt.clf()

ts = test_X.loc[(test_X.classifier==0) & (test_X.label==0)]
tb = test_X.loc[(test_X.classifier==1) & (test_X.label==1)]
fs = test_X.loc[(test_X.classifier==0) & (test_X.label==1)]
fb = test_X.loc[(test_X.classifier==1) & (test_X.label==0)]
#plot the events by their decision function score
plt.hist(ts.scores.values.flatten(), bins=50, label='True Other', alpha=.5)
plt.hist(tb.scores.values.flatten(), bins=50, label='True Lithium-9', alpha=.5)
plt.hist(fs.scores.values.flatten(), bins=50, label='False Other', alpha=.5)
plt.hist(fb.scores.values.flatten(), bins=50, label='False Lithium-9', alpha=.5)
plt.yscale('log')
plt.title('Lithium-9 Finder scores, training')
plt.legend(loc='best')
plt.xlabel('Decision scores')
plt.ylabel('Frequency (log)')
plt.savefig("df_li9finder_training.pdf")
#plt.show()
plt.clf()
