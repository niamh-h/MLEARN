import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score, confusion_matrix, auc
import joblib
import sys
import os.path
import glob
import readline
import shutil

#load the model
clf_route = input('\n\n Enter path to the classifier to begin: \n\n')
clf = joblib.load(clf_route)
#load the validation data
input = input('\n\nEnter the path to the validation data directory.\n\n *** End the file path with "/*.root ***.\n\n')
location = os.path.join(input)
filenames = sorted(glob.glob(location))
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
  if (x == 'fn'):
    y['label'] = 1
    y['source'] = 1
  elif (x == 'geo'):
    y['label'] = 0
    y['source'] = 2
  elif (x == 'hartlepool1'):
    y['label'] = 0
    y['source'] = 3
  elif (x == 'hartlepool2'):
    y['label'] = 0
    y['source'] = 4
  elif (x == 'heysham'):
    y['label'] = 0
    y['source'] = 5
  elif (x == 'li9'):
    y['label'] = 0
    y['source'] = 6
  elif (x == 'n17'):
    y['label'] = 0
    y['source'] = 7
  elif (x == 'torness'):
    y['label'] = 0
    y['source'] = 8
  elif (x == 'world'):
    y['label'] = 0
    y['source'] = 9
  else:
    print('Please rename your root files!')
    break

#neutron model
X = pd.concat(d.values(), ignore_index=True, keys=d.keys())
y = X[['label']]
y = y.to_numpy()
y = y.flatten()
source = X[['source']]
source = source.to_numpy()
source = source.flatten()
X = X.drop(['label', 'source'], axis=1)
print('\n\nData:')
print(X)
print('Applying the model ...\n\n')
pred = clf.predict(X)
prob = clf.predict_proba(X)
scores = clf.decision_function(X)
print ('Classification Report:')
print (classification_report(y, pred))
cm = confusion_matrix(y, pred, labels=clf.classes_)
print ('\n\nConfusion Matrix: ')
print(cm)
print ('\n\nCreating figures ...')
ConfusionMatrixDisplay.from_predictions(test_y, pred)
plt.title("Fast Neutron finder Validation")
plt.savefig('cm_fnfinder_validation.pdf')
plt.show(block=False)
plt.clf()

signal=prob[:,1]
fpr, tpr, _ = roc_curve(y,signal)
auc = auc(fpr, tpr)
plt.plot(fpr, tpr, marker=',', label='Fast Neutrons (area = {:.2f})'.format(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Fast Neutron finder Validation')
plt.savefig('roc_fnfinder_validation.pdf')
plt.show(block=False)
plt.clf()

ts = X.loc[(X.classifier==0) & (X.label==0)]
tb = X.loc[(X.classifier==1) & (X.label==1)]
fs = X.loc[(X.classifier==0) & (X.label==1)]
fb = X.loc[(X.classifier==1) & (X.label==0)]
#plot the events by their decision function score
plt.hist(ts.scores.values.flatten(), bins=50, label='True Other', alpha=.5)
plt.hist(tb.scores.values.flatten(), bins=50, label='True Fast Neutron', alpha=.5)
plt.hist(fs.scores.values.flatten(), bins=50, label='False Other', alpha=.5)
plt.hist(fb.scores.values.flatten(), bins=50, label='False Fast Neutron', alpha=.5)
plt.yscale('log')
plt.title('Fast Neutron Finder scores, validation')
plt.legend(loc='best')
plt.xlabel('Decision scores')
plt.ylabel('Frequency (log)')
plt.savefig('df_fnfinder_validation.pdf')
plt.show(block=False)
plt.clf()

X.loc[:,'classifier'] = pred
X.loc[:,'source'] = source
X.loc[:,'label'] = y
X.loc[:,'scores'] = scores
X.loc[:,'prob_fn'] = prob[:,1]
X.loc[:,'prob_other'] = prob[:,0]
print ('Added ML data:\n\n',X, '\n\n')
print('Outputting X to a csv...\n\n')
X.to_csv('classified_valdata.csv')

df1 = X.loc[X.source==1]
df2 = X.loc[X.source==2]
df3 = X.loc[X.source==3]
df4 = X.loc[X.source==4]
df5 = X.loc[X.source==5]
df6 = X.loc[X.source==6]
df7 = X.loc[X.source==7]
df8 = X.loc[X.source==8]
df9 = X.loc[X.source==9]
print('Copying and updating ROOT files...\n\n')
d1 = {df1,df2,df3,df4,df5,df6,df7,df8,df9}
for df in d1:
  df = df.drop(['source', 'label'], axis=1)
  for file in filenames:
    
    with uproot.update(file) as f3:
      f3.mktree("M",{"prediction":"i8","score":"f8","fn_prob":"f8","other_prob":"f8"} ,"ML_data")
      f3["M"] = df

print ('END')
