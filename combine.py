import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, auc
import sys
import os.path
import glob
import readline
import shutil
import uproot
#load models
route1 = input('\n\n Enter path to the Fast Neutron classifier to begin: \n\n')
fn_clf = joblib.load(route1)
route2 = input('Enter path to the Lithium-9 classifier: \n\n')
clf = joblib.load(route2)
#load data
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

#FN model

fn = pd.concat(d.values(), ignore_index=True, keys=d.keys())
fn_lab = fn[['label']]
fn_lab = fn_lab.to_numpy()
fn_lab = fn_lab.flatten()
fn_source = fn[['source']]
fn_source = fn_source.to_numpy()
fn_source = fn_source.flatten()
fn = fn.drop(['label', 'source'], axis=1)
print ('\n\nData:')
print (fn)
print ('\n\nApplying Fast Neutron model...\n\n')
fn_pred = fn_clf.predict(fn)
fn_prob = fn_clf.predict_proba(fn)
fn_scores = fn_clf.decision_function(fn)
print ('\n\nClassification Report:')
print (classification_report(fn_lab,fn_pred)
print ('\n\nConfusion Matrix:')
cm = confusion_matrix(fn_lab,fn_pred,labels=fn_clf.classes_)
print (cm)
print ('\n\nCreating figures...')
ConfusionMatrixDisplay.from_predictions(fn_lab, fn_pred)
plt.title('Fast Neutron Finder confusion matrix')
plt.savefig('cm_fnfinder_combine.pdf')
plt.show(block=False)
plt.pause(3)
plt.clf()

fn_signal=fn_prob[;,1]
fn_fpr, fn_tpr, _ = roc_curve(fn_lab, fn_signal)
fn_auc = auc(fn_fpr, fn_tpr)
plt.plot(fn_fpr, fn_tpr, marker=',', label='Fast Neurons (area = {:.2f})'.format(fn_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Fast Neutron finder ROC')
plt.savefig('roc_fnfinder_combine.pdf')
plt.show(block=False)
plt.pause(3)
plt.clf()

fn.loc[:,'source'] = fn_source
fn.loc[:,'label'] = fn_lab
fn.loc[:,'fn_classifier'] = fn_pred

ts = fn.loc[(fn.fn_classifier==0) & (fn.label==0)]
tb = fn.loc[(fn.fn_classifier==1) & (fn.label==1)]
fs = fn.loc[(fn.fn_classifier==0) & (fn.label==1)]
fb = fn.loc[(fn.fn_classifier==1) & (fn.label==0)]
#plot the events by their decision function score
plt.hist(ts.scores.values.flatten(), bins=50, label='True Other', alpha=.5)
plt.hist(tb.scores.values.flatten(), bins=50, label='True Fast Neutron', alpha=.5)
plt.hist(fs.scores.values.flatten(), bins=50, label='False Other', alpha=.5)
plt.hist(fb.scores.values.flatten(), bins=50, label='False Fast Neutron', alpha=.5)
plt.yscale('log')
plt.title('Fast Neutron Finder scores')
plt.legend(loc='best')
plt.xlabel('Decision scores')
plt.ylabel('Frequency (log)')
plt.savefig('df_fnfinder_combine.pdf')
plt.show(block=False)
plt.pause(3)
plt.clf()

'''
fn.loc[:,'fn_scores'] = fn_scores
fn.loc[:,'prob_fn'] = fn_prob[:,1]
fn.loc[:,'prob_otherfn'] = fn_prob[:,0]
fn.loc[:,'mc_energy'] = fn_en
print(fn)
fn.to_csv('/path/to/file.csv')
'''
li9 = fn.loc[fn.fn_classifier==0] #keep only what fn_finder leaves
li9.label.replace(to_replace=1,value=0,inplace=True) #replace labels as 0
li9.loc[li9.source==2, 'label'] = 1 #relabel li9 as 1
print(li9)
print(li9.loc[li9.source==2])

li9_lab = li9['label']
li9_lab = li9_lab.to_numpy()
li9_lab = li9_lab.flatten()
li9_source = li9['source']
li9_source = li9_source.to_numpy()
li9_source = li9_source.flatten()
li9 = li9.drop(['label', 'source', 'fn_classifier'],axis=1)
print ('\n\nNew data (no fast neutrons):')
print(li9)
print ('Applying Lithium-9 model...')
li9_pred = li9_clf.predict(li9)
li9_prob = li9_clf.predict_proba(li9)
li9_scores = li9_clf.decision_function(li9)
print ('\n\nClassification Report:')
print (classification_report(li9_lab, li9_pred))
cm1 = confusion_matrix(li9_lab,li9_pred,labels=li9_clf.classes_)
print ('\n\nConfusion Matrix:')
print (cm1)
print ('\n\nCreating Figures...')
ConfusionMatrixDisplay.from_predictions(li9_lab, li9_pred)
plt.title('Lithium-9 Finder confusion matrix')
plt.savefig('cm_li9finder_combine.pdf')
plt.show(block=False)
plt.pause(3)
plt.clf()

li9_signal=li9_prob[;,1]
li9_fpr, li9_tpr, __ = roc_curve(li9_lab, li9_signal)
li9_auc = auc(li9_fpr, li9_tpr)
plt.plot(li9_fpr,li9_tpr, marker=',', label='Lithium-9 (area = {:.2f})'.format(li9_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Lithium-9 finder ROC')
plt.savefig('roc_li9finder_combine.pdf')
plt.show(bloack=False)
plt.pause(3)
plt.clf()

li9.loc[:,'li9_classifier'] = li9_pred
li9.loc[:,'source'] = li9_source
li9.loc[:,'label'] = li9_lab
li9.loc[:,'li9_scores'] = li9_scores
li9.loc[:,'prob_li9'] = li9_prob[:,1]
li9.loc[:,'prob_otherli9'] = li9_prob[:,0]

ts = li9.loc[(li9.li9_classifier==0) & (li9.label==0)]
tb = li9.loc[(li9.li9_classifier==1) & (li9.label==1)]
fs = li9.loc[(li9.li9_classifier==0) & (li9.label==1)]
fb = li9.loc[(li9.li9_classifier==1) & (li9.label==0)]
#plot the events by their decision function score
plt.hist(ts.scores.values.flatten(), bins=50, label='True Other', alpha=.5)
plt.hist(tb.scores.values.flatten(), bins=50, label='True Lithium-9', alpha=.5)
plt.hist(fs.scores.values.flatten(), bins=50, label='False Other', alpha=.5)
plt.hist(fb.scores.values.flatten(), bins=50, label='False Lithium-9', alpha=.5)
plt.yscale('log')
plt.title('Lithium-9 Finder scores')
plt.legend(loc='best')
plt.xlabel('Decision scores')
plt.ylabel('Frequency (log)')
plt.savefig('df_li9finder_combine.pdf')
plt.show(block=False)
plt.pause(3)
plt.clf()

df1 = X.loc[X.source==1]
df2 = X.loc[X.source==2]
df3 = X.loc[X.source==3]
df4 = X.loc[X.source==4]
df5 = X.loc[X.source==5]
df6 = X.loc[X.source==6]
df7 = X.loc[X.source==7]
df8 = X.loc[X.source==8]
df9 = X.loc[X.source==9]
print ('Copying and updating ROOT files...\n\n')
d1 = [df1,df2,df3,df4,df5,df6,df7,df8,df9]
for df in d1:
  df = df.drop(['source', 'label'], axis=1)
  for file in filenames:
    fil = shutil.copy(file, os.getcwd())
    with uproot.update(fil) as f3:
      f3.mktree("M",{"prediction":"i8","score":"f8","li9_prob":"f8","other_prob":>
      f3["M"].extend({"prediction":df.loc[:,"li9_classifier"], "score":df.loc[:,"li9_scores"], "li9_prob":df.loc[:,"prob_li9"], "other_prob":df.loc[:,"prob_otherli9"]})

print ('END')

