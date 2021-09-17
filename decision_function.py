import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

test_data = pd.read_csv("/path/to/classified/data") #Input the final classified data after either: just the fn model, just the li9 model, or both models
test_data=test_data.drop(['Unnamed: 0'],axis=1)
print(test_data)
#classifier column is 'fn_classifier' if only fn model has been  applied, 'li9_classifier' if only li9 or both models have been applied
ts = test_data.loc[(test_data.li9_classifier==0) & (test_data.label==0)] #ts = anything other than fast neutrons OR lithium-9 correctly classified as 'not neutron' OR 'not li9'
tb = test_data.loc[(test_data.li9_classifier==1) & (test_data.label==1)] #tb = neutrons OR li9 successsfully classified and thrown away
fs = test_data.loc[(test_data.li9_classifier==0) & (test_data.label==1)] #fs = neutrons classified as 'not neutrons' OR 'not li9'
fb = test_data.loc[(test_data.li9_classifier==1) & (test_data.label==0)] #fb = anything other than neutrons OR classified as neutrons OR li9
#shows the numbers of each source kept by model
li9 = fs.shape[0]
n17 = ts.loc[ts.source==3].shape[0]
world = ts.loc[ts.source==4].shape[0]
tor = ts.loc[ts.source==5].shape[0]
geo = ts.loc[ts.source==7].shape[0]
hey = ts.loc[ts.source==1].shape[0]
neu = ts.loc[ts.source==6].shape[0]

print('li9 ',li9, '\nn17 ', n17, '\nworld ', world, '\ngeoneutrinos ', geo, '\nheysham ', hey, '\nneutrons ', neu, '\ntorness ', tor)
#plot the events by their decision function score
plt.hist(ts.scores.values.flatten(), bins=50, label='True Other', alpha=.5)
plt.hist(tb.scores.values.flatten(), bins=50, label='True Lithium-9', alpha=.5) #or li9
plt.hist(fs.scores.values.flatten(), bins=50, label='False Other', alpha=.5)
plt.hist(fb.scores.values.flatten(), bins=50, label='False Lithium-9, alpha=.5) #or li9
plt.yscale('log')
plt.title('Decision Function title')
plt.legend(loc='best')
plt.xlabel('Decision scores')
plt.ylabel('Frequency (log)')
plt.savefig("/path/to/file")
plt.show()
# *** ONLY IF BOTH MODELS APPLIED ***
#display the number of events above 6 MeV
a = fs.loc[fs.mc_energy > 6].shape[0]
b = hey.loc[hey.mc_energy > 6].shape[0]
c = tor.loc[tor.mc_energy > 6].shape[0]
d = n17.loc[n17.mc_energy > 6].shape[0]
e = geo.loc[geo.mc_energy > 6].shape[0]
f = world.loc[world.mc_energy >6].shape[0]
g = neu.loc[neu.mc_energy > 6].shape[0]
print('li9' , a, '\nhey ', b, '\ntor ', c, '\nn17 ', d, '\ngeo ', e, '\nworld ', f, '\nneu ', g)
