import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

test_data = pd.read_csv("/path/to/classified/data")
test_data=test_data.drop(['Unnamed: 0'],axis=1)
print(test_data)

ts = test_data.loc[(test_data.classifier==0) & (test_data.label==0)] #ts = anything other than fast neutrons correctly classified as 'not neutron' 
tb = test_data.loc[(test_data.classifier==1) & (test_data.label==1)] #tb = neutrons successsfully classified and thrown away
fs = test_data.loc[(test_data.classifier==0) & (test_data.label==1)] #fs = neutrons classified as 'not neutrons'
fb = test_data.loc[(test_data.classifier==1) & (test_data.label==0)] #fb = anything other than neutrons classified as neutrons
#shows the numbers of each source kept by model
li9 = ts.loc[ts.source==3].shape[0]
n17 = ts.loc[ts.source==4].shape[0]
world = ts.loc[ts.source==5].shape[0]
geo = ts.loc[ts.source==7].shape[0]
hey = ts.loc[ts.source==1].shape[0]
neu = fs.shape[0]

print('li9 ',li9, '\nn17 ', n17, '\nworld ', world, '\ngeoneutrinos ', geo, '\nheysham ', hey)
print('neutrons ', neu)
#plot the events by their decision function score
plt.hist(ts.scores.values.flatten(), bins=50, label='True Other', alpha=.5)
plt.hist(tb.scores.values.flatten(), bins=50, label='True Fast Neutrons', alpha=.5)
plt.hist(fs.scores.values.flatten(), bins=50, label='False Other', alpha=.5)
plt.hist(fb.scores.values.flatten(), bins=50, label='False Fast Neutrons', alpha=.5)
plt.yscale('log')
plt.title('Decision Function title')
plt.legend(loc='best')
plt.xlabel('Decision scores')
plt.ylabel('Frequency (log)')
plt.savefig("/path/to/file")
plt.show()
