import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

test_data = pd.read_csv("world_classifiedvaldata.csv") #read in data from text file
test_data=test_data.drop(['Unnamed: 0'],axis=1)
print(test_data)

ts = test_data.loc[(test_data.classifier==1) & (test_data.label==1)] 
tb = test_data.loc[(test_data.classifier==0) & (test_data.label==0)]
fs = test_data.loc[(test_data.classifier==1) & (test_data.label==0)]
fb = test_data.loc[(test_data.classifier==0) & (test_data.label==1)]

plt.hist(ts.scores.values.flatten(), bins=100,
		label='True Signal', alpha=.5) 
plt.hist(tb.scores.values.flatten(), bins=100,
		label='True Background', alpha=.5)
plt.hist(fs.scores.values.flatten(), bins=100, 
		label='False Signal', alpha=.5)
plt.hist(fb.scores.values.flatten(), bins=100,
		label='False Background', alpha=.5)
plt.yscale('log')
plt.title('Boulby World Background Classifier (validation)')
plt.legend(loc='best')
plt.xlabel('Decision scores')
plt.ylabel('Frequency (log)')
plt.show()
