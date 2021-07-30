import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

test_data = pd.read_csv("/mnt/c/Users/Niamh/Documents/SummerJob/data/22m_gdh20/tn_uncal_clf/22h20_clfclfdata.csv")
test_data=test_data.drop(['Unnamed: 0'],axis=1)
print(test_data)

ts = test_data.loc[(test_data.classifier==0) & (test_data.label==0)]
tb = test_data.loc[(test_data.classifier==1) & (test_data.label==1)]
fs = test_data.loc[(test_data.classifier==0) & (test_data.label==1)]
fb = test_data.loc[(test_data.classifier==1) & (test_data.label==0)]

li9 = ts.loc[ts.source==3].shape[0]
n17 = ts.loc[ts.source==4].shape[0]
world = ts.loc[ts.source==5].shape[0]
geo = ts.loc[ts.source==7].shape[0]
hey = ts.loc[ts.source==1].shape[0]
neu = fs.shape[0]
print(fs)
print('li9 ',li9, '\nn17 ', n17, '\nworld ', world, '\ngeoneutrinos ', geo, '\nheysham ', hey)
print('neutrons ', neu)

plt.hist(ts.scores.values.flatten(), bins=50, label='True Other', alpha=.5)
plt.hist(tb.scores.values.flatten(), bins=50, label='True Fast Neutrons', alpha=.5)
plt.hist(fs.scores.values.flatten(), bins=50, label='False Other', alpha=.5)
plt.hist(fb.scores.values.flatten(), bins=50, label='False Fast Neutrons', alpha=.5)
plt.yscale('log')
plt.title('Fast Neutrons finder, Torness signal (final validation, 22m_gdh20)')
plt.legend(loc='best')
plt.xlabel('Decision scores')
plt.ylabel('Frequency (log)')
plt.savefig("/mnt/c/Users/Niamh/Documents/SummerJob/figures/22m_gdh20/tn_uncal/df_clf22h20.png")
plt.show()
