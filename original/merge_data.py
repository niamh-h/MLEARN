#merging the individual text files into one
#filenames = ['newsignalRawData.txt', 'newli9RawData.txt', 'newn17RawData.txt', 'newworldRawData.txt', 'newsmall_reactorRawData.txt']
#with open('merged_data.txt', 'w') as outfile:
#	for fname in filenames:
#		with open(fname) as infile:
#			for line in infile:
#				outfile.write(line)
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('merged_data.txt', sep=' ', header=None) #putting into dataframe
data.columns = ['n9', 'inner_hit', 'dt_prev_us', 'beta_one', 'beta_two', 'beta_three', 'beta_four', 'beta_five', 'beta_six']
df = pd.DataFrame(data)
print(df)

#b12 = df.plot.scatter(x='beta_one',y='beta_two', s=0.01)
#plt.show()
#plt.savefig()

#col_choice = ["beta_one", "beta_two", "beta_three" , "beta_four", "beta_five", "beta_six"]
#for pos, axis1 in enumerate(col_choice): #picks a first column
#	for axis2 in enumerate(col_choice[pos+1:]):
#		print(df.loc[1,axis1])
#		print(axis2)
#		plt.scatter(df.reindex(axis1),df.reindex(axis2),s=0.01)
#		plt.show()
#		plt.savefig(str(axis1)+" vs " + str(axis2))

for column1 in df[['beta_one', 'beta_two', 'beta_three', 'beta_four', 'beta_five', 'beta_six']]:
	for column2 in df[['beta_one', 'beta_two', 'beta_three', 'beta_four', 'beta_five', 'beta_six']]:
		column1data = df[column1]
		column2data = df[column2]
		plt.scatter(column1data, column2data, s=0.01)
		plt.xlabel(str(column1))
		plt.ylabel(str(column2))
		plt.title(str(column1)+" vs "+str(column2))
		plt.show
		plt.savefig(str(column1)+" vs "+str(column2))

