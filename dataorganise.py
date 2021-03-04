import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data1 = pd.read_csv("newsignalRawData.txt", sep=" ", header=None) #read in data from text file
data1.columns = ["n9", "inner_hit", "dt_prev_us", "beta_one", "beta_two", "beta_three", "beta_four", "beta_five", "beta_six"]		#name the columns
df1 = pd.DataFrame(data1)				#put into a data frame for the signal

data2 = pd.read_csv("newli9RawData.txt", sep=" ", header=None) #same for each background source
data2.columns = ["n9", "inner_hit", "dt_prev_us", "beta_one", "beta_two", "beta_three", "beta_four", "beta_five", "beta_six"]
df2 = pd.DataFrame(data2)

data3 = pd.read_csv("newn17RawData.txt", sep=" ", header=None)
data3.columns = ["n9", "inner_hit", "dt_prev_us", "beta_one", "beta_two", "beta_three", "beta_four", "beta_five", "beta_six"]
df3 = pd.DataFrame(data3)

data4 = pd.read_csv("newsmall_reactorRawData.txt", sep=" ", header=None)
data4.columns = ["n9", "inner_hit", "dt_prev_us", "beta_one", "beta_two", "beta_three", "beta_four", "beta_five", "beta_six"]
df4 = pd.DataFrame(data4) 

data5 = pd.read_csv("newworldRawData.txt", sep=" ", header=None)
data5.columns = ["n9", "inner_hit", "dt_prev_us", "beta_one", "beta_two", "beta_three", "beta_four", "beta_five", "beta_six"]
df5 = pd.DataFrame(data5)

df1['label'] = 1
df2['label'] = 0
df3['label'] = 0
df4['label'] = 0 #labels each frame as signal or background so the success of the algorithm can be known at the end.
df5['label'] = 0

df1 = df1.head(5000)
df2 = df2.head(5000)
df3 = df3.head(5000)
df4 = df4.head(5000)
df5 = df5.head(5000)

print(df1)
print(df2)
print(df3)
print(df4)
print(df5)

frames = [df1, df2, df3, df4, df5] #array of the frames i am merging
merged_df = pd.concat(frames) #concat appends the columns of each dataframe and makes into a new one

#merged_df = merged_df.drop(merged_df[merged_df.n9 <= 8].index) #apply a cut to the data to get rid of any event where the n9 value was less than or equal to 8

ax = df1.plot(x='dt_prev_us')
#df2.plot(ax=ax)
#df3.plot(ax=ax)
#df4.plot(ax=ax)
#df5.plot(ax=ax)

plt.show()
