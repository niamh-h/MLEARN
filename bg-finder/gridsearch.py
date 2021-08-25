import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, plot_confusion_matrix, roc_curve, roc_auc_score, confusion_matrix, auc
from matplotlib import pyplot as plt
import joblib
import sys
#organise the data from the text files into a dataset
data2 = pd.read_csv("/mnt/c/Users/Niamh/Documents/SummerJob/data/22m_gdh20/test/li9_RawData22.txt", sep=" ", header=None)
data2.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev", "beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df2 = pd.DataFrame(data2)
df2['label'] = 0

data3 = pd.read_csv("/mnt/c/Users/Niamh/Documents/SummerJob/data/22m_gdh20/test/n17_RawData22.txt", sep=" ", header=None)
data3.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev",  "beta_three",
                "beta_three_prev","beta_four", "beta_four_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df3 = pd.DataFrame(data3)
df3['label'] = 0

data5 = pd.read_csv("/mnt/c/Users/Niamh/Documents/SummerJob/data/22m_gdh20/test/world_RawData22.txt", sep=" ", header=None)
data5.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df5 = pd.DataFrame(data5)
df5['label'] = 0
datat = pd.read_csv("/mnt/c/Users/Niamh/Documents/SummerJob/data/22m_gdh20/test/tornessfull_RawData22.txt", sep=" ", header=None)
datat.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
dft = pd.DataFrame(datat)
dft['label'] = 0

data6 = pd.read_csv("/mnt/c/Users/Niamh/Documents/SummerJob/data/22m_gdh20/test/fn_RawData22.txt", sep=" ", header=None)
data6.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df6 = pd.DataFrame(data6)
df6['label'] = 1

data7 = pd.read_csv("/mnt/c/Users/Niamh/Documents/SummerJob/data/22m_gdh20/test/geo_RawData22.txt", sep=" ", header=None)
data7.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev", "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df7 = pd.DataFrame(data7)
df7['label'] = 0

data8 = pd.read_csv("/mnt/c/Users/Niamh/Documents/SummerJob/data/22m_gdh20/test/heysham_2_RawData22.txt", sep=" ", header=None)
data8.columns = ["n9", "n9_prev", "dt_prev_us", "inner_hit", "inner_hit_prev", "beta_one", "beta_one_prev","beta_two", "beta_two_prev", "beta_three",
                "beta_three_prev", "beta_four", "beta_four_prev",  "beta_five", "beta_five_prev", "beta_six", "beta_six_prev", "good_pos", "good_pos_prev", "closestPMT"]
df8 = pd.DataFrame(data8)
df8['label'] = 0

df2 = df2.head(200)
df3 = df3.head(200)
df5 = df5.head(200)
dft = dft.head(200)
df6 = df6.head(200)
df7 =df7.head(200)
df8 = df8.head(200)

#neutron model
frames = [df3, df2, df5, dft, df7, df8]
neu = df6.append(frames, ignore_index=True)
neu_labelcolumn = neu[['label']]
neu_label = neu_labelcolumn.to_numpy()
neu_label = neu_label.flatten()
neu = neu.drop(['label'], axis=1)
train_neu, test_neu, train_neulab, test_neulab = train_test_split(neu, neu_label, random_state=None, stratify=neu_label)

dtc = DecisionTreeClassifier()
ada_classifier = AdaBoostClassifier(base_estimator=dtc)
ada_classifier.fit(train_neu, train_neulab)
ada_predictions = ada_classifier.predict(test_neu)
ada_pred_prob = ada_classifier.predict_proba(test_neu)

#Grid search code
grid = {"n_estimators" : [10, 50, 100, 200, 500],
        "learning_rate" : [0.0001, 0.001, 0.01, 0.1, 1.0],
        "base_estimator__max_depth" : [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40]
        }
#define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#define the grid search procedure
grid_search = GridSearchCV(estimator=ada_classifier, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

#execute the grid search
grid_result = grid_search.fit(test_neu, test_neulab)

#summarise the best score and config
print('Fast Neutron Finder optimised by:\n')
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#summarise all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means,stds, params):
       print("%f (%f) with: %r" % (mean, stdev, param))
