# Machine-Learning-Project
-- /original/ -- : contains old code from developing

-- indiv_modelstraining.py -- : Input text files of data from each background source: lithium9, nitrogen17, fast neutrons, worldwide reactors and geoneutrinos, and signal from reactor. Outputs 5 trained boosted decision tree mnodels that classify a source of bg and the signal from the reactor. Creates a confusion matrix and classification report for each model and an roc curve for each source on one plot. Outputs the classified data into csv files.

-- decision_function.py -- : Input: the file name of the classified data. Output: plot of the decision function of the model.

-- validation.py -- : Input validation data and models. Outputs classified data. Gives ROC, confusion matrix, classification report and permutation importances for each model.
