# Machine-Learning-Project
-- /original/ -- : contains old code from developing

-- /multiple-models/ -- contains the following files: 

  -- indiv_modelstraining.py -- : Input text files of data from each background source: lithium9, nitrogen17, fast neutrons, worldwide reactors and geoneutrinos, and signal        from reactor. Outputs 5 trained boosted decision tree models that classify a source of bg and the signal from the reactor. Creates a confusion matrix and classification        report for each model and an roc curve for each source on one plot. Outputs the classified data into csv files.

  -- decision_function.py -- : Input: the file name of the classified data. Output: plot of the decision function of the model.

  -- validation.py -- : Input validation data and models. Outputs classified data. Gives ROC, confusion matrix, classification report and permutation importances for each          model.

  -- voting_training -- : Input training data files and the 5 trained models. Outputs a voting classifier and the success of its performance on the data. 

  -- voting_validation.py -- : Input validation data and the voting classifier model. Outputs the success of the model on this data and the classified data into a csv file. 

  -- dwell_time.py -- : Input the filename of the classified data by the voting classifier. Outputs the dwell time with an error for the separation acheived by the model. 

-- /fn-finder/ -- contains the following files:
  
   -- model_training.py -- :
   
   -- validation.py -- : 
   
   -- dwell_time.py -- : 
