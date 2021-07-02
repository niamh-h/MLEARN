# Machine-Learning-Project
/original/ contains old code from developing\n
indiv_modelstraining.py: Input text files of data from each background source: lithium9, nitrogen17, fast neutrons,\n
                          worldwide reactors and geoneutrinos, and signal from reactor.\n 
                          Outputs 5 trained boosted decision tree mnodels that classify a source of bg and the signal \n
                          from the reactor. Creates a confusion matrix and classification report for each model. makes \n
                          an roc curve for each source on one plot. Outputs the classified data into csv files. \n
