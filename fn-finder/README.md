dwell_time.py : input the classified data, the size of the detector, the total MC and remaining events for each source. Specify the signal source from: heysham 2, 
                heysham full, torness or torness+heysham 2. Outputs the daily rate of each source after classification and the time to get to 3sigma (dwell time). 
                
model_training.py : input data from the detector, outputs a trained fast neutron finder, an roc curve and confusion matrix. Outputs the classified test data to a csv file and the trained model to a .sav file.

validation.py : input validation data and the trained model. Outputs the classified data into a csv file and creates an ROC curve and confusion matrix. 
