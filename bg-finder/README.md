ALL INFORMATION IS IN 'Guide_to_Machine_Learning_scripts.pdf'


***

dwell_time.py : input the classified data, the size of the detector and the total MC. Specify the signal source from: heysham 2, 
                heysham full, torness or torness+heysham 2. Outputs the daily rate of each source after classification and the time to get to 3sigma (dwell time). 
                
fn_finder.py : input data from the detector, outputs a trained fast neutron finder, an roc curve and confusion matrix. Outputs the classified test data to a csv file and the trained model to a .sav file.

li9_finder : input data (no fast neutrons), outputs a trained lithium-9 finder, roc curve and confusion matrix. Outputs classified test data to a csv file and the trained
model to a .sav file. 

validation.py : input validation data and the trained model (lithium-9 or fast neutron finder). Outputs the classified data into a csv file and creates an ROC curve and confusion matrix. 

combine.py : input validation data and both the lithium and fast neutron finders. Runs the fast neutron finder on the data and gives an roc, cm and outputs the classified data to a csv. Then runs the lithium finder on the remaining data, with another roc, cm and outputs final classified data to a csv. 

extract.C, extract.h : input the .root data files, outputs the data in text files. 
