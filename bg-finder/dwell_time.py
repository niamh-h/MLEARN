import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
import readline
#read in data from voting classifier
data = pd.read_csv("/path/to/final/classified/data.csv")
data = data.drop(['Unnamed: 0'],axis=1)
print(data)

#keep only what's classified as signal
ts = data.loc[(data.li9_classifier==0) & (data.label==0)]
fs = data.loc[(data.li9_classifier==0) & (data.label==1)]
fb = data.loc[(data.li9_classifier==1) & (data.label ==0)]
tb = data.loc[(data.li9_classifier==1)  & (data.label == 1)]
print('fb, n17: ', fb.loc[fb.source==4].shape[0]) #shows what is lost from each source
print('fb, world: ', fb.loc[fb.source==5].shape[0])
print('fb, geo: ', fb.loc[fb.source==7].shape[0])
print('fb, signal: ', fb.loc[fb.source==1].shape[0], '\n')
print('tb, li9: ', tb.shape[0], '\n')
li9 = fs.shape[0] 
n17 = ts.loc[ts.source==4].shape[0]
world = ts.loc[ts.source==5].shape[0]
neu = ts.loc[ts.source==6].shape[0]
geo = ts.loc[ts.source==7].shape[0]
print('li9 ',li9, '\nn17 ', n17, '\nworld ', world, '\ngeoneutrinos ', geo, '\nneutrons ', neu)
signal = ts.loc[ts.source==1].shape[0]
print('signal ', signal)
li9_6  = int(input("Input no. li9 events >6MeV: ")) #Number of li9 events that can be removed by energy fitter
li9 = li9 - (2*li9_6)
tanksize = int(input("Input tank size in m: "))
t_li9,t_n17,t_neu,t_world,t_hey,t_geo,t_tor=input(
                'Enter total simulated events for li9, n17, neutrons,  world, heysham (2 or full), geo, tor (in order): ').split(', ')
t_li9=float(t_li9)
t_n17=float(t_n17)
t_neu=float(t_neu)
t_world=float(t_world)
t_hey=float(t_hey)
t_geo=float(t_geo)
t_tor=float(t_tor)

if (tanksize == 16):
	r_li9 = 1.25e-05 * 86400 #(seconds in a day) 
	r_n17 = 7.64e-06 * 86400
	r_neu = 1.849e-02 * 86400
	r_geo = 2.4480e-06 * 86400
	r_worlda = 7.5470e-06 * 86400
	r_tor = 1.7000e-06 * 86400
	r_hey2 = 2.6520e-06 * 86400
	r_hey = 4.6920E-06 * 86400
	sigsource = input("Signal Source? (h2/t/t+h/hf): ")
	if sigsource=='h2':
		r_sig = r_hey2
		t_sig = t_hey
		r_world = r_worlda + r_tor
		t_world = t_world + t_tor
	elif sigsource == 't':
		r_sig = r_tor
		t_sig = t_tor
		r_world = r_worlda + r_hey2
		t_world = t_world + t_tor
	elif sigsource == 't+h':
		r_sig = r_tor + r_hey2
		t_sig = t_tor + t_hey
		r_world = r_worlda
	elif sigsource == 'hf':
		r_sig = r_hey
		t_sig = t_hey
		r_world = r_worlda + r_tor
		t_world = t_world + t_tor
	else:
		print('Not calibrated for this signal source')
		sys.exit()
	print('Tank size 16m, PSUP 5700mm: \n\nDaily Signal Rate: ', r_sig)
	print('\nDaily Background Rates: ', '\nli9: ', r_li9, '\nn17: ', r_n17, '\nneutrons: ', r_neu,  '\nworlda: ', r_worlda)
	print('geo: ', r_geo, '\ntor: ', r_tor, '\nhey: ', r_hey)
elif (tanksize == 22):
	r_li9 = 3.25e-05 * 86400
	r_n17 = 1.99e-05 * 86400
	r_neu = 3.217e-02 * 86400
	r_geo = 6.3640e-06 * 86400
	r_worlda = 1.9620e-05 * 86400
	r_tor = 4.4190e-06 * 86400
	r_hey2 = 6.8940e-06 * 86400
	r_hey = 1.2200E-05 * 86400
	sigsource = input("Signal Source? (h2/t/t+h): ")
	if sigsource=='h2':
		r_sig = r_hey2
		t_sig = t_hey
		r_world = r_worlda + r_tor
		t_world = t_world + t_tor
	elif sigsource=='t':
		r_sig = r_tor
		t_sig = t_tor
		r_world = r_worlda + r_hey2
		t_world = t_world + t_hey
	elif sigsource=='t+h':
		r_sig = r_tor + r_hey2
		t_sig = t_tor + t_hey
		r_world = r_worlda
	elif sigsource == 'hf':
		r_sig = r_hey
		t_sig = t_hey
		r_world = r_worlda + r_tor
		t_world = t_world + t_tor
	else:
                print('Not calibrated for this signal source')
                sys.exit()
	print('Tank size 22m, PSUP 9000mm: \n\nDaily Signal Rate: ', r_sig)
	print('\nDaily Background Rates: ', '\nli9: ', r_li9, '\nn17: ', r_n17, '\nneutrons: ', r_neu, '\nworlda: ', r_worlda)
	print('geo: ', r_geo, '\ntor: ', r_tor, '\nhey: ', r_hey, '\n\n')
elif (tanksize == 28):
	r_li9 = 6.70E-05 * 86400
	r_n17 = 4.09E-05 * 86400
	r_neu = 0.04962 * 86400
	r_geo = 1.312E-05 * 86400
	r_worlda = 4.044E-05 * 86400
	r_tor = 9.108E-06 * 86400
	r_hey2 = 1.421E-05 * 86400
	r_hey = 2.514E-05 * 86400
	sigsource = input("Signal Source? (h2/t/t+h): ")
	if sigsource == 'h2':
		r_sig = r_hey2
		t_sig = t_hey
		r_world = r_worlda + r_tor
		t_world = t_world + t_tor
	elif sigsource=='t':
		r_sig = r_tor
		t_sig = t_tor
		r_world = r_worlda + r_hey2
		t_world = t_world + t_hey
	elif sigsource=='t+h':
		r_sig = r_tor + r_hey2
		t_sig = t_tor + t_hey
		r_world = r_worlda
	elif sigsource == 'hf':
		r_sig = r_hey
		t_sig = t_hey
		r_world = r_worlda + r_tor
		t_world = t_world + t_tor
	else:
		print('Not calibrated for this signal source')
		sys.exit()
	print('Tank size 28m, PSUP 9000mm: \n\nDaily Signal Rate: ', r_sig)
	print('\nDaily Background Rates: ', '\nli9: ', r_li9, '\nn17: ', r_n17, '\nneutrons: ', r_neu, '\nworlda: ', r_worlda)
	print('geo: ', r_geo, '\ntor: ', r_tor, '\nhey: ', r_hey, '\n\n')
else:
	print('Unknown detector set-up, please add rates to script')
	sys.exit()
	
#efficiency of model on each source
e_li9 = li9/t_li9
e_n17 = n17/t_n17
e_neu = neu/t_neu
e_world = world/t_world
e_sig = signal/t_sig
e_geo = geo/t_geo
print('\n\ne_li9: ', e_li9, '\ne_n17: ', e_n17, '\ne_neu: ',e_neu, '\ne_world: ', e_world, '\ne_sig: ', e_sig, '\ne_geo: ' , e_geo)

#efficiency * rate = new rate
b_li9 = e_li9 * r_li9
b_n17 = e_n17 * r_n17
b_neu = e_neu * r_neu
b_world = e_world * r_world
b_geo = e_geo * r_geo
s_sig = e_sig * r_sig
print('\n\nLithium rate: ', b_li9, '\nNitrogen rate: ', b_n17, '\nNeutrons rate: ', b_neu, '\nWorld rate: ', b_world, '\nGeoneutrinos rate: ', b_geo)
print('Signal rate: ', s_sig)

#put into the significance equation
t_one = (9 * (b_li9 + b_n17 + b_neu + b_world + b_geo))/(s_sig)**2
print('Dwell Time correlated, both cores (days): ', t_one)
#errors
#Binomial error on efficiency for each bg and signal is 1/N * (events(1-events/total))**1/2
print('\nBinomial errors')
li9_eff_err = (1/t_li9) * (li9 * (1 - li9/t_li9))**(1/2)
n17_eff_err = (1/t_n17) * (n17 * (1 - n17/t_n17))**(1/2)
neu_eff_err = (1/t_neu) * (neu * (1 - neu/t_neu))**(1/2)
world_eff_err = (1/t_world) * (world * (1 - world/t_world))**(1/2)
sig_eff_err = (1/t_sig) * (signal * (1 - signal/t_sig))**(1/2)
geo_eff_err = (1/t_geo) * (geo * (1 - geo/t_geo))**(1/2)

#multiply the error by the rate to get the error on each rate
li9_err = li9_eff_err * r_li9
n17_err = n17_eff_err * r_n17
neu_err = neu_eff_err * r_neu
world_err = world_eff_err * r_world
geo_err = geo_eff_err * r_geo
sig_err = sig_eff_err * r_sig

#propagating errors
A_one = b_li9 + b_n17 + b_neu + b_world + b_geo
dA_one = ( (li9_err)**2 + (n17_err)**2 + (neu_err)**2 + (world_err)**2 + (geo_err)**2 )**(1/2)
C_one = s_sig ** 2
dC_one = 2 * s_sig * sig_err
dt_one = t_one * ( (dA_one/A_one)**2 + (2*sig_err/s_sig)**2 )**(1/2)
print('\nDwell time, (days): ', t_one, ' +/- ', dt_one)

