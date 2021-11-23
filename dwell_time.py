import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
import readline
import os
import glob

#read in data
path = input('Enter the path to the classified data csv file: \n')
data = pd.read_csv(path)
data = data.drop(['Unnamed: 0'],axis=1)
print(data)

#keep only what's classified as signal
ts = data.loc[(data.classifier==0) & (data.label==0)]
fs = data.loc[(data.classifier==0) & (data.label==1)]
fb = data.loc[(data.classifier==1) & (data.label==0)]
tb = data.loc[(data.classifier==1) & (data.label==1)]
model = input("\n\nWhich classifier output is this? (fn/lithium): ")
sigsource = input("\nSignal Source? (h2/t/t+h/hf/hp): ")
tanksize = int(input("\nInput tank size in m: "))
t_hp1,t_hp2,t_li9,t_n17,t_neu,t_world,t_hey,t_geo,t_tor=input(
                '\nEnter total Monte Carlo events for hartlepool1, hartlepool2, li9, n17, neutrons,  world, heysham (2 or full), geo, tor (in order). If a source has not been simulated enter "1".\n ').split(', ')

t_hp1=float(t_hp1)
t_hp2=float(t_hp2)
t_li9=float(t_li9)
t_n17=float(t_n17)
t_neu=float(t_neu)
t_world=float(t_world)
t_hey=float(t_hey)
t_geo=float(t_geo)
t_tor=float(t_tor)

geo = ts.loc[ts.source==2].shape[0]
n17 = ts.loc[ts.source==7].shape[0]
li9 = ts.loc[ts.source==6].shape[0]
world = ts.loc[ts.source==9].shape[0]
hp1 = ts.loc[ts.source==3].shape[0]
hp2 = ts.loc[ts.source==4].shape[0]
hey = ts.loc[ts.source==5].shape[0]
tor = ts.loc[ts.source==8].shape[0]
if model == "lithium":
	neu = ts.loc[ts.source==1].shape[0]
	li9 = fs.shape[0]
elif model == "fn":
	li9 = ts.loc[ts.source==6].shape[0]
	neu = fs.shape[0]
if (tanksize == 16):
	r_li9 = 1.25e-05 * 86400 #(seconds in a day)
	r_n17 = 7.64e-06 * 86400
	r_neu = 1.849e-02 * 86400
	r_geo = 2.4480e-06 * 86400
	r_worlda = 7.5470e-06 * 86400
	r_tor = 1.7000e-06 * 86400
	r_hey2 = 2.6520e-06 * 86400
	r_hey = 4.6920E-06 * 86400
	r_hp1 = 4.0390E-05 * 86400
	r_hp2 = 3.0240E-05 * 86400
	if sigsource=='h2':
		r_sig = r_hey2
		t_sig = t_hey
		r_world = r_worlda + r_tor
		t_world = t_world + t_tor
		signal = hey
	elif sigsource == 't':
		r_sig = r_tor
		t_sig = t_tor
		r_world = r_worlda + r_hey2
		t_world = t_world + t_tor
		signal = tor
	elif sigsource == 't+h':
		r_sig = r_tor + r_hey2
		t_sig = t_tor + t_hey
		r_world = r_worlda
		signal = tor + hey
	elif sigsource == 'hf':
		r_sig = r_hey
		t_sig = t_hey
		r_world = r_worlda + r_tor
		t_world = t_world + t_tor
		signal = hey
	elif sigsource == 'hp':
		r_sig = r_hp1 + r_hp2
		t_sig = t_hp1 + t_hp2
		r_world = r_worlda + r_hey + r_tor
		t_world = t_world + t_hey + t_tor
		signal = hp1 + hp2
	else:
		print('Not calibrated for this signal source')
		sys.exit()
	print('Tank size 16m, PSUP 5700mm: \n\nDaily Signal Rate: ', r_sig)
	print('\nDaily Background Rates: ', '\nli9: ', r_li9, '\nn17: ', r_n17, '\nneutrons: ', r_neu,  '\nworld: ', r_world, '\ngeo: ', r_geo)
elif (tanksize == 22):
	r_li9 = 3.25e-05 * 86400
	r_n17 = 1.99e-05 * 86400
	r_neu = 3.217e-02 * 86400
	r_geo = 6.3640e-06 * 86400
	r_worlda = 1.9620e-05 * 86400
	r_tor = 4.4190e-06 * 86400
	r_hey2 = 6.8940e-06 * 86400
	r_hey = 1.2200E-05 * 86400
	r_hp1 = 1.0500E-04 * 86400
	r_hp2 = 7.8610E-05 * 86400
	if sigsource=='h2':
		r_sig = r_hey2
		t_sig = t_hey
		r_world = r_worlda + r_tor
		t_world = t_world + t_tor
		signal = hey
	elif sigsource=='t':
		r_sig = r_tor
		t_sig = t_tor
		r_world = r_worlda + r_hey2
		t_world = t_world + t_hey
		signal = tor
	elif sigsource=='t+h':
		r_sig = r_tor + r_hey2
		t_sig = t_tor + t_hey
		r_world = r_worlda
		signal = hey + tor
	elif sigsource == 'hf':
		r_sig = r_hey
		t_sig = t_hey
		r_world = r_worlda + r_tor
		t_world = t_world + t_tor
		signal = hey
	elif sigsource == 'hp':
		r_sig = r_hp1 + r_hp2
		t_sig = t_hp1 + t_hp2
		r_world = r_worlda + r_hey + r_tor
		t_world = t_world + t_hey + t_tor
		signal = hp1 + hp2
	else:
                print('Not calibrated for this signal source')
                sys.exit()
	print('Tank size 22m, PSUP 9000mm: \n\nDaily Signal Rate: ', r_sig)
	print('\nDaily Background Rates: ', '\nli9: ', r_li9, '\nn17: ', r_n17, '\nneutrons: ', r_neu, '\nworld: ', r_world, '\ngeo: ', r_geo)
elif (tanksize == 28):
	r_li9 = 6.70E-05 * 86400
	r_n17 = 4.09E-05 * 86400
	r_neu = 0.04962 * 86400
	r_geo = 1.312E-05 * 86400
	r_worlda = 4.044E-05 * 86400
	r_tor = 9.108E-06 * 86400
	r_hey2 = 1.421E-05 * 86400
	r_hey = 2.514E-05 * 86400
	r_hp1 = 0.0002164 * 86400
	r_hp2 = 0.000162 * 86400
	if sigsource == 'h2':
		r_sig = r_hey2
		t_sig = t_hey
		r_world = r_worlda + r_tor
		t_world = t_world + t_tor
		signal = hey
	elif sigsource=='t':
		r_sig = r_tor
		t_sig = t_tor
		r_world = r_worlda + r_hey2
		t_world = t_world + t_hey
		signal = tor
	elif sigsource=='t+h':
		r_sig = r_tor + r_hey2
		t_sig = t_tor + t_hey
		r_world = r_worlda
		signal = tor + hey
	elif sigsource == 'hf':
		r_sig = r_hey
		t_sig = t_hey
		r_world = r_worlda + r_tor
		t_world = t_world + t_tor
		signal = hey
	elif sigsource == 'hp':
		r_sig = r_hp1 + r_hp2
		t_sig = t_hp1 + t_hp2
		r_world = r_worlda + r_hey + r_tor
		t_world = t_world + t_hey + t_tor
		signal = hp1 + hp2
	else:
		print('Not calibrated for this signal source')
		sys.exit()
	print('Tank size 28m, PSUP 9000mm: \n\nDaily Signal Rate: ', r_sig)
	print('\nDaily Background Rates: ', '\nli9: ', r_li9, '\nn17: ', r_n17, '\nneutrons: ', r_neu, '\nworld: ', r_world, '\ngeo: ', r_geo)
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

#efficiency * rate = new rate
b_li9 = e_li9 * r_li9
b_n17 = e_n17 * r_n17
b_neu = e_neu * r_neu
b_world = e_world * r_world
b_geo = e_geo * r_geo
s_sig = e_sig * r_sig
print('Rates after machine learning:')
print('\n\nLithium rate: ', b_li9, '\nNitrogen rate: ', b_n17, '\nNeutrons rate: ', b_neu, '\nWorld rate: ', b_world, '\nGeoneutrinos rate: ', b_geo)
print('Signal rate: ', s_sig)

#put into the significance equation
t = (9 * (b_li9 + b_n17 + b_neu + b_world + b_geo))/(s_sig)**2
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
A = b_li9 + b_n17 + b_neu + b_world + b_geo
dA = ( (li9_err)**2 + (n17_err)**2 + (neu_err)**2 + (world_err)**2 + (geo_err)**2 )**(1/2)
C = s_sig ** 2
dC = 2 * s_sig * sig_err
dt = t * ( (dA/A)**2 + (dC/C)**2 )**(1/2)
print('\nDwell time, (days): ', t, ' +/- ', dt)

