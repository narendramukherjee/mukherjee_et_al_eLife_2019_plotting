import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import theano.tensor as tt
import os
import pandas as pd
from scipy.special import digamma, gammaln
from scipy.stats import zscore

# Load data from respective folders
os.chdir("/media/patience/resorted_data/Plots/2500ms_EMG")
gapes_2500 = np.load("gapes.npy")
os.chdir("/media/patience/resorted_data/Plots/500ms_EMG")
gapes_500 = np.load("gapes.npy")
os.chdir("/media/patience/resorted_data/Jenn_Data/EMG_Plots")
gapes_Jenn = np.load("gapes.npy")
os.chdir("/media/patience/resorted_data/Plots/EM_switch")
gapes_before_700 = np.load("Gapes_before_700_Dur500_Lag700.npy")
gapes_after_700 = np.load("Gapes_after_700_Dur500_Lag700.npy")
os.chdir("/media/patience/resorted_data/late_analysis/Plots")
late_switchpoints_expanded = np.load("Switchpoint2_Dur500_Lag1400.npy")

# The probabilities in the gapes arrays are all close to 0.0 or 1.0. That means that movements in 3.5-6Hz are an important component of the animal's orofacial behaviors
# We therefore convert these probabilities to 0 or 1 and treat them as binary events and use beta distributions to model the probability distribution of those binary events
gapes_2500 = (gapes_2500 > 0.5).astype('int')
gapes_500 = (gapes_500 > 0.5).astype('int')
gapes_Jenn = (gapes_Jenn > 0.5).astype('int')
for i in range(4):
	gapes_before_700[i] = (gapes_before_700[i] > 0.5).astype('int')
	gapes_after_700[i] = (gapes_after_700[i] > 0.5).astype('int')

# Define a function to calculate the kl divergence between 2 beta (or equal sized Dirichlet) distributions
def kl(alpha, beta):
	return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) - gammaln(np.sum(beta)) + np.sum(gammaln(beta)) + np.sum((alpha-beta)*(digamma(alpha) - digamma(np.sum(alpha))))

# We calculate the kl divergence between the beta distributions for gaping probabilities inferred from the data in the gapes arrays.
# The parameters of these posterior beta distributions are: alpha = # of trials where gaping probability = 1, beta = # of trials where gaping probability = 0
# We place very weak priors of alpha = 1, beta = 1
kl_2500 = np.zeros((gapes_2500.shape[0], gapes_2500.shape[-1]))
kl_500 = np.zeros((gapes_500.shape[0], gapes_500.shape[-1]))
kl_Jenn = np.zeros((gapes_Jenn.shape[0], gapes_Jenn.shape[-1]))
kl_before_700 = np.zeros(gapes_before_700[3].shape[1])
kl_after_700 = np.zeros(gapes_after_700[3].shape[1])
kl_late_expanded = np.zeros(gapes_500.shape[-1])
kl_late_expanded_after_laser = np.zeros(gapes_500.shape[-1])

# Calculate the kl divergences
# The 2500ms condition
for i in range(gapes_2500.shape[0]):
	for j in range(gapes_2500.shape[-1]):
		conc = np.array([np.sum(gapes_2500[i, 3, :, j]) + 1, np.sum(1 - gapes_2500[i, 3, :, j]) + 1])
		dil = np.array([np.sum(gapes_2500[i, 2, :, j]) + 1, np.sum(1 - gapes_2500[i, 2, :, j]) + 1])
		kl_2500[i, j] = kl(conc, dil)
# The 500ms condition
for i in range(gapes_500.shape[0]):
	for j in range(gapes_500.shape[-1]):
		conc = np.array([np.sum(gapes_500[i, 3, :, j]) + 1, np.sum(1 - gapes_500[i, 3, :, j]) + 1])
		dil = np.array([np.sum(gapes_500[i, 2, :, j]) + 1, np.sum(1 - gapes_500[i, 2, :, j]) + 1])
		kl_500[i, j] = kl(conc, dil)
# The Jenn condition
for i in range(gapes_Jenn.shape[0]):
	for j in range(gapes_Jenn.shape[-1]):
		conc = np.array([np.sum(gapes_Jenn[i, 3, :, j]) + 1, np.sum(1 - gapes_Jenn[i, 3, :, j]) + 1])
		dil = np.array([np.sum(gapes_Jenn[i, 2, :, j]) + 1, np.sum(1 - gapes_Jenn[i, 2, :, j]) + 1])
		kl_Jenn[i, j] = kl(conc, dil)
# The before 700ms condition
for i in range(kl_before_700.shape[0]):
	conc = np.array([np.sum(gapes_before_700[3][:, i]) + 1, np.sum(1 - gapes_before_700[3][:, i]) + 1])
	dil = np.array([np.sum(gapes_before_700[2][:, i]) + 1, np.sum(1 - gapes_before_700[2][:, i]) + 1])
	kl_before_700[i] = kl(conc, dil)
# The after 700ms condition
for i in range(kl_after_700.shape[0]):
	conc = np.array([np.sum(gapes_after_700[3][:, i]) + 1, np.sum(1 - gapes_after_700[3][:, i]) + 1])
	dil = np.array([np.sum(gapes_after_700[2][:, i]) + 1, np.sum(1 - gapes_after_700[2][:, i]) + 1])
	kl_after_700[i] = kl(conc, dil)

# We fit two straight lines to the cumulative sum of the KL divergences in every laser condition. The mean onset of behavior/switchpoint is the point at which the model switches from one line to the other
# Inference via NUTS in every situation
# If our aim was to just estimate the parameters of the straight lines on two sides of the switch, we wouldn't use the DiscreteUniform switchpoint as here
# That discrete parameters is sampled by Metropolis instead of NUTS by definition
# If we could avoid sampling the switchpoint itself, we could do this: https://stackoverflow.com/questions/49144144/convert-numpy-function-to-theano/49152694#49152694

def logistic(x, x0, a, b):
	x0 = tt.tile(tt.shape_padright(x0), (1, tt.shape(x)[0]))
	x = tt.tile(tt.shape_padleft(x), (tt.shape(x0)[0], 1))
	a = tt.tile(tt.shape_padright(a), (1, tt.shape(x)[1]))
	return a/(1 + tt.exp(-b*(x - x0)))

data_2500 = (np.cumsum(kl_2500[:, 2000:4000], axis = 1) - np.mean(np.cumsum(kl_2500[:, 2000:4000], axis = 1)))/np.std(np.cumsum(kl_2500[:, 2000:4000], axis = 1))
with pm.Model() as model_2500:
	alpha = pm.Normal("alpha", mu = 0.0, sd = 3.0, shape = (kl_2500.shape[0], 2))
	beta = pm.Normal("beta", mu = 0.0, sd = 1.0, shape = (kl_2500.shape[0], 2))
	switchpoints = pm.Beta("switchpoints", 1, 1, shape = kl_2500.shape[0])
	sd = pm.HalfCauchy("sd", 0.5, shape = (kl_2500.shape[0], 2))
	intercept = tt.tile(alpha[:, 0][:, None], (1, 2000)) + logistic(np.linspace(0, 1, 2000), switchpoints, alpha[:, 1], 1000)
	slope = tt.tile(beta[:, 0][:, None], (1, 2000)) + logistic(np.linspace(0, 1, 2000), switchpoints, beta[:, 1], 1000)
	dev = tt.tile(sd[:, 0][:, None], (1, 2000)) + logistic(np.linspace(0, 1, 2000), switchpoints, sd[:, 1], 1000)
	regression = intercept + slope*np.linspace(0, 1, 2000)
	observed = pm.Normal("observed", mu = regression, sd = dev, observed = data_2500)

with model_2500:
	trace_2500 = pm.sample(tune = 10000, draws = 5000, njobs = 4)		

'''
# Run pymc3 model for the 2500ms condition
data_2500 = (np.cumsum(kl_2500[:, 1500:4000], axis = 1) - np.mean(np.cumsum(kl_2500[:, 1500:4000], axis = 1)))/np.std(np.cumsum(kl_2500[:, 1500:4000], axis = 1))
with pm.Model() as model_2500:
	alpha = pm.Normal("alpha", mu = 0.0, sd = 3.0, shape = (kl_2500.shape[0], 2))
	beta = pm.Normal("beta", mu = 0.0, sd = 1.0, shape = (kl_2500.shape[0], 2))
	switchpoints = pm.DiscreteUniform("switchpoints", lower = 800, upper = 2200, shape = kl_2500.shape[0])
	state = tt.switch(switchpoints >= np.repeat(np.arange(2500).reshape(2500, 1), kl_2500.shape[0], axis = 1), 0, 1)
	sd = pm.HalfCauchy("sd", 0.5, shape = (kl_2500.shape[0], 2))
	trials = np.vstack([np.ones(2500)*i for i in range(kl_2500.shape[0])]).astype("int")
	regression = alpha[trials, state.T] + beta[trials, state.T]*np.vstack([np.arange(2500) for i in range(kl_2500.shape[0])])
	observed = pm.Normal("observed", mu = regression, sd = sd[trials, state.T], observed = data_2500)
	
with model_2500:
	trace_2500 = pm.sample(tune = 6000, draws = 2000, njobs = 3)
'''

# Run pymc3 model for the 500ms condition
data_500 = (np.cumsum(kl_500[:, 2000:4000], axis = 1) - np.mean(np.cumsum(kl_500[:, 2000:4000], axis = 1)))/np.std(np.cumsum(kl_500[:, 2000:4000], axis = 1))
with pm.Model() as model_500:
	alpha = pm.Normal("alpha", mu = 0.0, sd = 3.0, shape = (kl_500.shape[0], 2))
	beta = pm.Normal("beta", mu = 0.0, sd = 1.0, shape = (kl_500.shape[0], 2))
	switchpoints = pm.Beta("switchpoints", 1, 1, shape = kl_500.shape[0])
	sd = pm.HalfCauchy("sd", 0.5, shape = (kl_500.shape[0], 2))
	intercept = tt.tile(alpha[:, 0][:, None], (1, 2000)) + logistic(np.linspace(0, 1, 2000), switchpoints, alpha[:, 1], 1000)
	slope = tt.tile(beta[:, 0][:, None], (1, 2000)) + logistic(np.linspace(0, 1, 2000), switchpoints, beta[:, 1], 1000)
	dev = tt.tile(sd[:, 0][:, None], (1, 2000)) + logistic(np.linspace(0, 1, 2000), switchpoints, sd[:, 1], 1000)
	regression = intercept + slope*np.linspace(0, 1, 2000)
	observed = pm.Normal("observed", mu = regression, sd = dev, observed = data_500)

with model_500:
	trace_500 = pm.sample(tune = 4000, draws = 2000, cores = 4)

'''
with pm.Model() as model_500:
	alpha = pm.Normal("alpha", mu = 0.0, sd = 3.0, shape = (kl_500.shape[0], 2))
	beta = pm.Normal("beta", mu = 0.0, sd = 1.0, shape = (kl_500.shape[0], 2))
	switchpoints = pm.DiscreteUniform("switchpoints", lower = 800, upper = 2200, shape = kl_500.shape[0])
	state = tt.switch(switchpoints >= np.repeat(np.arange(2500).reshape(2500, 1), kl_500.shape[0], axis = 1), 0, 1)
	sd = pm.HalfCauchy("sd", 0.5, shape = (kl_500.shape[0], 2))
	trials = np.vstack([np.ones(2500)*i for i in range(kl_500.shape[0])]).astype("int")
	regression = alpha[trials, state.T] + beta[trials, state.T]*np.vstack([np.arange(2500) for i in range(kl_500.shape[0])])
	observed = pm.Normal("observed", mu = regression, sd = sd[trials, state.T], observed = data_500)
	
with model_500:
	trace_500 = pm.sample(tune = 6000, draws = 2000, njobs = 3)

# Run pymc3 model for Jenn's data
data_Jenn = zscore(np.cumsum(kl_Jenn[0, 1500:4000]))
with pm.Model() as model_Jenn:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.DiscreteUniform("switchpoints", lower = 800, upper = 2200)
	state = tt.switch(switchpoints >= np.arange(2500), 0, 1)
	#sd = pm.Uniform("sd", lower = 0.1, upper = 10.0)
	sd = pm.HalfCauchy("sd", 0.5, shape = 2)	
	regression = alpha[state] + beta[state]*np.arange(2500)
	observed = pm.Normal("observed", mu = regression, sd = sd[state], observed = data_Jenn)
	
with model_Jenn:
	trace_Jenn = pm.sample(tune = 18000, draws = 4000, njobs = 3)
'''

# Run pymc3 model for the 700ms (palatability switchpoint) condition
data_before_700 = zscore(np.cumsum(kl_before_700[2000:4000]))
with pm.Model() as model_before_700:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.Beta("switchpoints", alpha =1, beta=1)
	sd = pm.HalfCauchy("sd", 0.5, shape = 2)
	intercept = alpha[0] + logistic(np.linspace(0, 1, 2000), switchpoints, alpha[1], 1000)
	slope = beta[0] + logistic(np.linspace(0, 1, 2000), switchpoints, beta[1], 1000)
	dev = sd[0] + logistic(np.linspace(0, 1, 2000), switchpoints, sd[1], 1000)
	regression = intercept + slope*np.linspace(0, 1, 2000)
	observed = pm.Normal("observed", mu = regression, sd = dev, observed = data_before_700)

with model_before_700:
	trace_before_700 = pm.sample(tune = 4000, draws = 2000, njobs = 4)

'''
with pm.Model() as model_before_700:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.DiscreteUniform("switchpoints", lower = 800, upper = 2200)
	state = tt.switch(switchpoints >= np.arange(2500), 0, 1)
	sd = pm.HalfCauchy("sd", 0.5, shape = 2)
	regression = alpha[state] + beta[state]*np.arange(2500)
	observed = pm.Normal("observed", mu = regression, sd = sd[state], observed = data_before_700)
	
with model_before_700:
	trace_before_700 = pm.sample(tune = 10000, draws = 2000, njobs = 3)
'''

data_after_700 = zscore(np.cumsum(kl_after_700[2500:4500]))
with pm.Model() as model_after_700:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.Beta("switchpoints", alpha =1, beta=1)
	sd = pm.HalfCauchy("sd", 0.5, shape = 2)
	intercept = alpha[0] + logistic(np.linspace(0, 1, 2000), switchpoints, alpha[1], 1000)
	slope = beta[0] + logistic(np.linspace(0, 1, 2000), switchpoints, beta[1], 1000)
	dev = sd[0] + logistic(np.linspace(0, 1, 2000), switchpoints, sd[1], 1000)
	regression = intercept + slope*np.linspace(0, 1, 2000)
	observed = pm.Normal("observed", mu = regression, sd = dev, observed = data_after_700)

with model_after_700:
	trace_after_700 = pm.sample(tune = 4000, draws = 2000, njobs = 4)

'''
with pm.Model() as model_after_700:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.DiscreteUniform("switchpoints", lower = 100, upper = 1800)
	state = tt.switch(switchpoints >= np.arange(2000), 0, 1)
	sd = pm.HalfCauchy("sd", 0.5, shape = 2)
	regression = alpha[state] + beta[state]*np.arange(2000)
	observed = pm.Normal("observed", mu = regression, sd = sd[state], observed = data_after_700)
	
with model_after_700:
	trace_after_700 = pm.sample(tune = 10000, draws = 2000, njobs = 3)
'''

# Save the traces and gelman_rubin statistics to file
# First the 2500ms condition
os.chdir("/media/patience/resorted_data/Plots/2500ms_EMG")
trace_2500_df = pm.backends.tracetab.trace_to_dataframe(trace_2500)
trace_2500_df.to_csv("trace_2500.csv")
np.save("trace_2500_switchpoints", trace_2500["switchpoints"])
np.save("trace_2500_alpha", trace_2500["alpha"])
np.save("trace_2500_beta", trace_2500["beta"])
with open("trace_2500_gelman_rubin.txt", "w") as f:
	print(pm.gelman_rubin(trace_2500), file = f)
np.save("kl_2500.npy", kl_2500)
# Then the 500ms condition
os.chdir("/media/patience/resorted_data/Plots/500ms_EMG")
trace_500_df = pm.backends.tracetab.trace_to_dataframe(trace_500)
trace_500_df.to_csv("trace_500.csv")
np.save("trace_500_switchpoints", trace_500["switchpoints"])
np.save("trace_500_alpha", trace_500["alpha"])
np.save("trace_500_beta", trace_500["beta"])
with open("trace_500_gelman_rubin.txt", "w") as f:
	print(pm.gelman_rubin(trace_500), file = f)
np.save("kl_500.npy", kl_500)
# Then the Jenn data condition
os.chdir("/media/patience/resorted_data/Jenn_Data/EMG_Plots")
trace_Jenn_df = pm.backends.tracetab.trace_to_dataframe(trace_Jenn)
trace_Jenn_df.to_csv("trace_Jenn.csv")
np.save("trace_Jenn_switchpoints", trace_Jenn["switchpoints"])
np.save("trace_Jenn_alpha", trace_Jenn["alpha"])
np.save("trace_Jenn_beta", trace_Jenn["beta"])
with open("trace_Jenn_gelman_rubin.txt", "w") as f:
	print(pm.gelman_rubin(trace_Jenn), file = f)
# Then the Before 700ms condition (palatability state comes before laser/700ms)
os.chdir("/media/patience/resorted_data/Plots/EM_switch")
trace_before_700_df = pm.backends.tracetab.trace_to_dataframe(trace_before_700)
trace_before_700_df.to_csv("trace_before_700.csv")
np.save("trace_before_700_switchpoints", trace_before_700["switchpoints"])
np.save("trace_before_700_alpha", trace_before_700["alpha"])
np.save("trace_before_700_beta", trace_before_700["beta"])
with open("trace_before_700_gelman_rubin.txt", "w") as f:
	print(pm.gelman_rubin(trace_before_700), file = f)
np.save("kl_before_700.npy", kl_before_700)
# Then the After 700ms condition (palatability state comes after laser/700ms)
os.chdir("/media/patience/resorted_data/Plots/EM_switch")
trace_after_700_df = pm.backends.tracetab.trace_to_dataframe(trace_after_700)
trace_after_700_df.to_csv("trace_after_700.csv")
np.save("trace_after_700_switchpoints", trace_after_700["switchpoints"])
np.save("trace_after_700_alpha", trace_after_700["alpha"])
np.save("trace_after_700_beta", trace_after_700["beta"])
with open("trace_after_700_gelman_rubin.txt", "w") as f:
	print(pm.gelman_rubin(trace_after_700), file = f)
np.save("kl_after_700.npy", kl_after_700)

# Now make another model for the before 700ms condition (palatability state comes after laser/700ms) by dropping the trials where the palatability transition happened too close to 700ms (>650ms).
# First load the palatability switchpoints array
os.chdir("/media/patience/resorted_data/Plots/EM_switch")
switch = np.load("Switchpoint2_Dur500_Lag700.npy")
# Then get the before laser trials for conc Quinine
before_trials_conc = np.where(switch[3] < 700)[0]
# And do the same for dil Quinine
before_trials_dil = np.where(switch[2] < 700)[0]
# Now calculate the kl divergence in gaping probability based only on the trials where palatability transitions happened <= 650ms
kl_before_650 = np.zeros(gapes_before_700[3].shape[1])
for i in range(kl_before_650.shape[0]):
	conc = np.array([np.sum(gapes_before_700[3][np.where(switch[3][before_trials_conc] <= 650)[0], i]) + 1, np.sum(1 - gapes_before_700[3][np.where(switch[3][before_trials_conc] <= 650)[0], i]) + 1])
	dil = np.array([np.sum(gapes_before_700[2][np.where(switch[2][before_trials_dil] <= 650)[0], i]) + 1, np.sum(1 - gapes_before_700[2][np.where(switch[2][before_trials_dil] <= 650)[0], i]) + 1])
	kl_before_650[i] = kl(conc, dil)
# Run the pymc3 model
data_before_650 = zscore(np.cumsum(kl_before_650[1500:3500]))
with pm.Model() as model_before_650:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.Beta("switchpoints", alpha = 1, beta = 1)
	sd = pm.HalfCauchy("sd", 0.5, shape = 2)
	intercept = alpha[0] + logistic(np.linspace(0, 1, 2000), switchpoints, alpha[1], 1000)
	slope = beta[0] + logistic(np.linspace(0, 1, 2000), switchpoints, beta[1], 1000)
	dev = sd[0] + logistic(np.linspace(0, 1, 2000), switchpoints, sd[1], 1000)
	regression = intercept + slope*np.linspace(0, 1, 2000)
	observed = pm.Normal("observed", mu = regression, sd = dev, observed = data_before_650)

with model_before_650:
	trace_before_650 = pm.sample(tune = 4000, draws = 2000, njobs = 4)

'''
with pm.Model() as model_before_650:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.DiscreteUniform("switchpoints", lower = 800, upper = 1900)
	state = tt.switch(switchpoints >= np.arange(2500), 0, 1)
	sd = pm.HalfCauchy("sd", 0.5, shape = 2)
	regression = alpha[state] + beta[state]*np.arange(2500)
	observed = pm.Normal("observed", mu = regression, sd = sd[state], observed = data_before_650)
	
with model_before_650:
	trace_before_650 = pm.sample(tune = 6000, draws = 2000, njobs = 3)
'''

# Save traces from this model
os.chdir("/media/patience/resorted_data/Plots/EM_switch")
trace_before_650_df = pm.backends.tracetab.trace_to_dataframe(trace_before_650)
trace_before_650_df.to_csv("trace_before_650.csv")
np.save("trace_before_650_switchpoints", trace_before_650["switchpoints"])
np.save("trace_before_650_alpha", trace_before_650["alpha"])
np.save("trace_before_650_beta", trace_before_650["beta"])
with open("trace_before_650_gelman_rubin.txt", "w") as f:
	print(pm.gelman_rubin(trace_before_650), file = f)
np.save("kl_before_650.npy", kl_before_650)
 
# Now make another model for the before 700ms condition (palatability state comes after laser/700ms) by dropping the trials where the palatability transition happened too close to 700ms (>680ms).
# First load the palatability switchpoints array
os.chdir("/media/patience/resorted_data/Plots/EM_switch")
switch = np.load("Switchpoint2_Dur500_Lag700.npy")
# Then get the before laser trials for conc Quinine
before_trials_conc = np.where(switch[3] < 700)[0]
# And do the same for dil Quinine
before_trials_dil = np.where(switch[2] < 700)[0]
# Now calculate the kl divergence in gaping probability based only on the trials where palatability transitions happened <= 680ms
kl_before_680 = np.zeros(gapes_before_700[3].shape[1])
for i in range(kl_before_680.shape[0]):
	conc = np.array([np.sum(gapes_before_700[3][np.where(switch[3][before_trials_conc] <= 680)[0], i]) + 1, np.sum(1 - gapes_before_700[3][np.where(switch[3][before_trials_conc] <= 680)[0], i]) + 1])
	dil = np.array([np.sum(gapes_before_700[2][np.where(switch[2][before_trials_dil] <= 680)[0], i]) + 1, np.sum(1 - gapes_before_700[2][np.where(switch[2][before_trials_dil] <= 680)[0], i]) + 1])
	kl_before_680[i] = kl(conc, dil)
# Run the pymc3 model
data_before_680 = zscore(np.cumsum(kl_before_680[1500:3500]))
with pm.Model() as model_before_680:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.Beta("switchpoints", alpha = 1, beta = 1)
	sd = pm.HalfCauchy("sd", 0.5, shape = 2)
	intercept = alpha[0] + logistic(np.linspace(0, 1, 2000), switchpoints, alpha[1], 1000)
	slope = beta[0] + logistic(np.linspace(0, 1, 2000), switchpoints, beta[1], 1000)
	dev = sd[0] + logistic(np.linspace(0, 1, 2000), switchpoints, sd[1], 1000)
	regression = intercept + slope*np.linspace(0, 1, 2000)
	observed = pm.Normal("observed", mu = regression, sd = dev, observed = data_before_680)

with model_before_680:
	trace_before_680 = pm.sample(tune = 4000, draws = 2000, njobs = 4)

'''
with pm.Model() as model_before_680:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.DiscreteUniform("switchpoints", lower = 800, upper = 1900)
	state = tt.switch(switchpoints >= np.arange(2000), 0, 1)
	sd = pm.HalfCauchy("sd", 0.5, shape = 2)
	regression = alpha[state] + beta[state]*np.arange(2000)
	observed = pm.Normal("observed", mu = regression, sd = sd[state], observed = data_before_680)
	
with model_before_680:
	trace_before_680 = pm.sample(tune = 10000, draws = 2000, njobs = 3)
'''

# Save traces from this model
os.chdir("/media/patience/resorted_data/Plots/EM_switch")
trace_before_680_df = pm.backends.tracetab.trace_to_dataframe(trace_before_680)
trace_before_680_df.to_csv("trace_before_680.csv")
np.save("trace_before_680_switchpoints", trace_before_680["switchpoints"])
np.save("trace_before_680_alpha", trace_before_680["alpha"])
np.save("trace_before_680_beta", trace_before_680["beta"])
with open("trace_before_680_gelman_rubin.txt", "w") as f:
	print(pm.gelman_rubin(trace_before_680), file = f)
np.save("kl_before_680.npy", kl_before_680)

# Neglect any control trials that have 4-6Hz activity before 700ms and find gaping latency
# Just to show that the 0-0.5s perturbations are not just inhibiting any gaping that happens early on in the trial
conc_trials = np.where(np.sum(gapes_500[0, 3, :, 2000:2700], axis = -1) <= 10)[0]
dil_trials = np.where(np.sum(gapes_500[0, 2, :, 2000:2700], axis = -1) <= 10)[0]
kl_neglect_700 = np.zeros(gapes_500.shape[-1])
for i in range(kl_neglect_700.shape[0]):
	conc = np.array([np.sum(gapes_500[0, 3, conc_trials, i]) + 1, np.sum(1 - gapes_500[0, 3, conc_trials, i]) + 1])
	dil = np.array([np.sum(gapes_500[0, 2, dil_trials, i]) + 1, np.sum(1 - gapes_500[0, 2, dil_trials, i]) + 1])
	kl_neglect_700[i] = kl(conc, dil)

# Run pymc3 model for the data with trials with gaping before 700ms neglected
data_neglect_700 = (np.cumsum(kl_neglect_700[2000:4000]) - np.mean(np.cumsum(kl_neglect_700[2000:4000])))/np.std(np.cumsum(kl_neglect_700[2000:4000]))
with pm.Model() as model_neglect_700:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.Beta("switchpoints", alpha = 1, beta = 1)
	sd = pm.HalfCauchy("sd", 0.5, shape = 2)
	intercept = alpha[0] + logistic(np.linspace(0, 1, 1750), switchpoints, alpha[1], 1000)
	slope = beta[0] + logistic(np.linspace(0, 1, 1750), switchpoints, beta[1], 1000)
	dev = sd[0] + logistic(np.linspace(0, 1, 1750), switchpoints, sd[1], 1000)
	regression = intercept + slope*np.linspace(0, 1, 1750)
	observed = pm.Normal("observed", mu = regression, sd = dev, observed = data_neglect_700[:1750])

with model_neglect_700:
	trace_neglect_700 = pm.sample(tune = 4000, draws = 2000, njobs = 4)

# Save this trace too
os.chdir("/media/patience/resorted_data/Plots/500ms_EMG")
trace_neglect_700_df = pm.backends.tracetab.trace_to_dataframe(trace_neglect_700)
trace_neglect_700_df.to_csv("trace_neglect_700.csv")
np.save("trace_neglect_700_switchpoints", trace_neglect_700["switchpoints"])
np.save("trace_neglect_700_alpha", trace_neglect_700["alpha"])
np.save("trace_neglect_700_beta", trace_neglect_700["beta"])
with open("trace_neglect_700_gelman_rubin.txt", "w") as f:
	print(pm.gelman_rubin(trace_neglect_700), file = f)
np.save("kl_neglect_700.npy", kl_neglect_700)

# Neglect any control trials that have 4-6Hz activity before 1200ms and find gaping latency
# Just to show that the 0.7-1.2s perturbations are not just truncating the distribution of gape times
conc_trials = np.where((np.sum(gapes_500[0, 3, :, 2000:3200], axis = -1) <= 10)*(np.sum(gapes_500[0, 3, :, 3200:4000], axis = -1) >= 100))[0]
dil_trials = np.where((np.sum(gapes_500[0, 2, :, 2000:3200], axis = -1) <= 10)*(np.sum(gapes_500[0, 2, :, 3200:4000], axis = -1) >= 100))[0]
kl_neglect_1200 = np.zeros(gapes_500.shape[-1])
for i in range(kl_neglect_1200.shape[0]):
	conc = np.array([np.sum(gapes_500[0, 3, conc_trials, i]) + 1, np.sum(1 - gapes_500[0, 3, conc_trials, i]) + 1])
	dil = np.array([np.sum(gapes_500[0, 2, dil_trials, i]) + 1, np.sum(1 - gapes_500[0, 2, dil_trials, i]) + 1])
	kl_neglect_1200[i] = kl(conc, dil)

# Run pymc3 model for the data with trials with gaping before 1200ms neglected
data_neglect_1200 = (np.cumsum(kl_neglect_1200[2000:4000]) - np.mean(np.cumsum(kl_neglect_1200[2000:4000])))/np.std(np.cumsum(kl_neglect_1200[2000:4000]))
with pm.Model() as model_neglect_1200:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.Beta("switchpoints", alpha = 1, beta = 1)
	sd = pm.HalfCauchy("sd", 0.5, shape = 2)
	intercept = alpha[0] + logistic(np.linspace(0, 1, 1750), switchpoints, alpha[1], 1000)
	slope = beta[0] + logistic(np.linspace(0, 1, 1750), switchpoints, beta[1], 1000)
	dev = sd[0] + logistic(np.linspace(0, 1, 1750), switchpoints, sd[1], 1000)
	regression = intercept + slope*np.linspace(0, 1, 1750)
	observed = pm.Normal("observed", mu = regression, sd = dev, observed = data_neglect_1200[:1750])

with model_neglect_1200:
	trace_neglect_1200 = pm.sample(tune = 4000, draws = 2000, njobs = 4)

# Run pymc3 model for the 1.4-1.9s trials, with only the trials that have switchpoints before 1.4s
conc_trials = np.where(late_switchpoints_expanded[3, :] < 1400.0)[0]
dil_trials = np.where(late_switchpoints_expanded[2, :] < 1400.0)[0]

for i in range(kl_late_expanded.shape[0]):
	conc = np.array([np.sum(gapes_500[3, 3, conc_trials, i]) + 1, np.sum(1 - gapes_500[3, 3, conc_trials, i]) + 1])
	dil = np.array([np.sum(gapes_500[3, 2, dil_trials, i]) + 1, np.sum(1 - gapes_500[3, 2, dil_trials, i]) + 1])
	kl_late_expanded[i] = kl(conc, dil)

data_late_expanded = zscore(np.cumsum(kl_late_expanded[2000:4000]))
with pm.Model() as model_late_expanded:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.Beta("switchpoints", alpha = 1, beta = 1)
	sd = pm.HalfCauchy("sd", 0.5, shape = 2)
	intercept = alpha[0] + logistic(np.linspace(0, 1, 2000), switchpoints, alpha[1], 1000)
	slope = beta[0] + logistic(np.linspace(0, 1, 2000), switchpoints, beta[1], 1000)
	dev = sd[0] + logistic(np.linspace(0, 1, 2000), switchpoints, sd[1], 1000)
	regression = intercept + slope*np.linspace(0, 1, 2000)
	observed = pm.Normal("observed", mu = regression, sd = dev, observed = data_late_expanded)

with model_late_expanded:
	trace_late_expanded  = pm.sample(tune = 6000, draws = 2000, cores = 4)

# Save the trace
os.chdir("/media/patience/resorted_data/Plots/500ms_EMG")
trace_late_expanded_df = pm.backends.tracetab.trace_to_dataframe(trace_late_expanded)
trace_late_expanded_df.to_csv("trace_late_expanded.csv")
np.save("trace_late_expanded_switchpoints", trace_late_expanded["switchpoints"])
with open("trace_late_expanded_gelman_rubin.txt", "w") as f:
	print(pm.gelman_rubin(trace_late_expanded), file = f)
np.save("kl_late_expanded.npy", kl_late_expanded)

# Run pymc3 model for the 1.4-1.9s trials, with only the trials that have switchpoints AFTER 1.4s (laser onset)
# Very few (about 10%) trials fall in this condition
conc_trials = np.where(late_switchpoints_expanded[3, :] >= 1400.0)[0]
dil_trials = np.where(late_switchpoints_expanded[2, :] >= 1400.0)[0]

for i in range(kl_late_expanded_after_laser.shape[0]):
	conc = np.array([np.sum(gapes_500[3, 3, conc_trials, i]) + 1, np.sum(1 - gapes_500[3, 3, conc_trials, i]) + 1])
	dil = np.array([np.sum(gapes_500[3, 2, dil_trials, i]) + 1, np.sum(1 - gapes_500[3, 2, dil_trials, i]) + 1])
	kl_late_expanded_after_laser[i] = kl(conc, dil)

data_late_expanded_after_laser = zscore(np.cumsum(kl_late_expanded_after_laser[2000:4000]))
with pm.Model() as model_late_expanded_after_laser:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.Beta("switchpoints", alpha = 1, beta = 1)
	sd = pm.HalfCauchy("sd", 0.5, shape = 2)
	intercept = alpha[0] + logistic(np.linspace(0, 1, 2000), switchpoints, alpha[1], 1000)
	slope = beta[0] + logistic(np.linspace(0, 1, 2000), switchpoints, beta[1], 1000)
	dev = sd[0] + logistic(np.linspace(0, 1, 2000), switchpoints, sd[1], 1000)
	regression = intercept + slope*np.linspace(0, 1, 2000)
	observed = pm.Normal("observed", mu = regression, sd = dev, observed = data_late_expanded_after_laser)

with model_late_expanded_after_laser:
	trace_late_expanded_after_laser  = pm.sample(tune = 8000, draws = 2000, cores = 4)

# Save the trace
os.chdir("/media/patience/resorted_data/Plots/500ms_EMG")
trace_late_expanded_after_laser_df = pm.backends.tracetab.trace_to_dataframe(trace_late_expanded_after_laser)
trace_late_expanded_after_laser_df.to_csv("trace_late_expanded_after_laser.csv")
np.save("trace_late_expanded_after_laser_switchpoints", trace_late_expanded_after_laser["switchpoints"])
with open("trace_late_expanded_after_laser_gelman_rubin.txt", "w") as f:
	print(pm.gelman_rubin(trace_late_expanded_after_laser), file = f)
np.save("kl_late_expanded_after_laser.npy", kl_late_expanded_after_laser)
