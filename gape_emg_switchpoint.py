import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import theano.tensor as tt
import os
import pandas as pd

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

# Get the mean difference in gaping probability between conc and dil Qui
diff_2500 = np.mean(gapes_2500[:, 3, :, 2000:4000], axis = 1) - np.mean(gapes_2500[:, 2, :, 2000:4000], axis = 1)
diff_500 = np.mean(gapes_500[:, 3, :, 2000:4000], axis = 1) - np.mean(gapes_500[:, 2, :, 2000:4000], axis = 1)
diff_Jenn = np.mean(gapes_Jenn[:, 3, :, 2000:4000], axis = 1) - np.mean(gapes_Jenn[:, 2, :, 2000:4000], axis = 1)
diff_before_700 = np.mean(gapes_before_700[3][:, :2000], axis = 0) - np.mean(gapes_before_700[2][:, :2000], axis = 0)
diff_after_700 = np.mean(gapes_after_700[3][:, :2000], axis = 0) - np.mean(gapes_after_700[2][:, :2000], axis = 0)

# We fit two straight lines to the cumulative sum of these differences in every laser condition. The start of behavior/switchpoint is the point at which the model switches from one line to the other
# Inference via NUTS in every situation

# Run pymc3 model for the 2500ms condition
with pm.Model() as model_2500:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = (diff_2500.shape[0], 2))
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = (diff_2500.shape[0], 2))
	switchpoints = pm.DiscreteUniform("switchpoints", lower = 250, upper = 1400, shape = diff_2500.shape[0])
	state = tt.switch(switchpoints >= np.repeat(np.arange(1750).reshape(1750, 1), diff_2500.shape[0], axis = 1), 0, 1)
	sd = pm.HalfCauchy("sd", 0.5, shape = diff_2500.shape[0])
	trials = np.vstack([np.ones(1750)*i for i in range(diff_2500.shape[0])]).astype("int")
	regression = alpha[trials, state.T] + beta[trials, state.T]*np.vstack([np.arange(1750) for i in range(diff_2500.shape[0])])
	observed = pm.Normal("observed", mu = regression, sd = sd[trials], observed = np.cumsum(diff_2500[:, 250:], axis = 1))
	
with model_2500:
	trace_2500 = pm.sample(tune = 3000, draws = 1000, njobs = 3)

# Run pymc3 model for the 500ms condition
with pm.Model() as model_500:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = (diff_500.shape[0], 2))
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = (diff_500.shape[0], 2))
	switchpoints = pm.DiscreteUniform("switchpoints", lower = 250, upper = 1400, shape = diff_500.shape[0])
	state = tt.switch(switchpoints >= np.repeat(np.arange(1750).reshape(1750, 1), diff_500.shape[0], axis = 1), 0, 1)
	sd = pm.HalfCauchy("sd", 0.5, shape = diff_500.shape[0])
	trials = np.vstack([np.ones(1750)*i for i in range(diff_500.shape[0])]).astype("int")
	regression = alpha[trials, state.T] + beta[trials, state.T]*np.vstack([np.arange(1750) for i in range(diff_500.shape[0])])
	observed = pm.Normal("observed", mu = regression, sd = sd[trials], observed = np.cumsum(diff_500[:, 250:], axis = 1))
	
with model_500:
	trace_500 = pm.sample(tune = 3000, draws = 1000, njobs = 3)	

# Run pymc3 model for Jenn's data
with pm.Model() as model_Jenn:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.DiscreteUniform("switchpoints", lower = 250, upper = 1400)
	state = tt.switch(switchpoints >= np.arange(1750), 0, 1)
	sd = pm.Uniform("sd", lower = 0.1, upper = 10.0)
	regression = alpha[state] + beta[state]*np.arange(1750)
	observed = pm.Normal("observed", mu = regression, sd = sd, observed = np.cumsum(diff_Jenn[0, 250:]))
	
with model_Jenn:
	trace_Jenn = pm.sample(tune = 3000, draws = 1000, njobs = 3)

# Run pymc3 model for the 700ms (palatability switchpoint) condition
with pm.Model() as model_before_700:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.DiscreteUniform("switchpoints", lower = 250, upper = 1400)
	state = tt.switch(switchpoints >= np.arange(1750), 0, 1)
	sd = pm.HalfCauchy("sd", 0.5)
	regression = alpha[state] + beta[state]*np.arange(1750)
	observed = pm.Normal("observed", mu = regression, sd = sd, observed = np.cumsum(diff_before_700[250:]))
	
with model_before_700:
	trace_before_700 = pm.sample(tune = 6000, draws = 2000, njobs = 3)

with pm.Model() as model_after_700:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.DiscreteUniform("switchpoints", lower = 250, upper = 1400)
	state = tt.switch(switchpoints >= np.arange(1750), 0, 1)
	#sd = pm.Uniform("sd", lower = 0.1, upper = 2.5)
	sd = pm.HalfCauchy("sd", 0.5)
	regression = alpha[state] + beta[state]*np.arange(1750)
	observed = pm.Normal("observed", mu = regression, sd = sd, observed = np.cumsum(diff_after_700[250:]))
	
with model_after_700:
	trace_after_700 = pm.sample(tune = 6000, draws = 2000, njobs = 3)

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
# Then the 500ms condition
os.chdir("/media/patience/resorted_data/Plots/500ms_EMG")
trace_500_df = pm.backends.tracetab.trace_to_dataframe(trace_500)
trace_500_df.to_csv("trace_500.csv")
np.save("trace_500_switchpoints", trace_500["switchpoints"])
np.save("trace_500_alpha", trace_500["alpha"])
np.save("trace_500_beta", trace_500["beta"])
with open("trace_500_gelman_rubin.txt", "w") as f:
	print(pm.gelman_rubin(trace_500), file = f)
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
# Then the After 700ms condition (palatability state comes after laser/700ms)
os.chdir("/media/patience/resorted_data/Plots/EM_switch")
trace_after_700_df = pm.backends.tracetab.trace_to_dataframe(trace_after_700)
trace_after_700_df.to_csv("trace_after_700.csv")
np.save("trace_after_700_switchpoints", trace_after_700["switchpoints"])
np.save("trace_after_700_alpha", trace_after_700["alpha"])
np.save("trace_after_700_beta", trace_after_700["beta"])
with open("trace_after_700_gelman_rubin.txt", "w") as f:
	print(pm.gelman_rubin(trace_after_700), file = f)

# Now make another model for the before 700ms condition (palatability state comes after laser/700ms) by dropping the trials where the palatability transition happened too close to 700ms (>650ms).
# First load the palatability switchpoints array
switch = np.load("Switchpoint2_Dur500_Lag700.npy")
# Then get the before laser trials for conc Quinine
before_trials_conc = np.where(switch[3] < 700)[0]
# And do the same for dil Quinine
before_trials_dil = np.where(switch[2] < 700)[0]
# Now calculate the difference in gaping probability based only on the trials where palatability transitions happened <= 650ms
diff_before_650 = np.mean(gapes_before_700[3][np.where(switch[3][before_trials_conc] <= 650)[0], :2000], axis = 0) - np.mean(gapes_before_700[2][np.where(switch[2][before_trials_dil] <= 650)[0], :2000], axis = 0)
# For the pymc3 model, use all 2000ms of data as the switchpoint is earlier in this condition
with pm.Model() as model_before_650:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.DiscreteUniform("switchpoints", lower = 500, upper = 1500)
	state = tt.switch(switchpoints >= np.arange(2000), 0, 1)
	#sd = pm.HalfCauchy("sd", 0.5)
	sd = pm.Uniform("sd", lower = 0.1, upper = 10.0)
	regression = alpha[state] + beta[state]*np.arange(2000)
	observed = pm.Normal("observed", mu = regression, sd = sd, observed = np.cumsum(diff_before_650))
	
with model_before_650:
	trace_before_650 = pm.sample(tune = 6000, draws = 2000, njobs = 3)

# Save traces from this model
os.chdir("/media/patience/resorted_data/Plots/EM_switch")
trace_before_650_df = pm.backends.tracetab.trace_to_dataframe(trace_before_650)
trace_before_650_df.to_csv("trace_before_650.csv")
np.save("trace_before_650_switchpoints", trace_before_650["switchpoints"])
np.save("trace_before_650_alpha", trace_before_650["alpha"])
np.save("trace_before_650_beta", trace_before_650["beta"])
with open("trace_before_650_gelman_rubin.txt", "w") as f:
	print(pm.gelman_rubin(trace_before_650), file = f)
 
# Now make another model for the before 700ms condition (palatability state comes after laser/700ms) by dropping the trials where the palatability transition happened too close to 700ms (>680ms).
# First load the palatability switchpoints array
switch = np.load("Switchpoint2_Dur500_Lag700.npy")
# Then get the before laser trials for conc Quinine
before_trials_conc = np.where(switch[3] < 700)[0]
# And do the same for dil Quinine
before_trials_dil = np.where(switch[2] < 700)[0]
# Now calculate the difference in gaping probability based only on the trials where palatability transitions happened <= 650ms
diff_before_680 = np.mean(gapes_before_700[3][np.where(switch[3][before_trials_conc] <= 680)[0], :2000], axis = 0) - np.mean(gapes_before_700[2][np.where(switch[2][before_trials_dil] <= 680)[0], :2000], axis = 0)
# For the pymc3 model, use all 2000ms of data as the switchpoint is earlier in this condition
with pm.Model() as model_before_680:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.DiscreteUniform("switchpoints", lower = 250, upper = 1400)
	state = tt.switch(switchpoints >= np.arange(1750), 0, 1)
	sd = pm.HalfCauchy("sd", 0.5)
	#sd = pm.Uniform("sd", lower = 0.1, upper = 10.0)
	regression = alpha[state] + beta[state]*np.arange(1750)
	observed = pm.Normal("observed", mu = regression, sd = sd, observed = np.cumsum(diff_before_680[250:]))
	
with model_before_680:
	trace_before_680 = pm.sample(tune = 6000, draws = 2000, njobs = 3)

# Save traces from this model
os.chdir("/media/patience/resorted_data/Plots/EM_switch")
trace_before_680_df = pm.backends.tracetab.trace_to_dataframe(trace_before_680)
trace_before_680_df.to_csv("trace_before_680.csv")
np.save("trace_before_680_switchpoints", trace_before_680["switchpoints"])
np.save("trace_before_680_alpha", trace_before_680["alpha"])
np.save("trace_before_680_beta", trace_before_680["beta"])
with open("trace_before_680_gelman_rubin.txt", "w") as f:
	print(pm.gelman_rubin(trace_before_680), file = f)

'''
with pm.Model() as model_2500_control:
	alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
	beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
	switchpoints = pm.DiscreteUniform("switchpoints", lower = 250, upper = 1400)
	state = tt.switch(switchpoints >= np.arange(1750), 0, 1)
	sd = pm.HalfCauchy("sd", 0.5)
	regression = alpha[state] + beta[state]*np.arange(1600)
	observed = pm.Normal("observed", mu = regression, sd = sd, observed = np.cumsum(diff_2500[0, 250:]))

with model_2500_control:
	tr_2500_control = pm.sample(tune = 3000, draws = 1000, njobs = 3)

In [70]: with pm.Model() as model3:
    ...:     alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
    ...:     beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
    ...:     

In [71]: with model3:
    ...:     switchpoint = pm.DiscreteUniform("switchpoint", lower = 100, upper 
    ...: = 1100)
    ...:     #state = tt.switch(switchpoint <= np.arange(1200), 0, 1)
    ...:     # Above line messes up the ordering of 0s and 1s - the state array has 1s first and 0s afterwards
	     # Instead use
	     state = tt.switch(np.arange(1200) <= switchpoint, 0, 1)

In [72]: with model3:
    ...:     sd = pm.HalfCauchy("sd", 1.0)
    ...:     regression = alpha[state] + beta[state]*np.arange(1200)
    ...:     

In [73]: with model3:
    ...:     observed = pm.Normal("observed", mu = regression, sd = sd, observed
    ...:  = np.cumsum(diff[3, 400:1600]))
    ...:     

In [74]: with model3:
    ...:     tr3 = pm.sample(tune=2000, njobs=2, samples=200)
    ...:     
Assigned NUTS to alpha
Assigned NUTS to beta
Assigned Metropolis to switchpoint
Assigned NUTS to sd_log__
 98%|██████████████████████████████████████▏| 2446/2500 [10:23<00:20,  2.67it/s]/home/narendra/anaconda3/lib/python3.6/site-packages/pymc3/step_methods/hmc/nuts.py:448: UserWarning: Chain 1 reached the maximum tree depth. Increase max_treedepth, increase target_accept or reparameterize.
  'reparameterize.' % self._chain_id)
100%|███████████████████████████████████████| 2500/2500 [10:39<00:00,  3.05it/s]/home/narendra/anaconda3/lib/python3.6/site-packages/pymc3/step_methods/hmc/nuts.py:448: UserWarning: Chain 0 reached the maximum tree depth. Increase max_treedepth, increase target_accept or reparameterize.
  'reparameterize.' % self._chain_id)

In [76]: pm.hpd(tr3['switchpoint'])
Out[76]: array([515, 549])

In [77]: pm.hpd(tr1['switchpoint'])
Out[77]: array([746, 750])

In [78]: pm.hpd(tr['switchpoint'])
Out[78]: array([468, 486])

In [79]: plt.plot(np.cumsum(diff[0]))
Out[79]: [<matplotlib.lines.Line2D at 0x7feb1dc29fd0>]

In [80]: plt.plot(np.cumsum(diff[0, 400:1600]))
Out[80]: [<matplotlib.lines.Line2D at 0x7feb1da77fd0>]

In [81]: plt.plot(np.cumsum(diff[3, 400:1600]))
Out[81]: [<matplotlib.lines.Line2D at 0x7feb1da33978>]

In [82]: pm.hpd(tr3['beta'][:, 0] - tr3['beta'][:, 1])
Out[82]: array([ 0.11984684,  0.12597782])

In [83]: pm.hpd(tr2['beta'][:, 0] - tr2['beta'][:, 1])
Out[83]: array([ 0.08265796,  0.08469996])

In [84]: pm.hpd(tr1['beta'][:, 0] - tr1['beta'][:, 1])
Out[84]: array([ 0.15463853,  0.15554819])

In [85]: pm.hpd(tr['beta'][:, 0] - tr['beta'][:, 1])
Out[85]: array([ 0.13173354,  0.1343013 ])

In [86]: type(tr)
Out[86]: pymc3.backends.base.MultiTrace

In [87]: a = pm.backends.tracetab.trace_to_dataframe(tr)

In [90]: import pandas as pd

In [91]: a.to_csv("trace_control.csv")

In [92]: a = pm.backends.tracetab.trace_to_dataframe(tr1)

In [93]: a.to_csv("trace_early.csv")

In [94]: a = pm.backends.tracetab.trace_to_dataframe(tr2)

In [95]: a.to_csv("trace_middle.csv")

In [96]: a = pm.backends.tracetab.trace_to_dataframe(tr3)

In [97]: a.to_csv("trace_late.csv")

# With Jenn's data
In [128]: with pm.Model() as model:
     ...:     alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
     ...:     beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
     ...:     switchpoint = pm.DiscreteUniform("switchpoint", lower = 100, upper
     ...:  = 1100)
     ...:     state = tt.switch(np.arange(1200) <= switchpoint, 0, 1)
     ...:     

In [129]: with model:
     ...:     sd = pm.HalfCauchy("sd", 0.5)
     ...:     regression = alpha[state] + beta[state]*np.arange(1200)
     ...:     observed = pm.Normal("observed", mu = regression, sd = sd, observe
     ...: d = np.cumsum(diff[400:1600]))
     ...:     

In [130]: with model:
     ...:     inference = pm.fit(n = 200000, method = "fullrank_advi")
     ...:     trace = inference.sample(10000)
     ...:     
Average Loss = -inf: 100%|████████████| 200000/200000 [01:56<00:00, 1721.27it/s]
Finished [100%]: Average Loss = 5,454.7

In [153]: with pm.Model() as model:
     ...:     alpha = pm.Normal("alpha", mu = 0, sd = 3.0, shape = 2)
     ...:     beta = pm.Normal("beta", mu = 0, sd = 1.0, shape = 2)
     ...:     switchpoint = pm.DiscreteUniform("switchpoint", lower = 100, upper
     ...:  = 1300)
     ...:     state = tt.switch(np.arange(1600) <= switchpoint, 0, 1)
     ...:     

In [154]: with model:
     ...:     sd = pm.HalfCauchy("sd", 0.5)
     ...:     regression = alpha[state] + beta[state]*np.arange(1600)
     ...:     observed = pm.Normal("observed", mu = regression, sd = sd, observe
     ...: d = np.cumsum(diff_2500[0, 400:2000]))
     ...:     

In [155]: with model:
     ...:     inference = pm.fit(n = 200000, method = "fullrank_advi")
     ...:     trace_2500_control = inference.sample(10000)
     ...:     
Average Loss = -inf: 100%|████████████| 200000/200000 [01:59<00:00, 1674.48it/s]
Finished [100%]: Average Loss = 4,558.3

'''
