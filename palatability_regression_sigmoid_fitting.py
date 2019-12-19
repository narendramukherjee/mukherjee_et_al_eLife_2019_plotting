# Fits sigmoid curves to the mean palatability regression coefficients in order to calculate the peak time of single neuron palatability firing
# Sigmoid curves have the form: f(x) = L/(1 + exp(-k(x - x0))) + delta where
# L = Maximum height achieved by the sigmoid, k = steepness, x0 = inflection point and delta = Minimum height achieved by the sigmoid
# We fit with delta = 0 as we've already scaled the regression coefficients by their value in the pre-stimulus window

import numpy as np
import pymc3 as pm
import theano.tensor as tt

# Load up the regression coefficients from both the 500ms and 2500ms conditions
# 2500ms
os.chdir("/media/patience/resorted_data/Plots/2500ms_allneurons/palatability_regression_include_baseline")
regress_2500 = np.load("palatability_regression_trace.npy")
# Scale the regression coefficients by subtracting out the mean of the 1st 10 coefficients (250ms of pre-stimulus data)
scaled_regress_2500 = regress_2500 - np.tile(np.mean(regress_2500[:, :10, :], axis = 1).reshape((-1, 1, 2)), (1, 71, 1))
# 500ms
os.chdir("/media/patience/resorted_data/Plots/500ms_allneurons/palatability_regression_include_baseline")
regress_500 = np.load("palatability_regression_trace.npy")
# Scale the regression coefficients by subtracting out the mean of the 1st 10 coefficients (250ms of pre-stimulus data)
scaled_regress_500 = regress_500 - np.tile(np.mean(regress_500[:, :10, :], axis = 1).reshape((-1, 1, 4)), (1, 71, 1))

# Make a pymc3 model for the 2500ms condition
# Priors: L-> around 0, k-> around 1, x0-> around 35 (midway through taste response) and delta = 0
with pm.Model() as model_2500:
	BoundedNormal = pm.Bound(pm.Normal, lower = 0.0)
	L = BoundedNormal("L", mu = 0.0, sd = 0.1, shape = 2)
	k = BoundedNormal("k", mu = 1.0, sd = 1.0, shape = 2)
	x0 = BoundedNormal("x0", mu = 35.0, sd = 3.0, shape = 2)
	#delta = pm.Normal("delta", mu = 0.0, sd = 0.1, shape = 2)
	sd = pm.HalfCauchy("sd", 0.5, shape = 2)
	mean = L[None, :]/(1.0 + tt.exp(-k[None, :]*(np.arange(71)[:, None] - x0[None, :]))) #+ delta[None, :]
	obs = pm.Normal("obs", mu = mean, sd = tt.tile(sd[None, :], (71, 1)), observed = np.mean(scaled_regress_2500, axis = 0))
with model_2500:
	tr_2500 = pm.sample(tune = 6000, draws = 2000, njobs = 4)	

# Make a pymc3 model for the 500ms condition
# Priors: L-> around 0, k-> around 1, x0-> around 35 (midway through taste response) and delta = 0
with pm.Model() as model_500:
	BoundedNormal = pm.Bound(pm.Normal, lower = 0.0)
	L = BoundedNormal("L", mu = 0.0, sd = 0.1, shape = 4)
	k = BoundedNormal("k", mu = 1.0, sd = 1.0, shape = 4)
	x0 = BoundedNormal("x0", mu = 35.0, sd = 3.0, shape = 4)
	#delta = pm.Normal("delta", mu = 0.0, sd = 0.01, shape = 4)
	sd = pm.HalfCauchy("sd", 0.5, shape = 4)
	mean = L[None, :]/(1.0 + tt.exp(-k[None, :]*(np.arange(71)[:, None] - x0[None, :]))) #+ delta[None, :]
	obs = pm.Normal("obs", mu = mean, sd = tt.tile(sd[None, :], (71, 1)), observed = np.mean(scaled_regress_500, axis = 0))
with model_500:
#	tr_500 = pm.sample(tune = 6000, draws = 2000, njobs = 4, nuts_kwargs=dict(target_accept=.95))
	tr_500 = pm.sample(tune = 6000, draws = 2000, njobs = 4)

# Save the traces and the Gelman-Rubin statistics to file
# 2500ms
os.chdir("/media/patience/resorted_data/Plots/2500ms_allneurons/palatability_regression_include_baseline")
f = open("sigmoid_fit_convergence.txt", "w")
print(str(pm.gelman_rubin(tr_2500)), file = f)
f.close()
np.save("sigmoid_fit_trace.npy", tr_2500)
pm.trace_to_dataframe(tr_2500).to_csv("sigmoid_fit_trace.csv")

# 500ms
os.chdir("/media/patience/resorted_data/Plots/500ms_allneurons/palatability_regression_include_baseline")
f = open("sigmoid_fit_convergence.txt", "w")
print(str(pm.gelman_rubin(tr_500)), file = f)
f.close()
np.save("sigmoid_fit_trace.npy", tr_500)
pm.trace_to_dataframe(tr_500).to_csv("sigmoid_fit_trace.csv")



