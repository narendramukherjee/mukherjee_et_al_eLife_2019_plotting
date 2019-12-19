import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
sns.set(style="white", context="talk", font_scale=3.5)
sns.set_color_codes(palette = 'colorblind')
sns.set_style("ticks", {"axes.linewidth": 2.0})
plt.ion()

import os
import numpy as np
import pandas as pd
import tables
import pymc3 as pm
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import zscore 

os.chdir("/media/patience/resorted_data/Plots/2500ms_allneurons/palatability_regression_include_baseline")
regress_2500 = np.load("palatability_regression_trace.npy")
# Scale the regression coefficients by subtracting out the mean of the 1st 10 coefficients (250ms of pre-stimulus data)
scaled_regress_2500 = regress_2500 - np.tile(np.mean(regress_2500[:, :10, :], axis = 1).reshape((-1, 1, 2)), (1, 71, 1))
sigmoid_fit_2500 = pd.read_csv("sigmoid_fit_trace.csv")
# Calculate the mean sigmoid regression line from the sigmoid fit trace
sigmoid_regress_2500 = np.zeros((71, 2))
sigmoid_regress_2500[:, 0] = np.mean(sigmoid_fit_2500["L__0"][:, None]/(1 + np.exp(-sigmoid_fit_2500["k__0"][:, None]*(np.arange(71)[None, :] - sigmoid_fit_2500["x0__0"][:, None]))), axis = 0)
sigmoid_regress_2500[:, 1] = np.mean(sigmoid_fit_2500["L__1"][:, None]/(1 + np.exp(-sigmoid_fit_2500["k__1"][:, None]*(np.arange(71)[None, :] - sigmoid_fit_2500["x0__1"][:, None]))), axis = 0)

os.chdir("/media/patience/resorted_data/Plots/500ms_allneurons/palatability_regression_include_baseline")
regress_500 = np.load("palatability_regression_trace.npy")
# Scale the regression coefficients by subtracting out the mean of the 1st 10 coefficients (250ms of pre-stimulus data)
scaled_regress_500 = regress_500 - np.tile(np.mean(regress_500[:, :10, :], axis = 1).reshape((-1, 1, 4)), (1, 71, 1))
sigmoid_fit_500 = pd.read_csv("sigmoid_fit_trace.csv")
# Calculate the mean sigmoid regression line from the sigmoid fit trace
sigmoid_regress_500 = np.zeros((71, 4))
sigmoid_regress_500[:, 0] = np.mean(sigmoid_fit_500["L__0"][:, None]/(1 + np.exp(-sigmoid_fit_500["k__0"][:, None]*(np.arange(71)[None, :] - sigmoid_fit_500["x0__0"][:, None]))), axis = 0)
sigmoid_regress_500[:, 1] = np.mean(sigmoid_fit_500["L__1"][:, None]/(1 + np.exp(-sigmoid_fit_500["k__1"][:, None]*(np.arange(71)[None, :] - sigmoid_fit_500["x0__1"][:, None]))), axis = 0)
sigmoid_regress_500[:, 2] = np.mean(sigmoid_fit_500["L__2"][:, None]/(1 + np.exp(-sigmoid_fit_500["k__2"][:, None]*(np.arange(71)[None, :] - sigmoid_fit_500["x0__2"][:, None]))), axis = 0)
sigmoid_regress_500[:, 3] = np.mean(sigmoid_fit_500["L__3"][:, None]/(1 + np.exp(-sigmoid_fit_500["k__3"][:, None]*(np.arange(71)[None, :] - sigmoid_fit_500["x0__3"][:, None]))), axis = 0)

os.chdir("/media/patience/resorted_data/Plots/500ms_allneurons/palatability_regression_include_baseline/palatability_regression_switch_EM")
regress_switch_EM = np.load("palatability_regression_trace.npy")
# Scale the regression coefficients by subtracting out the mean of the 1st 10 coefficients (250ms of pre-stimulus data)
scaled_regress_switch_EM = regress_switch_EM - np.tile(np.mean(regress_switch_EM[:, :10, :], axis = 1).reshape((-1, 1, 2)), (1, 71, 1))

os.chdir("/media/patience/resorted_data/Plots/2500ms_EMG")
gapes_2500 = np.load("gapes.npy")
trace_2500_df = pd.read_csv("trace_2500.csv")
kl_2500 = np.load("kl_2500.npy")

os.chdir("/media/patience/resorted_data/Plots/500ms_EMG")
gapes_500 = np.load("gapes.npy")
gapes_Li_500 = np.load("gapes_Li.npy")
trace_500_df = pd.read_csv("trace_500.csv")
kl_500 = np.load("kl_500.npy")
gapes_linedup_Li = np.load("gapes_linedup_Li.npy")
gapes_linedup_video = np.load("gapes_linedup_video.npy")
trace_late_expanded = pd.read_csv("trace_late_expanded.csv")
trace_late_expanded_after_laser = pd.read_csv("trace_late_expanded_after_laser.csv")

os.chdir("/media/patience/resorted_data/Plots/EM_switch")
gapes_before_700 = np.load("Gapes_before_700_Dur500_Lag700.npy")
gapes_after_700 = np.load("Gapes_after_700_Dur500_Lag700.npy")
trace_before_700_df = pd.read_csv("trace_before_700.csv")
trace_after_700_df = pd.read_csv("trace_after_700.csv")
trace_before_650_df = pd.read_csv("trace_before_650.csv")
trace_before_680_df = pd.read_csv("trace_before_680.csv")
switch_700 = np.load("Switchpoint2_Dur500_Lag700.npy")
switch_control = np.load("Switchpoint2_Dur0_Lag0.npy")
kl_before_700 = np.load("kl_before_700.npy")
kl_after_700 = np.load("kl_after_700.npy")
kl_before_650 = np.load("kl_before_650.npy")
kl_before_680 = np.load("kl_before_680.npy")

os.chdir("/media/patience/resorted_data/Jenn_Data/EMG_Plots")
gapes_Jenn = np.load("gapes.npy")
trace_Jenn_df = pd.read_csv("trace_Jenn.csv")

# Also look at NM47 for unit examples

#os.chdir("/media/patience/resorted_data/NM50/NM50_2500ms_161028_131004")
#hf5 = tables.open_file("NM50_2500ms_161028_131004_repacked.h5", "r")
#spikes_2500_data = hf5.root.spike_trains.dig_in_3.spike_array[:, 2, :]
#laser_durations_2500_data = hf5.root.spike_trains.dig_in_3.laser_durations[:]  
#laser_onset_lag_2500_data = hf5.root.spike_trains.dig_in_3.laser_onset_lag[:]  
os.chdir("/media/patience/resorted_data/NM47/NM47_2500ms_160927_114148")
hf5 = tables.open_file("NM47_2500ms_160927_114148_repacked.h5", "r")
spikes_2500_data = hf5.root.spike_trains.dig_in_0.spike_array[:, 1, :]
laser_durations_2500_data = hf5.root.spike_trains.dig_in_0.laser_durations[:]  
laser_onset_lag_2500_data = hf5.root.spike_trains.dig_in_0.laser_onset_lag[:]  
hf5.close()

#os.chdir("/media/patience/resorted_data/NM51/NM51_500ms_161029_132459")
#hf5 = tables.open_file("NM51_500ms_161029_132459_repacked.h5", "r")
#spikes_500_data = hf5.root.spike_trains.dig_in_3.spike_array[:, 1, :]
#laser_durations_500_data = hf5.root.spike_trains.dig_in_3.laser_durations[:]  
#laser_onset_lag_500_data = hf5.root.spike_trains.dig_in_3.laser_onset_lag[:]
os.chdir("/media/patience/resorted_data/NM47/NM47_500ms_160926_114420")
hf5 = tables.open_file("NM47_500ms_160926_114420_repacked.h5", "r")
spikes_500_data = hf5.root.spike_trains.dig_in_0.spike_array[:, 1, :]
laser_durations_500_data = hf5.root.spike_trains.dig_in_0.laser_durations[:]  
laser_onset_lag_500_data = hf5.root.spike_trains.dig_in_0.laser_onset_lag[:]
hf5.close()

os.chdir("/media/patience/resorted_data/NM43/NM43_2500ms_160515_104159")
hf5 = tables.open_file("NM43_2500ms_160515_104159_repacked.h5", "r")
spikes_2500_excitatory_data_1 = hf5.root.spike_trains.dig_in_0.spike_array[:, 7, :]
laser_durations_2500_excitatory_data_1 = hf5.root.spike_trains.dig_in_0.laser_durations[:]  
laser_onset_lag_2500_excitatory_data_1 = hf5.root.spike_trains.dig_in_0.laser_onset_lag[:]  
hf5.close()

os.chdir("/media/patience/resorted_data/NM50/NM50_2500ms_161028_131004")
hf5 = tables.open_file("NM50_2500ms_161028_131004_repacked.h5", "r")
spikes_2500_excitatory_data_2 = hf5.root.spike_trains.dig_in_0.spike_array[:, 1, :]
laser_durations_2500_excitatory_data_2 = hf5.root.spike_trains.dig_in_0.laser_durations[:]  
laser_onset_lag_2500_excitatory_data_2 = hf5.root.spike_trains.dig_in_0.laser_onset_lag[:]  
hf5.close()

os.chdir("/media/patience/resorted_data/Plots/500ms_laser_effect")
mean_firing_rates_500 = np.load("mean_firing_rates.npy")

os.chdir("/media/patience/resorted_data/Plots/2500ms_laser_effect")
mean_firing_rates_2500 = np.load("mean_firing_rates.npy")

os.chdir("/media/patience/resorted_data/NM50/NM50_500ms_161027_144406")
emg_filt = np.load("emg_filt.npy")
env = np.load("env.npy")

fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
# 2500ms Control trials
pre_stim = 2000
trials = np.where((laser_durations_2500_data[:] == 0.0)*(laser_onset_lag_2500_data[:] == 0.0) > 0)[0]
time = np.arange(spikes_2500_data.shape[1] + 1) - pre_stim
for i in range(len(trials)):
	x = np.where(spikes_2500_data[trials[i], :] > 0.0)[0]
	ax.vlines(x, i, i + 1, colors = 'black')	
#sns.despine(offset=5, trim = True)
ax.set_xticks([1000, 2000, 3000, 4000, 5000])
ax.set_xticklabels([-1.0, 0.0, 1.0, 2.0, 3.0])
plt.xlim([1000, 5000])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel("Trials")

plt.figure(2)
pre_stim = 2000
trials = np.where((laser_durations_2500_data[:] == 2500.0)*(laser_onset_lag_2500_data[:] == 0.0) > 0)[0]
time = np.arange(spikes_2500_data.shape[1] + 1) - pre_stim
for i in range(len(trials)):
	x = np.where(spikes_2500_data[trials[i], :] > 0.0)[0]
	plt.vlines(x, i, i + 1, colors = 'black')	
#sns.despine(offset=5, trim = True)
plt.xticks([1000, 2000, 3000, 4000, 5000], [-1.0, 0.0, 1.0, 2.0, 3.0])
plt.xlim([1000, 5000])
plt.axvspan(2000, 4500, alpha = 0.7, color = sns.xkcd_rgb["green"])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel("Trials")

plt.figure(3)
# 500ms Control trials
pre_stim = 2000
trials = np.where((laser_durations_500_data[:] == 0.0)*(laser_onset_lag_500_data[:] == 0.0) > 0)[0]
time = np.arange(spikes_500_data.shape[1] + 1) - pre_stim
for i in range(len(trials)):
	x = np.where(spikes_500_data[trials[i], :] > 0.0)[0]
	plt.vlines(x, i, i + 1, colors = 'black')	
#sns.despine(offset=5, trim = True)
plt.xticks([1000, 2000, 3000, 4000, 5000], [-1.0, 0.0, 1.0, 2.0, 3.0])
plt.yticks([0, 2, 4, 6, 8])
plt.xlim([1000, 5000])
#plt.axvspan(2000, 4500, alpha = 0.7, color = sns.xkcd_rgb["green"])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel("Trials")

plt.figure(4)
pre_stim = 2000
trials = np.where((laser_durations_500_data[:] == 500.0)*(laser_onset_lag_500_data[:] == 0.0) > 0)[0]
time = np.arange(spikes_500_data.shape[1] + 1) - pre_stim
for i in range(len(trials)):
	x = np.where(spikes_500_data[trials[i], :] > 0.0)[0]
	plt.vlines(x, i, i + 1, colors = 'black')	
#sns.despine(offset=5, trim = True)
plt.xticks([1000, 2000, 3000, 4000, 5000], [-1.0, 0.0, 1.0, 2.0, 3.0])
plt.yticks([0, 2, 4, 6, 8])
plt.xlim([1000, 5000])
plt.axvspan(2000, 2500, alpha = 0.7, color = sns.xkcd_rgb["green"])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel("Trials")

plt.figure(5)
pre_stim = 2000
trials = np.where((laser_durations_500_data[:] == 500.0)*(laser_onset_lag_500_data[:] == 700.0) > 0)[0]
time = np.arange(spikes_500_data.shape[1] + 1) - pre_stim
for i in range(len(trials)):
	x = np.where(spikes_500_data[trials[i], :] > 0.0)[0]
	plt.vlines(x, i, i + 1, colors = 'black')	
#sns.despine(offset=5, trim = True)
plt.xticks([1000, 2000, 3000, 4000, 5000], [-1.0, 0.0, 1.0, 2.0, 3.0])
plt.yticks([0, 2, 4, 6, 8])
plt.xlim([1000, 5000])
plt.axvspan(2700, 3200, alpha = 0.7, color = sns.xkcd_rgb["green"])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel("Trials")

plt.figure(6)
pre_stim = 2000
trials = np.where((laser_durations_500_data[:] == 500.0)*(laser_onset_lag_500_data[:] == 1400.0) > 0)[0]
time = np.arange(spikes_500_data.shape[1] + 1) - pre_stim
for i in range(len(trials)):
	x = np.where(spikes_500_data[trials[i], :] > 0.0)[0]
	plt.vlines(x, i, i + 1, colors = 'black')	
#sns.despine(offset=5, trim = True)
plt.xticks([1000, 2000, 3000, 4000, 5000], [-1.0, 0.0, 1.0, 2.0, 3.0])
plt.yticks([0, 2, 4, 6, 8])
plt.xlim([1000, 5000])
plt.axvspan(3400, 3900, alpha = 0.7, color = sns.xkcd_rgb["green"])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel("Trials")

tastes = ["Dil Suc", "Conc Suc", "Dil Qui", "Conc Qui"]
colors = ["light blue", "blue", "pink", "red"]

plt.figure(7)
for i in range(4):
	mean = np.mean(gapes_500[0, i, :, 2000:4000], axis = 0)
	plt.plot(mean, color = sns.xkcd_rgb[colors[i]], label = tastes[i], linewidth = 4.0)
	#plt.fill_between(mean - sd_500[0, i, 2000:4000], mean + sd_500[0, i, 2000:4000], color = sns.xkcd_rgb[colors[i]], alpha = 0.3)
sns.despine(offset=5, trim = True)
plt.xticks([0, 1000, 2000], [0.0, 1.0, 2.0])
plt.xlabel("Time post taste delivery (s)")
#plt.ylabel(r"$\Pr_{Gape}$" + " averaged across trials")
plt.ylabel(r"$\mathrm{mean(\Pr_{Gape})}$")
plt.legend(loc = "upper left", fontsize = 20)

# Posterior probabilities in 3.5-6Hz are either very close to 0 or to 1. So we convert them to 0 and 1 accordingly, and deduce beta distributions (with pseudocounts = 1) over them
gapes_500[gapes_500 > 0.5] = 1.0
gapes_500[gapes_500 < 0.5] = 0.0
gapes_2500[gapes_2500 > 0.5] = 1.0
gapes_2500[gapes_2500 < 0.5] = 0.0
# Now count the number of 0s and 1s at every time point, calculate the parameters of the beta distributions, and use pymc3 to infer their HPD regions
# First the 500ms condition
a = np.sum(gapes_500, axis = 2)
b = np.sum(1 - gapes_500, axis = 2)
beta_500 = np.zeros(a.shape + (2,))
sd_500 = np.zeros(a.shape)
for i in range(a.shape[0]):
	for j in range(a.shape[1]):
		for k in range(a.shape[2]):
			samples = np.random.beta(a[i, j, k] + 1, b[i, j, k] + 1, size = 100)
			beta_500[i, j, k, :] = list(pm.hpd(samples))
			sd_500[i, j, k] = np.std(samples)  
# Then the 2500ms condition
a = np.sum(gapes_2500, axis = 2)
b = np.sum(1 - gapes_2500, axis = 2)
beta_2500 = np.zeros(a.shape + (2,))
sd_2500 = np.zeros(a.shape)
for i in range(a.shape[0]):
	for j in range(a.shape[1]):
		for k in range(a.shape[2]):
			samples = np.random.beta(a[i, j, k] + 1, b[i, j, k] + 1, size = 100)
			beta_2500[i, j, k, :] = list(pm.hpd(samples))
			sd_2500[i, j, k] = np.std(samples)  
# Plotting the beta bands makes the plots look very cluttered, so we don't do it after all at the end

plt.figure(8)
plt.plot(kl_500[0, 2000:4000], color = sns.xkcd_rgb["black"], linewidth = 4.0)
#plt.axhline(y = 0.0, linewidth = 3.0, linestyle = "--", color = sns.xkcd_rgb["grey"])
sns.despine(offset=5, trim = True)
plt.xticks([0, 1000, 2000], [0.0, 1.0, 2.0])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel(r"$\mathrm{D_{KL}(Conc\/Qui\/ || \/Dil\/Qui)}$", fontsize = 45)

plt.figure(9)
mean = np.mean(np.cumsum(kl_500[0, 2000:4000]))
std = np.std(np.cumsum(kl_500[0, 2000:4000]))
plt.plot(np.cumsum(kl_500[0, 2000:4000]), color = sns.xkcd_rgb["black"], linewidth = 4.0)
switchpoint_500 = np.mean(trace_500_df["switchpoints__0"]*2000)
alpha_0_500 = np.mean(trace_500_df["alpha__0_0"])
alpha_1_500 = np.mean(trace_500_df["alpha__0_0"] + trace_500_df["alpha__0_1"])
beta_0_500 = np.mean(trace_500_df["beta__0_0"])
beta_1_500 = np.mean(trace_500_df["beta__0_0"] + trace_500_df["beta__0_1"])
plt.plot(np.arange(2000), std*(alpha_0_500 + beta_0_500*np.linspace(0, 1, 2000)) + mean, color = sns.xkcd_rgb["grey"], linewidth = 4.0, linestyle = "--")
plt.plot(np.arange(2000), std*(alpha_1_500 + beta_1_500*np.linspace(0, 1, 2000)) + mean, color = sns.xkcd_rgb["grey"], linewidth = 4.0, linestyle = "--")
plt.ylim([-1, 4600])
plt.xticks([0, 1000, 2000], [0.0, 1.0, 2.0])
plt.annotate("Mean onset of gaping", xy = (switchpoint_500, 480), xytext = (switchpoint_500 - 350, 1750), arrowprops=dict(arrowstyle = "fancy", facecolor = sns.xkcd_rgb["black"]), fontsize = 24)
plt.xlabel("Time post taste delivery (s)")
plt.ylabel(r"$\mathrm{cumsum(D_{KL})}$")
sns.despine(offset=5, trim = True)

plt.figure(10)
plt.plot(gapes_500[0, 3, 52, 2000:4000], color = sns.xkcd_rgb["black"], linewidth = 4.0, label = "Posterior probability" + "\n" + "of gaping")
plt.plot(gapes_Li_500[0, 3, 52, :2000]*0.25, color = sns.xkcd_rgb["grey"], linewidth = 5.0, label = "Gapes picked by" + "\n" + "quadratic classifier")
plt.xticks([0, 1000, 2000], [0.0, 1.0, 2.0])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel(r"$\Pr_{Gape}$" + ": Conc Qui trial")
plt.legend(loc = "upper left", fontsize = 20)
sns.despine(offset=5, trim = True)

plt.figure(11)
plt.plot(gapes_500[0, 3, 48, 2000:4000], color = sns.xkcd_rgb["black"], linewidth = 4.0, label = "Posterior probability" + "\n" + "of gaping")
plt.plot(gapes_Li_500[0, 3, 48, :2000]*0.25, color = sns.xkcd_rgb["grey"], linewidth = 5.0, label = "Gapes picked by" + "\n" + "quadratic classifier")
plt.xticks([0, 1000, 2000], [0.0, 1.0, 2.0])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel(r"$\Pr_{Gape}$" + ": Conc Qui trial")
plt.legend(loc = "upper right", fontsize = 20)
sns.despine(offset=5, trim = True)

lasers_500 = ["Control", "Early 0.5s", "Middle 0.5s", "Late 0.5s"]
lasers_500_colors = ["black", "cyan", "magenta", "yellow"]
lasers_500_styles = ["-", "-", "-", "-"]
lasers_2500 = ["Control", "2.5s"]
lasers_2500_colors = ["black", "green"]

plt.figure(12)
hpd_regress_500 = pm.hpd(scaled_regress_500[:, 10:, :], alpha = 0.01)
sig_places_500 = hpd_regress_500[:, :, 0] * hpd_regress_500[:, :, 1] > 0
filtered_regress_500 = np.mean(scaled_regress_500[:, 10:, :], axis = 0)
#std_regress_500 = np.std(scaled_regress_500[:, 10:, i], axis = 0)
plt.plot(filtered_regress_500[:, 0], label = lasers_500[0], color = sns.xkcd_rgb[lasers_2500_colors[0]], linewidth = 4.0)
plt.scatter(np.arange(61)[sig_places_500[:, 0]], filtered_regress_500[sig_places_500[:, 0], 0], color = sns.xkcd_rgb[lasers_2500_colors[0]], s = 200)
plt.plot(filtered_regress_500[:, 1], label = lasers_500[1], color = sns.xkcd_rgb[lasers_2500_colors[1]], linewidth = 4.0)
plt.scatter(np.arange(61)[sig_places_500[:, 1]], filtered_regress_500[sig_places_500[:, 1], 1], color = sns.xkcd_rgb[lasers_2500_colors[1]], s = 200)
plt.plot(sigmoid_regress_500[10:, 0], linewidth = 4.0, color = sns.xkcd_rgb[lasers_2500_colors[0]], linestyle = "--")
plt.plot(sigmoid_regress_500[10:, 1], linewidth = 4.0, color = sns.xkcd_rgb[lasers_2500_colors[1]], linestyle = "--")
#plt.fill_between(np.arange(61), filtered_regress_500 - std_regress_500, filtered_regress_500 + std_regress_500, alpha = 0.3)
#max_pos = np.argmax(filtered_regress_500)	
#plt.plot((max_pos, max_pos), (-0.0012, np.max(filtered_regress_500)), linewidth = 4.0, linestyle = lasers_500_styles[i], color = sns.xkcd_rgb[lasers_500_colors[i]])
#plt.ylim(ymin = -0.0012)
sns.despine(offset = 5, trim = True)
plt.xticks([0, 20, 40, 60], [0.0, 0.5, 1.0, 1.5])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel(r"$\beta_{palatability}$")
plt.legend(loc = "upper left", fontsize = 20)
#ax = plt.gca()
#ax.xaxis.set_label_coords(0.5, -0.18)
#ax.add_line(Line2D([0, 20], [-0.008, -0.008], linewidth = 6.0, color = sns.xkcd_rgb[lasers_500_colors[1]], clip_on = False))
#ax.add_line(Line2D([28, 48], [-0.008, -0.008], linewidth = 6.0, color = sns.xkcd_rgb[lasers_500_colors[2]], clip_on = False))
#ax.add_line(Line2D([56, 60], [-0.008, -0.008], linewidth = 6.0, color = sns.xkcd_rgb[lasers_500_colors[3]], clip_on = False))

plt.figure(12)
plt.plot(filtered_regress_500[:, 0], label = lasers_500[0], color = sns.xkcd_rgb[lasers_2500_colors[0]], linewidth = 4.0)
plt.scatter(np.arange(61)[sig_places_500[:, 0]], filtered_regress_500[sig_places_500[:, 0], 0], color = sns.xkcd_rgb[lasers_2500_colors[0]], s = 200)
plt.plot(sigmoid_regress_500[10:, 0], linewidth = 4.0, color = sns.xkcd_rgb[lasers_2500_colors[0]], linestyle = "--")
plt.plot(filtered_regress_500[:, 2], label = lasers_500[2], color = sns.xkcd_rgb[lasers_2500_colors[1]], linewidth = 4.0)
plt.scatter(np.arange(61)[sig_places_500[:, 2]], filtered_regress_500[sig_places_500[:, 2], 2], color = sns.xkcd_rgb[lasers_2500_colors[1]], s = 200)
plt.plot(sigmoid_regress_500[10:, 2], linewidth = 4.0, color = sns.xkcd_rgb[lasers_2500_colors[1]], linestyle = "--")
plt.ylim(ymax = 0.03)
sns.despine(offset = 5, trim = True)
plt.xticks([0, 20, 40, 60], [0.0, 0.5, 1.0, 1.5])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel(r"$\beta_{palatability}$")
plt.legend(loc = "upper left", fontsize = 20)

plt.figure(12)
plt.plot(filtered_regress_500[:, 0], label = lasers_500[0], color = sns.xkcd_rgb[lasers_2500_colors[0]], linewidth = 4.0)
plt.scatter(np.arange(61)[sig_places_500[:, 0]], filtered_regress_500[sig_places_500[:, 0], 0], color = sns.xkcd_rgb[lasers_2500_colors[0]], s = 200)
plt.plot(sigmoid_regress_500[10:, 0], linewidth = 4.0, color = sns.xkcd_rgb[lasers_2500_colors[0]], linestyle = "--")
plt.plot(filtered_regress_500[:, 3], label = lasers_500[3], color = sns.xkcd_rgb[lasers_2500_colors[1]], linewidth = 4.0)
plt.scatter(np.arange(61)[sig_places_500[:, 3]], filtered_regress_500[sig_places_500[:, 3], 3], color = sns.xkcd_rgb[lasers_2500_colors[1]], s = 200)
plt.plot(sigmoid_regress_500[10:, 3], linewidth = 4.0, color = sns.xkcd_rgb[lasers_2500_colors[1]], linestyle = "--")
plt.ylim(ymax = 0.03)
sns.despine(offset = 5, trim = True)
plt.xticks([0, 20, 40, 60], [0.0, 0.5, 1.0, 1.5])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel(r"$\beta_{palatability}$")
plt.legend(loc = "upper left", fontsize = 20)

plt.figure(13)
hpd_regress_2500 = pm.hpd(scaled_regress_2500[:, 10:, :], alpha = 0.01)
sig_places_2500 = hpd_regress_2500[:, :, 0] * hpd_regress_2500[:, :, 1] > 0
for i in range(2):
	plt.plot(np.mean(scaled_regress_2500[:, 10:, i], axis = 0), label = lasers_2500[i], color = sns.xkcd_rgb[lasers_2500_colors[i]], linewidth = 4.0)
	plt.scatter(np.arange(61)[sig_places_2500[:, i]], np.mean(scaled_regress_2500[:, 10:, i], axis = 0)[sig_places_2500[:, i]], color = sns.xkcd_rgb[lasers_2500_colors[i]], s = 200)
	plt.plot(sigmoid_regress_2500[10:, i], color = sns.xkcd_rgb[lasers_2500_colors[i]], linewidth = 4.0, linestyle = "--")
plt.xticks([0, 20, 40, 60], [0.0, 0.5, 1.0, 1.5])
plt.yticks([-.01, 0.0, .01, .02, .03])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel(r"$\beta_{palatability}$")
plt.legend(loc = "upper left", fontsize = 20)
sns.despine(offset = 5, trim = True)
#ax = plt.gca()
#ax.xaxis.set_label_coords(0.5, -0.18)
#ax.add_line(Line2D([0, 60], [-0.018, -0.018], linewidth = 6.0, color = sns.xkcd_rgb[lasers_2500_colors[1]], clip_on = False))

plt.figure(14)
mean_diff_500 = np.mean(mean_firing_rates_500[:, :, :, :, 0] - \
		mean_firing_rates_500[:, :, :, :, 1], axis = (2, 3))/ \
		np.mean(mean_firing_rates_500[:, :, :, :, 0], axis = (2, 3)) 
for condition in range(3):
	#plt.hist(mean_diff_500[:, condition], alpha = 0.5, bins = 20, label = lasers_500[condition+1], color = lasers_500_colors[condition + 1])
	sns.distplot(mean_diff_500[:, condition], bins = 20, label = lasers_500[condition+1], hist_kws=dict(alpha=0.7), kde = False)
plt.xlabel(r"$\frac{Firing_{LaserOFF} - Firing_{LaserON}}{Firing_{LaserOFF}}$")
plt.ylabel("Number of neurons" + "\n" + "(Total = 244)")
plt.legend(loc = 'upper left', fontsize = 20)

plt.figure(15)
mean_diff_2500 = np.mean(mean_firing_rates_2500[:, :, :, :, 0] - \
		mean_firing_rates_2500[:, :, :, :, 1], axis = (2, 3))/ \
		np.mean(mean_firing_rates_2500[:, :, :, :, 0], axis = (2, 3)) 
#plt.hist(mean_diff_2500[:, 0], alpha = 0.5, bins = 20, label = lasers_2500[1])
sns.distplot(mean_diff_2500[:, 0], bins = 20, label = lasers_2500[1], color = lasers_2500_colors[1], hist_kws = dict(alpha=0.7), kde = False)
plt.xlabel(r"$\frac{Firing_{LaserOFF} - Firing_{LaserON}}{Firing_{LaserOFF}}$")
plt.ylabel("Number of neurons" + "\n" + "(Total = 73)")
plt.legend(loc = 'upper left', fontsize = 20)

plt.figure(16)
emg_2500_artist_colors = ["white", "light green"]
emg_2500_line_colors = ["grass green", "grass green"]
plot_df_2500 = {"Mean onset of behavior post taste delivery (s)": np.concatenate((trace_2500_df["switchpoints__0"]*2000, trace_2500_df["switchpoints__1"]*2000)),
		"Condition": ["Control" for i in range(trace_2500_df["switchpoints__0"].shape[0])] + ["0-2.5s" for i in range(trace_2500_df["switchpoints__1"].shape[0])], 
		"Diff in slopes": np.concatenate((trace_2500_df["beta__0_1"] - trace_2500_df["beta__0_0"], trace_2500_df["beta__1_1"] - trace_2500_df["beta__1_0"]))}
plot_df_2500 = pd.DataFrame(plot_df_2500)
plot_df_2500["Mean onset of behavior post taste delivery (s)"] /= 1000.0
sns.boxplot(x = "Mean onset of behavior post taste delivery (s)", y = "Condition", data = plot_df_2500, order = ["Control", "0-2.5s"], whis = [2.5, 97.5], fliersize = 0)
plt.xticks([1.2, 1.3, 1.4, 1.5, 1.6])
# Check this link: https://stackoverflow.com/questions/36874697/how-to-edit-properties-of-whiskers-fliers-caps-etc-in-seaborn-boxplot
ax = plt.gca()
for i in range(len(ax.artists)):
	ax.artists[i].set_facecolor(sns.xkcd_rgb[emg_2500_artist_colors[i]])
	ax.artists[i].set_edgecolor(sns.xkcd_rgb[emg_2500_line_colors[i]])

	for j in range(6*i, 6*(i+1)):
		ax.lines[j].set_color(sns.xkcd_rgb[emg_2500_line_colors[i]])
		ax.lines[j].set_mfc(sns.xkcd_rgb[emg_2500_line_colors[i]])
		ax.lines[j].set_mec(sns.xkcd_rgb[emg_2500_line_colors[i]])

plt.figure(17)
emg_500_artist_colors = ["white", "light green", "light green", "light green"]
emg_500_line_colors = ["grass green", "grass green", "grass green", "grass green"]
plot_df_500 = {"Mean onset of behavior post taste delivery (s)": np.concatenate((trace_500_df["switchpoints__0"]*2000, trace_500_df["switchpoints__1"]*2000, trace_500_df["switchpoints__2"]*2000, trace_500_df["switchpoints__3"]*2000)),
		"Condition": ["Control" for i in range(trace_500_df["switchpoints__0"].shape[0])] + ["0-0.5s" for i in range(trace_500_df["switchpoints__1"].shape[0])] + ["0.7-1.2s" for i in range(trace_500_df["switchpoints__2"].shape[0])] + ["1.4-1.9s" for i in range(trace_500_df["switchpoints__3"].shape[0])], 
		"Diff in slopes": np.concatenate((trace_500_df["beta__0_1"] - trace_500_df["beta__0_0"], trace_500_df["beta__1_1"] - trace_500_df["beta__1_0"], trace_500_df["beta__2_1"] - trace_500_df["beta__2_0"], trace_500_df["beta__3_1"] - trace_500_df["beta__3_0"]))}
plot_df_500 = pd.DataFrame(plot_df_500)
plot_df_500["Mean onset of behavior post taste delivery (s)"] /= 1000.0
sns.boxplot(x = "Mean onset of behavior post taste delivery (s)", y = "Condition", data = plot_df_500, order = ["Control", "0-0.5s", "0.7-1.2s", "1.4-1.9s"], whis = [2.5, 97.5], fliersize = 0)
ax = plt.gca()
for i in range(len(ax.artists)):
	ax.artists[i].set_facecolor(sns.xkcd_rgb[emg_500_artist_colors[i]])
	ax.artists[i].set_edgecolor(sns.xkcd_rgb[emg_500_line_colors[i]])

	for j in range(6*i, 6*(i+1)):
		ax.lines[j].set_color(sns.xkcd_rgb[emg_500_line_colors[i]])
		ax.lines[j].set_mfc(sns.xkcd_rgb[emg_500_line_colors[i]])
		ax.lines[j].set_mec(sns.xkcd_rgb[emg_500_line_colors[i]])
plt.xticks([0.9, 1.0, 1.1, 1.2])

plt.figure(18)
emg_700_artist_colors = ["light green", "light green", "light green", "light green"]
emg_700_line_colors = ["grass green", "grass green", "grass green", "grass green"]
plot_df_700 = {"Mean onset of behavior post taste delivery (s)": np.concatenate((trace_before_700_df["switchpoints"]*2000, trace_after_700_df["switchpoints"]*2000 + 500, trace_before_650_df["switchpoints"]*2000 - 500)),
		"Condition": ["Before" + "\n" + "0.7s" for i in range(trace_before_700_df["switchpoints"].shape[0])] + ["After" + "\n" + "0.7s" for i in range(trace_after_700_df["switchpoints"].shape[0])] +  ["Before" + "\n" + "0.65s" for i in range(trace_before_650_df["switchpoints"].shape[0])], 
		"Diff in slopes": np.concatenate((trace_before_700_df["beta__1"] - trace_before_700_df["beta__0"], trace_after_700_df["beta__1"] - trace_after_700_df["beta__0"], trace_before_650_df["beta__1"] - trace_before_650_df["beta__0"]))}
plot_df_700 = pd.DataFrame(plot_df_700)
plot_df_700["Mean onset of behavior post taste delivery (s)"] /= 1000.0
sns.boxplot(x = "Mean onset of behavior post taste delivery (s)", y = "Condition", data = plot_df_700, order = ["Before" + "\n" + "0.7s", "After" + "\n" + "0.7s", "Before" + "\n" + "0.65s"], whis = [2.5, 97.5], fliersize = 0)
ax = plt.gca()
for i in range(len(ax.artists)):
	ax.artists[i].set_facecolor(sns.xkcd_rgb[emg_700_artist_colors[i]])
	ax.artists[i].set_edgecolor(sns.xkcd_rgb[emg_700_line_colors[i]])

	for j in range(6*i, 6*(i+1)):
		ax.lines[j].set_color(sns.xkcd_rgb[emg_700_line_colors[i]])
		ax.lines[j].set_mfc(sns.xkcd_rgb[emg_700_line_colors[i]])
		ax.lines[j].set_mec(sns.xkcd_rgb[emg_700_line_colors[i]])
plt.xticks([0.6, 0.8, 1.0, 1.2, 1.4])

plt.figure(19)
sns.distplot(switch_700[2, :], bins = 40, color = sns.xkcd_rgb[colors[2]], kde = False, label = tastes[2])
plt.xlim([0, 2000])
plt.xticks([0, 500, 1000, 1500, 2000], [0, 0.5, 1.0, 1.5, 2.0])
plt.yticks([0, 2, 4, 6, 8, 10])
plt.xlabel("Palatability changepoint (s)")
plt.ylabel("Number of trials")
plt.legend(loc = "upper left", fontsize = 20)

plt.figure(20)
sns.distplot(switch_700[3, :], bins = 40, color = sns.xkcd_rgb[colors[3]], kde = False, label = tastes[3])
plt.xlim([0, 2000])
plt.xticks([0, 500, 1000, 1500, 2000], [0, 0.5, 1.0, 1.5, 2.0])
#plt.yticks([0, 2, 4, 6, 8, 10])
plt.xlabel("Palatability changepoint (s)")
plt.ylabel("Number of trials")
plt.legend(loc = "upper left", fontsize = 20)

plt.figure(20)
sns.distplot(np.concatenate((switch_control[2, :], switch_control[3, :])), bins = 40, color = sns.xkcd_rgb["black"], kde = False, hist_kws = {"alpha": 0.6}, label = "Control")
sns.distplot(np.concatenate((switch_700[2, :], switch_700[3, :])), bins = 40, color = sns.xkcd_rgb["grass green"], kde = False, hist_kws = {"alpha": 0.8}, label = "0.7-1.2s")
plt.xlim([0, 2000])
plt.xticks([0, 500, 1000, 1500, 2000], [0, 0.5, 1.0, 1.5, 2.0])
#plt.yticks([0, 2, 4, 6, 8, 10])
plt.xlabel("Palatability changepoint, $C_P$ (s)")
plt.ylabel("Number of trials")
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.legend(loc = "upper left", fontsize = 20)

plt.figure(21)
means = [np.mean(trace_500_df["switchpoints__0"]),  
	 np.mean(trace_500_df["switchpoints__1"]), 
	 np.mean(trace_500_df["switchpoints__2"]), 
	 np.mean(trace_500_df["switchpoints__3"])]
hpds = [pm.hpd(trace_500_df["switchpoints__0"]),  
	 pm.hpd(trace_500_df["switchpoints__1"]), 
	 pm.hpd(trace_500_df["switchpoints__2"]), 
	 pm.hpd(trace_500_df["switchpoints__3"])]
means = np.array(means)*2
hpds = np.array(hpds)*2
errors = hpds - means[:, None]
plt.bar(np.arange(4) + 1, means, yerr = np.abs(errors).T, edgecolor = sns.xkcd_rgb["grass green"])
plt.xticks(np.arange(4) + 1, ["Control", "0-0.5s", "0.7-1.2s", "1.4-1.9s"])
plt.ylim([0.8, 1.2])

plt.figure(22)
means = [np.mean(trace_2500_df["switchpoints__0"]), np.mean(trace_500_df["switchpoints__0"])]   
hpds = [pm.hpd(trace_2500_df["switchpoints__0"]), pm.hpd(trace_500_df["switchpoints__0"])]
means = np.array(means)*2
hpds = np.array(hpds)*2
errors = hpds - means[:, None]
plt.barh(np.arange(2) + 1, np.flip(means, axis = 0), xerr = np.abs(np.flip(errors, axis = 0)).T)
plt.yticks(np.arange(2) + 1, ["Control" + "\n" + "0.5s", "Control" + "\n" + "2.5s"])
plt.ylabel("Condition")
plt.xlim([0.9, 1.4])
plt.xticks([0.9, 1.1, 1.3])
plt.xlabel("Mean onset of aversive behavior (s)")
ax = plt.gca()
for axis in ['top','bottom','left','right']:
	ax.spines[axis].set_linewidth(5.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for element in ax.containers[1]:
	element.set_color("white")
	element.set_edgecolor(sns.xkcd_rgb["black"])
	element.set_linewidth(5.0)

plt.figure(23)
means = [np.mean(trace_2500_df["switchpoints__1"]) - np.mean(trace_2500_df["switchpoints__0"]),
	 np.mean(trace_500_df["switchpoints__1"]) - np.mean(trace_500_df["switchpoints__0"]), 
	 np.mean(trace_500_df["switchpoints__2"]) - np.mean(trace_500_df["switchpoints__0"]), 
	 np.mean(trace_500_df["switchpoints__3"]) - np.mean(trace_500_df["switchpoints__0"])]
hpds = [ pm.hpd(trace_2500_df["switchpoints__1"] - trace_2500_df["switchpoints__0"]),
	 pm.hpd(trace_500_df["switchpoints__1"] - trace_500_df["switchpoints__0"]), 
	 pm.hpd(trace_500_df["switchpoints__2"] - trace_500_df["switchpoints__0"]), 
	 pm.hpd(trace_500_df["switchpoints__3"] - trace_500_df["switchpoints__0"])]
means = np.array(means)*2
hpds = np.array(hpds)*2
errors = hpds - means[:, None]
plt.barh(np.arange(0, 8, 2) + 2, np.flip(means, axis = 0), xerr = np.abs(np.flip(errors, axis = 0)).T)
plt.yticks(np.arange(0, 8, 2) + 2, ["Late" + "\n" + "0.5s", "Middle" + "\n" + "0.5s", "Early" + "\n" + "0.5s", "2.5s"])
#plt.ylabel("Condition")
plt.ylim([0, 9])
plt.xlabel("Delay in onset of aversive behavior," + "\n" + "compared to within-session control (s)")
plt.xticks([0.0, 0.1, 0.2])
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for element in ax.containers[1]:
	element.set_color(sns.xkcd_rgb["light green"])
	element.set_edgecolor(sns.xkcd_rgb["grass green"])
	element.set_linewidth(2.0)

plt.figure(24)
means = [np.mean(trace_500_df["switchpoints__2"]*2000) - np.mean(trace_500_df["switchpoints__0"]*2000),
	 np.mean(trace_before_650_df["switchpoints"]*2000 - 500) - np.mean(trace_500_df["switchpoints__0"]*2000),
	 np.mean(trace_after_700_df["switchpoints"]*2000 + 500) - np.mean(trace_500_df["switchpoints__0"]*2000)]
hpds = [ pm.hpd(trace_500_df["switchpoints__2"]*2000 - trace_500_df["switchpoints__0"]*2000),
	 pm.hpd(trace_before_650_df["switchpoints"]*2000 - 500 - trace_500_df["switchpoints__0"]*2000),
	 pm.hpd(trace_after_700_df["switchpoints"]*2000 + 500 - trace_500_df["switchpoints__0"]*2000)]
means = np.array(means)/1000
hpds = np.array(hpds)/1000
errors = hpds - means[:, None]
plt.barh(np.arange(3) + 1, np.flip(means, axis = 0), xerr = np.abs(np.flip(errors, axis = 0)).T)
plt.yticks(np.arange(3) + 1, ["C$_P$ after" + "\n" + "laser onset", "C$_P$ before" + "\n" + "laser onset", "All middle" + "\n" + "0.5s trials"])
#plt.ylabel("Condition")
plt.xlabel("Delay in onset of aversive behavior," + "\n" + "compared to within-session control (s)")
plt.xticks([-0.3, -0.1, 0.1, 0.3, 0.5])
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for element in ax.containers[1]:
	element.set_color(sns.xkcd_rgb["light green"])
	element.set_edgecolor(sns.xkcd_rgb["grass green"])
	element.set_linewidth(2.0)

plt.figure(25)
switch_regress_labels = ["$C_P$ before laser onset", "$C_P$ after laser onset"]
hpd_regress_switch_EM = pm.hpd(scaled_regress_switch_EM, alpha = 0.05)
# Check if the HPD for the regression coefficients overlaps zero
sig_places = hpd_regress_switch_EM[:, :, 0] * hpd_regress_switch_EM[:, :, 1] > 0
for i in range(2):
	plt.plot(np.mean(scaled_regress_switch_EM[:, 10:, i], axis = 0), label = switch_regress_labels[i], linewidth = 4.0)
	plt.scatter(np.arange(61)[sig_places[10:, i]], np.mean(scaled_regress_switch_EM[:, 10:, i], axis = 0)[sig_places[10:, i]], s = 200)
plt.xlim([0, 30])
plt.xticks([0, 10, 20, 30], [0.0, 0.25, 0.5, 0.75])
plt.yticks([-.01, 0.0, .01, .02])
sns.despine(offset=10, trim = True)
plt.xlabel("Time post taste delivery (s)")
plt.ylabel(r"$\beta_{palatability}$")
plt.legend(loc = "upper left", fontsize = 20)

plt.figure(26)
x = np.arange(0, 4751, 25)
plot_places = np.where(x <= 2000)[0]
convolved = np.zeros((4, 4, 84, x.shape[0]))
for i in range(4):
	for j in range(4):
		for k in range(84):
			convolved[i, j, k, :] = np.convolve(gapes_Li_500[i, j, k, :], np.ones(250), mode = "valid")[[pos for pos in range(0, 4751, 25)]]
plt.plot(np.mean(gapes_500[0, 3, :, 2000:4000], axis = 0), color = "black", linewidth = 4.0, label = "Bayesian spectrum analysis")
plt.plot(x[plot_places], np.mean(convolved[0, 3, :, :], axis = 0)[plot_places], color = "grey", linewidth = 3.0, linestyle = "--", label = "Quadratic classifier")
#plt.xlim([0, 2000])
plt.xticks([0, 500, 1000, 1500, 2000], [0, 0.5, 1.0, 1.5, 2.0])
plt.yticks([0.0, 0.2, 0.4])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel(r"$\mathrm{\overline{\Pr}_{Gape}}$" + "(Conc Qui)")
plt.legend(loc = "lower right", fontsize = 20)
sns.despine(offset=5, trim = True)

plt.figure(27)
plt.plot(np.mean(gapes_500[0, 2, :, 2000:4000], axis = 0), color = "black", linewidth = 4.0, label = "Bayesian spectrum analysis")
plt.plot(x[plot_places], np.mean(convolved[0, 2, :, :], axis = 0)[plot_places], color = "grey", linewidth = 3.0, linestyle = "--", label = "Quadratic classifier")
#plt.xlim([0, 2000])
plt.xticks([0, 500, 1000, 1500, 2000], [0, 0.5, 1.0, 1.5, 2.0])
plt.yticks([0.0, 0.2, 0.4])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel(r"$\mathrm{\overline{\Pr}_{Gape}}$" + "(Dil Qui)")
plt.legend(loc = "lower right", fontsize = 20)
sns.despine(offset=5, trim = True)

plt.figure(28)
plt.plot(np.mean(gapes_linedup_video, axis = 0), linewidth = 4.0, color = "black", label = "Video coding")
plt.plot(np.mean(gapes_linedup_Li, axis = 0), linewidth = 4.0, color = "grey", label = "Quadratic classifier")
plt.axvline(x = 1000, linewidth = 3.0, color = "black", linestyle = "--", label = "First gape")
plt.xticks([0, 500, 1000, 1500, 2000], [-1.0, -0.5, 0.0, 0.5, 1.0])
plt.yticks([0.25, 0.5, 0.75])
plt.xlabel("Time before first gape from classifier/video (s)")
plt.ylabel(r"$\mathrm{\overline{\Pr}_{Gape}}$" + "(Conc Qui)")
plt.legend(loc = "upper left", fontsize = 20)
sns.despine(offset=5, trim = True)

plt.figure(29)
plt.plot(emg_filt[3, 26, 2000:4000], alpha = 0.8)
plt.plot(env[3, 26, 2000:4000], color = "black", linewidth = 4.0)
plt.xticks([0, 1000, 2000], [0.0, 1.0, 2.0])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel("EMG signal " + r"($\mu$V)")
plt.annotate("Onset of gaping", xy = (960, 90), xytext = (790, 170), arrowprops=dict(arrowstyle = "wedge", facecolor = sns.xkcd_rgb["black"]), fontsize = 24)
sns.despine(offset=5, trim = True)

plt.figure(30)
plt.plot(emg_filt[3, 22, 2000:4000], alpha = 0.8)
plt.plot(env[3, 22, 2000:4000], color = "black", linewidth = 4.0)
plt.xticks([0, 1000, 2000], [0.0, 1.0, 2.0])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel("EMG signal " + r"($\mu$V)")
plt.annotate("Onset of gaping", xy = (375, 90), xytext = (210, 170), arrowprops=dict(arrowstyle = "wedge", facecolor = sns.xkcd_rgb["black"]), fontsize = 24)
sns.despine(offset=5, trim = True)

# We calculate the time of asymptote of the sigmoid as the time when it hits 95% of its full height (L)
plt.figure(31)
asymptote_2500 = []
#asymptote_2500.append(np.log((19*sigmoid_fit_2500["L__0"] - sigmoid_fit_2500["delta__0"])/(sigmoid_fit_2500["L__0"] + sigmoid_fit_2500["delta__0"]))/sigmoid_fit_2500["k__0"] + sigmoid_fit_2500["x0__0"]) 
#asymptote_2500.append(np.log((19*sigmoid_fit_2500["L__1"] - sigmoid_fit_2500["delta__1"])/(sigmoid_fit_2500["L__1"] + sigmoid_fit_2500["delta__1"]))/sigmoid_fit_2500["k__1"] + sigmoid_fit_2500["x0__1"])
asymptote_2500.append(np.log(19)/sigmoid_fit_2500["k__0"] + sigmoid_fit_2500["x0__0"])
asymptote_2500.append(np.log(19)/sigmoid_fit_2500["k__1"] + sigmoid_fit_2500["x0__1"])
asymptote_2500 = np.array(asymptote_2500)
# Some values end up being NaN because of DivideByZero issues. So we only take those that are not NaN.
# For laser condition 1, when the coefficients are never significantly different from 0, all the values are NaN. 
#asymptote_2500[0] = asymptote_2500[0][np.logical_not(np.isnan(asymptote_2500[0]))] 
#asymptote_2500[1] = asymptote_2500[1][np.logical_not(np.isnan(asymptote_2500[1]))] 
# The heights don't have NaNs as they don't involve a complicated transformation
height_2500 = []
height_2500.append(sigmoid_fit_2500["L__0"])
height_2500.append(sigmoid_fit_2500["L__1"])
height_2500 = np.array(height_2500)
# Get current axis
ax1 = plt.gca()
# First calculate the error bars for the barplot
#asymptote_errors = np.abs(pm.hpd(asymptote_2500[0]) - np.mean(asymptote_2500[0]))
asymptote_errors = np.abs(pm.hpd(np.swapaxes(asymptote_2500, 0, 1)) - np.mean(asymptote_2500, axis = 1)[:, None]) 
height_errors = np.abs(pm.hpd(np.swapaxes(height_2500, 0, 1)) - np.mean(height_2500, axis = 1)[:, None])
# Now plot the asymptotes first - subtract 10 from the mean to account for the 10 bins of pre-stimulus data
ax1.bar([1, 3], np.mean(asymptote_2500, axis = 1) - 10, yerr = asymptote_errors.T, width = 0.5, color = sns.color_palette()[0])
ax1.set_yticks([0, 20, 40, 60, 80])
ax1.set_yticklabels([0, 0.5, 1.0, 1.5, 2.0])
ax1.set_ylim(0.0, 83.870017465662485)
ax1.tick_params('y', colors = sns.color_palette()[0])
ax1.set_ylabel("Palatability peak time, $t_{peak}$ (s)", color = sns.color_palette()[0], fontsize = 45)
# Twin the x axis
ax2 = ax1.twinx()
# Now plot the heights
ax2.bar([1.5, 3.5], np.mean(height_2500, axis = 1), width = 0.5, yerr = height_errors.T, color = sns.color_palette()[2])
ax2.set_ylabel("Palatability peak height, $L$", color = sns.color_palette()[2], fontsize = 45)
ax2.tick_params('y', colors = sns.color_palette()[2])
ax1.set_xticks([1.25, 3.25])
ax1.set_xticklabels(["Control", "0-2.5s"])

plt.figure(32)
asymptote_500 = []
#asymptote_500.append(np.log((19*sigmoid_fit_500["L__0"] - sigmoid_fit_500["delta__0"])/(sigmoid_fit_500["L__0"] + sigmoid_fit_500["delta__0"]))/sigmoid_fit_500["k__0"] + sigmoid_fit_500["x0__0"]) 
#asymptote_500.append(np.log((19*sigmoid_fit_500["L__1"] - sigmoid_fit_500["delta__1"])/(sigmoid_fit_500["L__1"] + sigmoid_fit_500["delta__1"]))/sigmoid_fit_500["k__1"] + sigmoid_fit_500["x0__1"])
#asymptote_500.append(np.log((19*sigmoid_fit_500["L__2"] - sigmoid_fit_500["delta__2"])/(sigmoid_fit_500["L__2"] + sigmoid_fit_500["delta__2"]))/sigmoid_fit_500["k__2"] + sigmoid_fit_500["x0__2"])
#asymptote_500.append(np.log((19*sigmoid_fit_500["L__3"] - sigmoid_fit_500["delta__3"])/(sigmoid_fit_500["L__3"] + sigmoid_fit_500["delta__3"]))/sigmoid_fit_500["k__3"] + sigmoid_fit_500["x0__3"])
for i in range(4):
	asymptote_500.append(np.log(19)/sigmoid_fit_500["k__{:d}".format(i)] + sigmoid_fit_500["x0__{:d}".format(i)])
asymptote_500 = np.array(asymptote_500)
# Some values end up being NaN because of DivideByZero issues. So we only take those that are not NaN.
# For laser condition 1, when the coefficients are never significantly different from 0, all the values are NaN. 
#asymptote_500[0] = asymptote_500[0][np.logical_not(np.isnan(asymptote_500[0]))] 
#asymptote_500[1] = asymptote_500[1][np.logical_not(np.isnan(asymptote_500[1]))] 
#asymptote_500[2] = asymptote_500[2][np.logical_not(np.isnan(asymptote_500[2]))] 
#asymptote_500[3] = asymptote_500[3][np.logical_not(np.isnan(asymptote_500[3]))] 
# The heights don't have NaNs as they don't involve a complicated transformation
height_500 = []
height_500.append(sigmoid_fit_500["L__0"])
height_500.append(sigmoid_fit_500["L__1"])
height_500.append(sigmoid_fit_500["L__2"])
height_500.append(sigmoid_fit_500["L__3"])
height_500 = np.array(height_500)
# Get current axis
ax1 = plt.gca()
# First calculate the error bars for the barplot
#asymptote_errors = np.array([np.abs(np.flip(pm.hpd(asymptote_500[0]) - np.mean(asymptote_500[0]), axis = 0)), \
#			     np.abs(np.flip(pm.hpd(asymptote_500[1]) - np.mean(asymptote_500[1]), axis = 0)), \
#			     np.abs(np.flip(pm.hpd(asymptote_500[2]) - np.mean(asymptote_500[2]), axis = 0)), \
#			     np.abs(np.flip(pm.hpd(asymptote_500[3]) - np.mean(asymptote_500[3]), axis = 0))])
asymptote_errors = np.abs(pm.hpd(np.swapaxes(asymptote_500, 0, 1)) - np.mean(asymptote_500, axis = 1)[:, None]) 
height_errors = np.abs(pm.hpd(np.swapaxes(height_500, 0, 1)) - np.mean(height_500, axis = 1)[:, None])
# Now plot the asymptotes first - subtract 10 from the mean to account for the 10 bins of pre-stimulus data
ax1.bar([1, 3, 5, 7], [np.mean(asymptote_500[0]) - 10, np.mean(asymptote_500[1]) - 10, np.mean(asymptote_500[2]) - 10, np.mean(asymptote_500[3]) - 10],  yerr = asymptote_errors.T, width = 0.5, color = sns.color_palette()[0])
ax1.set_yticks([0, 20, 40, 60, 80])
ax1.set_yticklabels([0, 0.5, 1.0, 1.5, 2.0])
ax1.set_ylabel("Palatability peak time, $t_{peak}$ (s)", color = sns.color_palette()[0], fontsize = 45)
ax1.tick_params('y', colors = sns.color_palette()[0])
# Twin the x axis
ax2 = ax1.twinx()
# Now plot the heights
ax2.bar([1.5, 3.5, 5.5, 7.5], np.mean(height_500, axis = 1), width = 0.5, yerr = height_errors.T, color = sns.color_palette()[2])
ax2.set_ylabel("Palatability peak height, $L$", color = sns.color_palette()[2], fontsize = 45)
ax2.tick_params('y', colors = sns.color_palette()[2])
ax1.set_xticks([1.25, 3.25, 5.25, 7.25])
ax1.set_xticklabels(["Control", "0-0.5s", "0.7-1.2s", "1.4-1.9s"])

plt.figure(33)
asymptote_2500 = []
asymptote_2500.append(np.log(19)/sigmoid_fit_2500["k__0"] + sigmoid_fit_2500["x0__0"])
asymptote_2500.append(np.log(19)/sigmoid_fit_2500["k__1"] + sigmoid_fit_2500["x0__1"])
asymptote_2500 = np.array(asymptote_2500)
asymptote_errors = np.abs(pm.hpd(np.swapaxes(asymptote_2500, 0, 1)) - np.mean(asymptote_2500, axis = 1)[:, None]) 
plt.bar([1, 2], np.mean(asymptote_2500, axis = 1) - 10, yerr = asymptote_errors.T, width = 0.5)
plt.yticks([0, 20, 40], [0, 0.5, 1.0])
plt.ylabel(r"$\beta_{palatability}$" + " asymptote (s)")
plt.xticks([1, 2], ["Control", "0-2.5s"])
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.containers[1][0].set_color(sns.xkcd_rgb["white"])
ax.containers[1][0].set_edgecolor(sns.xkcd_rgb["grass green"])
ax.containers[1][0].set_linewidth(2.0)
ax.containers[1][1].set_color(sns.xkcd_rgb["light green"])
ax.containers[1][1].set_edgecolor(sns.xkcd_rgb["grass green"])
ax.containers[1][1].set_linewidth(2.0)

plt.figure(34)
asymptote_500 = []
for i in range(4):
	asymptote_500.append(np.log(19)/sigmoid_fit_500["k__{:d}".format(i)] + sigmoid_fit_500["x0__{:d}".format(i)])
asymptote_500 = np.array(asymptote_500)
asymptote_errors = np.abs(pm.hpd(np.swapaxes(asymptote_500, 0, 1)) - np.mean(asymptote_500, axis = 1)[:, None])
plt.bar([1, 2, 3, 4], np.mean(asymptote_500, axis = 1) - 10,  yerr = asymptote_errors.T, width = 0.5, color = sns.xkcd_rgb["grey"])
plt.yticks([0, 20, 40, 60, 80], [0, 0.5, 1.0, 1.5, 2.0])
plt.ylabel(r"$\beta_{palatability}$" + " asymptote (s)")
plt.xticks([1, 2, 3, 4], ["Control", "0-0.5s", "0.7-1.2s", "1.4-1.9s"])
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.containers[1][0].set_color(sns.xkcd_rgb["white"])
ax.containers[1][0].set_edgecolor(sns.xkcd_rgb["grass green"])
ax.containers[1][0].set_linewidth(2.0)
for i in range(1, 4, 1):
	ax.containers[1][i].set_color(sns.xkcd_rgb["light green"])
	ax.containers[1][i].set_edgecolor(sns.xkcd_rgb["grass green"])
	ax.containers[1][i].set_linewidth(2.0)

plt.figure(35)
height_2500 = []
height_2500.append(sigmoid_fit_2500["L__0"])
height_2500.append(sigmoid_fit_2500["L__1"])
height_2500 = np.array(height_2500)
height_errors = np.abs(pm.hpd(np.swapaxes(height_2500, 0, 1)) - np.mean(height_2500, axis = 1)[:, None])
plt.bar([1, 2], np.mean(height_2500, axis = 1), width = 0.5, yerr = height_errors.T, color = sns.xkcd_rgb["grey"])
plt.ylabel(r"$\beta_{palatability}$" + " peak height")
plt.xticks([1, 2], ["Control", "0-2.5s"])
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.containers[1][0].set_color(sns.xkcd_rgb["white"])
ax.containers[1][0].set_edgecolor(sns.xkcd_rgb["grass green"])
ax.containers[1][0].set_linewidth(2.0)
ax.containers[1][1].set_color(sns.xkcd_rgb["light green"])
ax.containers[1][1].set_edgecolor(sns.xkcd_rgb["grass green"])
ax.containers[1][1].set_linewidth(2.0)

plt.figure(36)
height_500 = []
height_500.append(sigmoid_fit_500["L__0"])
height_500.append(sigmoid_fit_500["L__1"])
height_500.append(sigmoid_fit_500["L__2"])
height_500.append(sigmoid_fit_500["L__3"])
height_500 = np.array(height_500)
height_errors = np.abs(pm.hpd(np.swapaxes(height_500, 0, 1)) - np.mean(height_500, axis = 1)[:, None])
plt.bar([1, 2, 3, 4], np.mean(height_500, axis = 1), width = 0.5, yerr = height_errors.T, color = sns.xkcd_rgb["grey"])
plt.ylabel(r"$\beta_{palatability}$" + " peak height")
plt.xticks([1, 2, 3, 4], ["Control", "0-0.5s", "0.7-1.2s", "1.4-1.9s"])
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.containers[1][0].set_color(sns.xkcd_rgb["white"])
ax.containers[1][0].set_edgecolor(sns.xkcd_rgb["grass green"])
ax.containers[1][0].set_linewidth(2.0)
for i in range(1, 4, 1):
	ax.containers[1][i].set_color(sns.xkcd_rgb["light green"])
	ax.containers[1][i].set_edgecolor(sns.xkcd_rgb["grass green"])
	ax.containers[1][i].set_linewidth(2.0)

# Get number of neurons suppressed and enhanced by optogenetics
diff_500 = mean_firing_rates_500[:, :, :, :, 0] - mean_firing_rates_500[:, :, :, :, 1]
diff_2500 = mean_firing_rates_2500[:, :, :, :, 0] - mean_firing_rates_2500[:, :, :, :, 1]
hpd_500 = pm.hpd(np.swapaxes(diff_500, 0, 3))
hpd_2500 = pm.hpd(np.swapaxes(diff_2500, 0, 3))
print("Neurons excited by 0.5s lasers:", np.sum(np.sum((hpd_500[:, :, :, 0]*hpd_500[:, :, :, 1] > 0)*(hpd_500[:, :, :, 1] < 0), axis = (0, 1)) > 0))
print("Neurons suppressed by 0.5s lasers:", np.sum(np.sum((hpd_500[:, :, :, 0]*hpd_500[:, :, :, 1] > 0)*(hpd_500[:, :, :, 0] > 0), axis = (0, 1)) > 0))
print("Neurons excited by 2.5s lasers:", np.sum(np.sum((hpd_2500[:, :, :, 0]*hpd_2500[:, :, :, 1] > 0)*(hpd_2500[:, :, :, 1] < 0), axis = (0, 1)) > 0))
print("Neurons suppressed by 2.5s lasers:", np.sum(np.sum((hpd_2500[:, :, :, 0]*hpd_2500[:, :, :, 1] > 0)*(hpd_2500[:, :, :, 0] > 0), axis = (0, 1)) > 0))

# Testing if the late (1.4-1.9s) perturbation led to the suppression of ongoing gaping
# We will first find the subset of trials (for both dil and conc Qui) that have significant gaping in the 100ms before laser onset
# Then we will find which of those trials continue to have 4-6Hz activity for 200ms after laser onset
# Find the trials with significant gaping from 1.3s to 1.4s
before_laser_conc = np.where(np.sum(gapes_500[3, 3, :, 3300:3400], axis = -1) > 0)[0]
before_laser_dil = np.where(np.sum(gapes_500[3, 2, :, 3300:3400], axis = -1) > 0)[0]
# Now find the trials that have continual gaping from 1.4s to 1.6s
during_laser_conc = np.where(np.sum(gapes_500[3, 3, :, 3400:3600], axis = -1) == 200)[0] 
during_laser_dil = np.where(np.sum(gapes_500[3, 2, :, 3400:3600], axis = -1) == 200)[0]
# Their intersection gives the trials which obeyed both conditions
not_suppressed_trials = np.concatenate((np.intersect1d(before_laser_conc, during_laser_conc), np.intersect1d(before_laser_dil, during_laser_dil)))
print("For the 1.4-1.9s perturbation:")
print("Total trials: {}".format(before_laser_conc.shape[0] + before_laser_dil.shape[0]))
print("Trials that continued gaping: {}".format(not_suppressed_trials.shape[0]))

plt.figure(37)
plt.plot(gapes_500[3, 3, 18, 2000:4000], color = sns.xkcd_rgb["black"], linewidth = 4.0, label = "Posterior probability" + "\n" + "of gaping")
plt.axvspan(1400, 1900, alpha = 0.7, color = sns.xkcd_rgb["green"], label = "Laser")
plt.xticks([0, 1000, 2000], [0.0, 1.0, 2.0])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel(r"$\Pr_{Gape}$" + ": Conc Qui trial")
plt.legend(loc = "upper left", fontsize = 20)
sns.despine(offset=5, trim = True)

plt.figure(38)
plt.plot(gapes_500[3, 3, 46, 2000:4000], color = sns.xkcd_rgb["black"], linewidth = 4.0, label = "Posterior probability" + "\n" + "of gaping")
plt.axvspan(1400, 1900, alpha = 0.7, color = sns.xkcd_rgb["green"], label = "Laser")
plt.xticks([0, 1000, 2000], [0.0, 1.0, 2.0])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel(r"$\Pr_{Gape}$" + ": Conc Qui trial")
plt.legend(loc = "upper left", fontsize = 20)
sns.despine(offset=5, trim = True)

plt.figure(39)
plt.plot(gapes_500[3, 3, 59, 2000:4000], color = sns.xkcd_rgb["black"], linewidth = 4.0, label = "Posterior probability" + "\n" + "of gaping")
plt.axvspan(1400, 1900, alpha = 0.7, color = sns.xkcd_rgb["green"], label = "Laser")
plt.xticks([0, 1000, 2000], [0.0, 1.0, 2.0])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel(r"$\Pr_{Gape}$" + ": Conc Qui trial")
plt.legend(loc = "upper left", fontsize = 20)
sns.despine(offset=5, trim = True)

plt.figure(40)
plt.plot(gapes_500[3, 3, 73, 2000:4000], color = sns.xkcd_rgb["black"], linewidth = 4.0, label = "Posterior probability" + "\n" + "of gaping")
plt.axvspan(1400, 1900, alpha = 0.7, color = sns.xkcd_rgb["green"], label = "Laser")
plt.xticks([0, 1000, 2000], [0.0, 1.0, 2.0])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel(r"$\Pr_{Gape}$" + ": Conc Qui trial")
plt.legend(loc = "upper left", fontsize = 20)
sns.despine(offset=5, trim = True)

# Plot example units that are activated by optogenetic inhibition
# 2500ms Control trials
plt.figure(41)
pre_stim = 2000
trials = np.where((laser_durations_2500_excitatory_data_1[:] == 0.0)*(laser_onset_lag_2500_excitatory_data_1[:] == 0.0) > 0)[0]
time = np.arange(spikes_2500_excitatory_data_1.shape[1] + 1) - pre_stim
for i in range(len(trials)):
	x = np.where(spikes_2500_excitatory_data_1[trials[i], :] > 0.0)[0]
	plt.vlines(x, i, i + 1, colors = 'black')	
#sns.despine(offset=5, trim = True)
plt.xticks([1000, 2000, 3000, 4000, 5000], [-1.0, 0.0, 1.0, 2.0, 3.0])
plt.xlim([1000, 5000])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel("Trials")

plt.figure(42)
pre_stim = 2000
trials = np.where((laser_durations_2500_excitatory_data_1[:] == 2500.0)*(laser_onset_lag_2500_excitatory_data_1[:] == 0.0) > 0)[0]
time = np.arange(spikes_2500_excitatory_data_1.shape[1] + 1) - pre_stim
for i in range(len(trials)):
	x = np.where(spikes_2500_excitatory_data_1[trials[i], :] > 0.0)[0]
	plt.vlines(x, i, i + 1, colors = 'black')	
#sns.despine(offset=5, trim = True)
plt.xticks([1000, 2000, 3000, 4000, 5000], [-1.0, 0.0, 1.0, 2.0, 3.0])
plt.xlim([1000, 5000])
plt.axvspan(2000, 4500, alpha = 0.7, color = sns.xkcd_rgb["green"])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel("Trials")

# 2500ms Control trials
plt.figure(43)
pre_stim = 2000
trials = np.where((laser_durations_2500_excitatory_data_2[:] == 0.0)*(laser_onset_lag_2500_excitatory_data_2[:] == 0.0) > 0)[0]
time = np.arange(spikes_2500_excitatory_data_2.shape[1] + 1) - pre_stim
for i in range(len(trials)):
	x = np.where(spikes_2500_excitatory_data_2[trials[i], :] > 0.0)[0]
	plt.vlines(x, i, i + 1, colors = 'black')	
#sns.despine(offset=5, trim = True)
plt.xticks([1000, 2000, 3000, 4000, 5000], [-1.0, 0.0, 1.0, 2.0, 3.0])
plt.xlim([1000, 5000])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel("Trials")

plt.figure(44)
pre_stim = 2000
trials = np.where((laser_durations_2500_excitatory_data_2[:] == 2500.0)*(laser_onset_lag_2500_excitatory_data_2[:] == 0.0) > 0)[0]
time = np.arange(spikes_2500_excitatory_data_2.shape[1] + 1) - pre_stim
for i in range(len(trials)):
	x = np.where(spikes_2500_excitatory_data_2[trials[i], :] > 0.0)[0]
	plt.vlines(x, i, i + 1, colors = 'black')	
#sns.despine(offset=5, trim = True)
plt.xticks([1000, 2000, 3000, 4000, 5000], [-1.0, 0.0, 1.0, 2.0, 3.0])
plt.xlim([1000, 5000])
plt.axvspan(2000, 4500, alpha = 0.7, color = sns.xkcd_rgb["green"])
plt.xlabel("Time post taste delivery (s)")
plt.ylabel("Trials")

# Now do the same analysis for control trials
# Find the trials with significant gaping from 1.3s to 1.4s
before_laser_conc = np.where(np.sum(gapes_500[0, 3, :, 3300:3400], axis = -1) > 0)[0]
before_laser_dil = np.where(np.sum(gapes_500[0, 2, :, 3300:3400], axis = -1) > 0)[0]
# Now find the trials that have continual gaping from 1.4s to 1.6s
during_laser_conc = np.where(np.sum(gapes_500[0, 3, :, 3400:3600], axis = -1) == 200)[0] 
during_laser_dil = np.where(np.sum(gapes_500[0, 2, :, 3400:3600], axis = -1) == 200)[0]
# Their intersection gives the trials which obeyed both conditions
not_suppressed_trials = np.concatenate((np.intersect1d(before_laser_conc, during_laser_conc), np.intersect1d(before_laser_dil, during_laser_dil)))
print("For the control condition:")
print("Total trials: {}".format(before_laser_conc.shape[0] + before_laser_dil.shape[0]))
print("Trials that continued gaping: {}".format(not_suppressed_trials.shape[0]))






