import numpy as np
import tables
import pandas as pd
import os

#---------------------------------------------------------------------------------------------------------------
# Line up gaping probabilities on video coded trials to see if the probability jumps when gaping starts on video
#---------------------------------------------------------------------------------------------------------------

# List of files that whose videos were coded
dir_list = ["NM47/NM47_500ms_160926_114420/", "NM47/NM47_500ms_160929_112820/", "NM50/NM50_500ms_161027_144406/", "NM50/NM50_500ms_161029_122301/", "NM51/NM51_500ms_161029_132459/"]

# Load up the BSA results from each of these files. Only load the results for conc Quinine (taste 3)
BSA_results = []
for filename in dir_list:
	hf5 = tables.open_file("/media/patience/resorted_data/" + filename + str.split(filename, "/")[1] + "_repacked.h5", "r")
	BSA_results.append(hf5.root.emg_BSA_results.taste3_p[:, :, :])
	hf5.close()

BSA_results = np.vstack(tuple(BSA_results))
# Get the probability of frequencies in 4.15-5.95 Hz (7-11)
gapes = np.sum(BSA_results[:, :, 6:11], axis = 2)/np.sum(BSA_results[:, :, :], axis = 2)

# Load up the file with the coded start of gaping from videos
videos = pd.read_table("/media/glia/videos/video_coded_gapes.txt")
# Remove trials where gapes weren't observed due to the animal's movements (marked by 0)
videos = videos[videos["Start of gaping"] > 0]

# Subset the BSA results from the corresponding set of trials 1s before and after the time of start of gaping observed on video
gapes_subset_video = np.zeros((videos.shape[0], 2000))
for trial in range(videos.shape[0]):
	gapes_subset_video[trial, :] = gapes[videos["Trial"].iloc[trial], int(videos["Start of gaping"].iloc[trial]) + 1000 : int(videos["Start of gaping"].iloc[trial]) + 3000]

# Save these subsetted gapes to the folder with EMG plots
os.chdir("/media/patience/resorted_data/Plots/500ms_EMG")
np.save("gapes_linedup_video.npy", gapes_subset_video)

#---------------------------------------------------------------------------------------------
# Now line up gaping probabilities by the time of first gape according to Li et al's algorithm
#---------------------------------------------------------------------------------------------

# Load the gaping probabilities
os.chdir("/media/patience/resorted_data/Plots/500ms_EMG")
gapes_500 = np.load("gapes.npy")
gapes_Li_500 = np.load("gapes_Li.npy")

# Stack up conc Quinine trials in the arrays, irrespective of laser condition
gapes_Li_500 = np.reshape(gapes_Li_500[:, 3, :, :], (-1, gapes_Li_500.shape[-1]))
gapes_500 = np.reshape(gapes_500[:, 3, :, :], (-1, gapes_500.shape[-1]))
# Get the gapes from the Li QDA only on conc Quinine trials
gape_times = np.where(gapes_Li_500 > 0)
gape_times = np.array(gape_times)

# Run through each trial and pick out the time of the first gape
first_gape = [(trial, np.min(gape_times[1, gape_times[0, :] == trial])) for trial in np.unique(gape_times[0, :])]
first_gape = np.array(first_gape)

# Subset the gaping probabilities from the corresponding set of trials 1s before and 1s after the first gape picked by the QDA classifier
gapes_subset_Li = np.zeros((first_gape.shape[0], 2000))
for trial in range(first_gape.shape[0]):
	gapes_subset_Li[trial, :] = gapes_500[first_gape[trial, 0], first_gape[trial, 1] + 1000 : first_gape[trial, 1] + 3000]

# Save these subsetted gapes to the folder with EMG plots
os.chdir("/media/patience/resorted_data/Plots/500ms_EMG")
np.save("gapes_linedup_Li.npy", gapes_subset_Li)








