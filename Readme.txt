2016.11.23
Code for bird detection. 
Dataset: warblrb (8000 clips of 10 second)
====================

* run matlab/generate_denoise_wavs.m	# generate denoise wavs. YOU NEED TO MODIFY PATHS IN THE .M FILE!

* MODIFY THE PATHS IN THE config.py	# Set the paths of the dataset and your working space. 
* run create_cv.py			# Create 10 folds for development dataset. (Because the whole dataset is large, we use the subfold for training)
* run prepare_data.py --warblrb --denoise_mel		# Extract 40 Mel filter bank feature. 
* run prepare_data.py --warblrb --denoise_spectrogram 	# (optional, extract spectrogram feature)

* run main_dnn_jdc.py --pre_load	# Pre-load data
* run main_dnn_jdc.py --train		# Train the model
* run main_dnn_jdc.py --recognize	# Recognize, acc of 83% will be obtained

