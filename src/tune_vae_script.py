import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
import matplotlib.pyplot as plt

from vae import VAE

import sys

parallelization_factor = 5

sess = tf.Session(config=
    tf.ConfigProto(
        inter_op_parallelism_threads=parallelization_factor,
               intra_op_parallelism_threads=parallelization_factor,
#                    device_count = {'CPU': parallelization_factor},
))


K.set_session(sess)
# Read the data

X_train = pd.read_pickle("../data/tcga_filtered_scaled_all.pkl")
print("LOADED DATA")

#Split validation set
test_set_percent = 0.1
X_autoencoder_val = X_train.sample(frac=test_set_percent)
X_autoencoder_train = X_train.drop(X_autoencoder_val.index)

# VAE auxiliary funtions

original_dim = X_train.shape[1]
input_shape = (original_dim,)
intermediate_dim = int(sys.argv[1])
latent_dim = int(sys.argv[2])
dropout_input=0
dropout_hidden=0

if intermediate_dim>0: 
	depth=2
	print("ARCHITECTURE: {}-->{}-->{}".format(original_dim, intermediate_dim, latent_dim))
else: 
	depth=1
	print("ARCHITECTURE: {}-->{}".format(original_dim, latent_dim))

print("DEPTH:{}".format(depth))


b_size = [50, 100, 150, 200]
ep = [150, 200]
l_rate = [0.01, 0.001, 0.0005]

for learning_rate in l_rate:
	for batch_size in b_size:
		for n_epoch in ep:
			print("EPOCHS:{}, BATCH:{}".format(n_epoch, batch_size))

			vae = VAE(original_dim=original_dim, 
				intermediate_dim=intermediate_dim, 
				latent_dim=latent_dim, 
				epochs=n_epoch, 
				batch_size=batch_size, 
				learning_rate=learning_rate, 
				dropout_rate_input=dropout_input,
				dropout_rate_hidden=dropout_hidden,
				freeze_weights=False)

			vae.initialize_model()
			vae.train_vae(train_df=X_autoencoder_train, val_df=X_autoencoder_val)
			

			output_filename="../paramsweep_{}emb_{}lr_{}bs_{}epoch.csv".format(latent_dim, learning_rate, batch_size, n_epoch)

			history_df = pd.DataFrame(fit_hist.history)
			history_df = history_df.assign(intermediate_dim=intermediate_dim)
			history_df = history_df.assign(latent_dim=latent_dim)
			history_df = history_df.assign(learning_rate=learning_rate)
			history_df = history_df.assign(batch_size=batch_size)
			history_df = history_df.assign(epochs=n_epoch)
			history_df = history_df.assign(dropout_input=dropout_input)
			history_df = history_df.assign(dropout_hidden=dropout_hidden)
			history_df.to_csv(output_filename, sep=',')

