import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import backend as K

from vae import VAEDropout
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

X_train = pd.read_csv("../data/tcga_filtered_scaled_all.csv")
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

if intermediate_dim>0: 
	depth=2
	print("ARCHITECTURE: {}-->{}-->{}".format(original_dim, intermediate_dim, latent_dim))
else: 
	depth=1
	print("ARCHITECTURE: {}-->{}".format(original_dim, latent_dim))

print("DEPTH:{}".format(depth))


d_rates = [0.2, 0.3, 0.4, 0.5]

for d_rate in d_rates:

	vae = VAEDropout(original_dim=original_dim, 
						intermediate_dim=intermediate_dim, 
						latent_dim=latent_dim, 
						epochs=100, 
						batch_size=50, 
						learning_rate=0.001, 
						dropout_rate=d_rate)

	vae.initialize_model()
	vae.train_vae(train_df=X_autoencoder_train, val_df=X_autoencoder_val)

	output_filename="../paramsweep_300_100_optimal_dropout_rate_"+str(d_rate)+".csv"

	vae.hist_dataframe.to_csv(output_filename, sep=',')

