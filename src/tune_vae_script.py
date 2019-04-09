import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Input, Lambda, BatchNormalization, Activation
from tensorflow.keras.losses import mse, binary_crossentropy, kullback_leibler_divergence
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import backend as K

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

def sampling(args):
    
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def vae_loss(y_true, y_pred):
    # E[log P(X|z)]
    reconstruction_loss = original_dim * binary_crossentropy(y_true, y_pred) # because it returns the mean cross-entropy
    # reconstruction_loss = mse(y_true, y_pred)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl_loss = -0.5 * K.sum(1. + z_log_var_encoded - K.exp(z_log_var_encoded) - K.square(z_mean_encoded), axis=1)

    return K.mean(reconstruction_loss + kl_loss)

b_size = [50]
ep = [200]
l_rate = [0.001]

for learning_rate in l_rate:
	for batch_size in b_size:
		for n_epoch in ep:
			print("EPOCHS:{}, BATCH:{}".format(n_epoch, batch_size))
			# Built  VAE

			#Build encoder
			inputs = Input(shape=input_shape, name='encoder_input')

			if depth==1:
				z_mean_dense = Dense(latent_dim, name='z_mean')(inputs)
				z_log_var_dense = Dense(latent_dim, name='z_log_var')(inputs)

			elif depth==2:
				hidden_dense = Dense(intermediate_dim)(inputs)
				hidden_dense_batchnorm = BatchNormalization()(hidden_dense)
				hidden_dense_encoded = Activation('relu')(hidden_dense_batchnorm)

				z_mean_dense = Dense(latent_dim, name='z_mean')(hidden_dense_encoded)
				z_log_var_dense = Dense(latent_dim, name='z_log_var')(hidden_dense_encoded)

			z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense)
			z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)

			z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense)
			z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)


			z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean_encoded, z_log_var_encoded])

			encoder = Model(inputs, [z_mean_encoded, z_log_var_encoded, z], name='encoder')

			# Build decoder
			latent_inputs = Input(shape=(latent_dim,), name='z_sampling')

			if depth==1:
				outputs = Dense(input_shape[0], activation='sigmoid', name='decoder_output')(latent_inputs)

			elif depth==2:
				decoder_hidden = Dense(intermediate_dim, activation='relu', name='decoder_hidden')(latent_inputs)
				outputs = Dense(input_shape[0], activation='sigmoid', name='decoder_output')(decoder_hidden)

			decoder = Model(latent_inputs, outputs, name='decoder')

			#Build final model
			outputs = decoder(encoder(inputs)[2]) # fetches the z layer, the sampled one
			vae = Model(inputs, outputs, name='vae')

			adam = optimizers.Adam(lr=learning_rate)

			vae.compile(optimizer=adam, loss=vae_loss)


			# Train the model
			fit_hist = vae.fit(X_autoencoder_train, X_autoencoder_train,
			        shuffle=True,
			        epochs=n_epoch,
			        batch_size=batch_size,
			        validation_data=(X_autoencoder_val, X_autoencoder_val))

			output_filename="../paramsweep_"+str(latent_dim)+"emb_"+str(learning_rate)+"lr_"+str(batch_size)+"bs_"+str(n_epoch)+"epoch.csv"
            #output_filename="../parameter_tuning/VAE/paramsweep_"+str(latent_dim)+"emb_"+str(learning_rate)+"lr_"+str(batch_size)+"bs_"+str(n_epoch)+"epoch.csv"

			history_df = pd.DataFrame(fit_hist.history)
			history_df = history_df.assign(intermediate_dim=intermediate_dim)
			history_df = history_df.assign(latent_dim=latent_dim)
			history_df = history_df.assign(learning_rate=learning_rate)
			history_df = history_df.assign(batch_size=batch_size)
			history_df = history_df.assign(epochs=n_epoch)
			history_df.to_csv(output_filename, sep=',')

