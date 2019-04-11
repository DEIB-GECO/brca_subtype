import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda, BatchNormalization, Activation, Dropout, concatenate
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from time import time
from tensorflow.python.keras.callbacks import TensorBoard

from IPython.display import SVG

from tensorflow.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

#Limit number of cores on Keras
parallelization_factor = 5

sess = tf.Session(config=
    tf.ConfigProto(
        inter_op_parallelism_threads=parallelization_factor,
               intra_op_parallelism_threads=parallelization_factor,
#                    device_count = {'CPU': parallelization_factor},
))

from base_VAE import BaseVAE


class VAE(BaseVAE):

	"""
    Building and Training a Conditional VAE)
    Modified from:
    https://wiseodd.github.io/techblog/2016/12/17/conditional-vae/
    """

	def __init__(self, original_dim,
						intermediate_dim=0,
						latent_dim=100,
						epochs=100,
						batch_size=50,
						learning_rate=0.01,
						verbose=True):

		BaseVAE.__init__(self)

		self.original_dim = original_dim
		self.intermediate_dim = intermediate_dim
		self.latent_dim = latent_dim
		self.epochs = epochs
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.verbose = verbose

		self.depth = 2 if (intermediate_dim>0) else 1

		print("AUTOENCODER HAS DEPTH {}".format(self.depth))

	def _build_encoder_layers(self):
		self.inputs = Input(shape=(self.original_dim, ), name="encoder_input")

		if self.depth==1:
			z_mean_dense = Dense(self.latent_dim)(self.inputs)
			z_log_var_dense = Dense(self.latent_dim)(self.inputs)

		elif self.depth==2:
			hidden_dense = Dense(self.intermediate_dim)(self.inputs)
			hidden_dense_batchnorm = BatchNormalization()(hidden_dense)
			hidden_dense_encoded = Activation("relu")(hidden_dense_batchnorm)
			z_mean_dense = Dense(self.latent_dim)(hidden_dense_encoded)
			z_log_var_dense = Dense(self.latent_dim)(hidden_dense_encoded)

		# Latent representation layers
		z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense)
		self.z_mean_encoded = Activation("relu")(z_mean_dense_batchnorm)

		z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense)
		self.z_log_var_encoded = Activation("relu")(z_log_var_dense_batchnorm)

		# Sample z
		self.z = Lambda(self.sampling, output_shape=(self.latent_dim,), name="z")([self.z_mean_encoded, self.z_log_var_encoded])

	def _build_decoder_layers(self):

		if self.depth==1:
			self.decoder_output = Dense(self.original_dim, activation="sigmoid", name="decoder_output")
			self.outputs = self.decoder_output(self.z)

		elif self.depth==2:
			self.decoder_hidden = Dense(self.intermediate_dim, activation="relu", name="decoder_hidden")
			self.decoder_output = Dense(self.original_dim, activation="sigmoid", name="decoder_output")
			vae_decoder_hidden = self.decoder_hidden(self.z)
			self.outputs = self.decoder_output(vae_decoder_hidden)

	def _compile_encoder_decoder(self):

		# Compile Encoder
		self.encoder = Model(self.inputs, [self.z_mean_encoded, self.z_log_var_encoded], name="encoder")

		# Compile Decoder
		decoder_input = Input(shape=(self.latent_dim,), name='z_sampling')

		if self.depth==1:
			x_decoded = self.decoder_output(decoder_input)
		elif self.depth==2:
			x_hidden = self.decoder_hidden(decoder_input)
			x_decoded = self.decoder_output(x_hidden)

		self.decoder = Model(decoder_input, x_decoded, name='decoder')


	def _compile_vae(self):
		"""
		Compiles all the layers together, creating the Variational Autoencoder
		"""

		adam = optimizers.Adam(lr=self.learning_rate)
		self.vae = Model(self.inputs, self.outputs, name='vae')
		self.vae.compile(optimizer=adam, loss=self.vae_loss)

	def _build_classifier(self):

		fully_con_classifier = Dense(self.latent_dim, activation="relu", name="classifier_fully_con")(self.z_mean_encoded)
		self.classifier_output = Dense(4, activation="softmax", name="classifier_output")(fully_con_classifier)

		self.classifier = Model(self.inputs, self.classifier_output, name="classifier")
		
		adam = optimizers.Adam(lr=self.learning_rate)
		self.classifier.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])


	def train_vae(self, train_df, val_df, val_flag=True):
		if(val_flag):
			self.train_hist = self.vae.fit(train_df, train_df,
							shuffle=True,
							epochs=self.epochs,
							batch_size=self.batch_size,
							callbacks=[tensorboard],
							validation_data=(val_df, val_df))
		else:
			self.train_hist = self.vae.fit(train_df, train_df,
							shuffle=True,
							epochs=self.epochs,
							batch_size=self.batch_size)

		self.hist_dataframe = pd.DataFrame(self.train_hist.history)


	def train_stacked_classifier(self, train_df, val_df, epochs):
		self.classifier.fit(train_df=X_train, y_df=y_labels_train, epochs=epochs)

	def evaluate_stacked_classifier(self, X_test, y_test):
		score = self.classifier.evaluate(X_test, y_test)
		return score



class VAEDropout(BaseVAE):

	"""
    Building and Training a Conditional VAE)
    Modified from:
    https://wiseodd.github.io/techblog/2016/12/17/conditional-vae/
    """

	def __init__(self, original_dim,
						intermediate_dim=0,
						latent_dim=100,
						epochs=100,
						batch_size=50,
						learning_rate=0.01,
						dropout_rate=0.2,
						verbose=True):

		BaseVAE.__init__(self)

		self.original_dim = original_dim
		self.intermediate_dim = intermediate_dim
		self.latent_dim = latent_dim
		self.epochs = epochs
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.verbose = verbose
		self.dropout_rate = dropout_rate

		self.depth = 2 if (intermediate_dim>0) else 1

		print("AUTOENCODER HAS DEPTH {}".format(self.depth))

	def _build_encoder_layers(self):
		self.inputs = Input(shape=(self.original_dim, ), name="encoder_input")

		dropout_input = Dropout(rate=self.dropout_rate)(self.inputs)

		if self.depth==1:
			z_mean_dense = Dense(self.latent_dim)(dropout_input)
			z_log_var_dense = Dense(self.latent_dim)(dropout_input)

		elif self.depth==2:
			hidden_dense = Dense(self.intermediate_dim)(dropout_input)
			hidden_dense_batchnorm = BatchNormalization()(hidden_dense)
			hidden_dense_encoded = Activation("relu")(hidden_dense_batchnorm)
			dropout_encoder_hidden = Dropout(rate=self.dropout_rate)(hidden_dense_encoded)
			z_mean_dense = Dense(self.latent_dim)(dropout_encoder_hidden)
			z_log_var_dense = Dense(self.latent_dim)(dropout_encoder_hidden)

		# Latent representation layers
		z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense)
		self.z_mean_encoded = Activation("relu")(z_mean_dense_batchnorm)

		z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense)
		self.z_log_var_encoded = Activation("relu")(z_log_var_dense_batchnorm)

		# Sample z
		self.z = Lambda(self.sampling, output_shape=(self.latent_dim,), name="z")([self.z_mean_encoded, self.z_log_var_encoded])

	def _build_decoder_layers(self):

		if self.depth==1:
			self.decoder_output = Dense(self.original_dim, activation="sigmoid", name="decoder_output")
			self.outputs = self.decoder_output(self.z)

		elif self.depth==2:
			self.decoder_hidden = Dense(self.intermediate_dim, activation="relu", name="decoder_hidden")
			self.decoder_output = Dense(self.original_dim, activation="sigmoid", name="decoder_output")
			vae_decoder_hidden = self.decoder_hidden(self.z)
			self.outputs = self.decoder_output(vae_decoder_hidden)

	def _compile_encoder_decoder(self):

		# Compile Encoder
		self.encoder = Model(self.inputs, [self.z_mean_encoded, self.z_log_var_encoded], name="encoder")

		# Compile Decoder
		decoder_input = Input(shape=(self.latent_dim,), name='z_sampling')

		if self.depth==1:
			x_decoded = self.decoder_output(decoder_input)
		elif self.depth==2:
			x_hidden = self.decoder_hidden(decoder_input)
			x_decoded = self.decoder_output(x_hidden)

		self.decoder = Model(decoder_input, x_decoded, name='decoder')


	def _compile_vae(self):
		"""
		Compiles all the layers together, creating the Variational Autoencoder
		"""

		adam = optimizers.Adam(lr=self.learning_rate)
		self.vae = Model(self.inputs, self.outputs, name='vae')
		self.vae.compile(optimizer=adam, loss=self.vae_loss)

	def _build_classifier(self):

		fully_con_classifier = Dense(self.latent_dim, activation="relu", name="classifier_fully_con")(concatenate([self.z_mean_encoded, self.z_log_var_encoded]))
		self.classifier_output = Dense(4, activation="softmax", name="classifier_output")(fully_con_classifier)

		self.classifier = Model(self.inputs, self.classifier_output, name="classifier")
		
		adam = optimizers.Adam(lr=self.learning_rate)
		self.classifier.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])


	def train_vae(self, train_df, val_df, val_flag=True):
		if(val_flag):
			self.train_hist = self.vae.fit(train_df, train_df,
							shuffle=True,
							epochs=self.epochs,
							batch_size=self.batch_size,
							callbacks=[EarlyStopping(monitor='val_loss', patience=10), tensorboard],
							validation_data=(val_df, val_df))
		else:
			self.train_hist = self.vae.fit(train_df, train_df,
							shuffle=True,
							epochs=self.epochs,
							batch_size=self.batch_size)

		self.hist_dataframe = pd.DataFrame(self.train_hist.history)


	def train_stacked_classifier(self, train_df, val_df, epochs):
		self.classifier.fit(train_df=X_train, y_df=y_labels_train, epochs=epochs)

	def evaluate_stacked_classifier(self, X_test, y_test):
		score = self.classifier.evaluate(X_test, y_test)
		return score


class ConditionalVAE(BaseVAE):

	"""
    Building and Training a Conditional VAE)
    Modified from:
    https://wiseodd.github.io/techblog/2016/12/17/conditional-vae/
    """

	def __init__(self, original_dim, 
						intermediate_dim=0, 
						latent_dim=100,
						cond_dim=35, 
						epochs=100, 
						batch_size=50, 
						learning_rate=0.01, 
						verbose=True):

		BaseVAE.__init__(self)

		self.original_dim = original_dim
		self.intermediate_dim = intermediate_dim
		self.latent_dim = latent_dim
		self.cond_dim = cond_dim
		self.cvae_input_dim = original_dim + cond_dim
		self.cvae_latent_dim = original_dim + cond_dim
		self.epochs = epochs
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.verbose = verbose

		self.depth = 2 if (intermediate_dim>0) else 1

	def _build_encoder_layers(self):
		"""
		Function to build encoder part of Conditional Variational Autoencoder
		"""

		# Input layers for input data and condition
		self.input_data = Input(shape=(self.original_dim, ))
		self.input_cond = Input(shape=(self.cond_dim, ))

		self.inputs = concatenate([self.input_data, self.input_cond])

		# Encoder layers
		if self.depth==1:
			z_mean_dense = Dense(self.latent_dim, name="z_mean_dense")(self.inputs)
			z_log_var_dense = Dense(self.latent_dim, name="z_log_var_dense")(self.inputs)

		elif self.depth==2:
			hidden_dense = Dense(self.intermediate_dim)(self.inputs)
			hidden_dense_batchnorm = BatchNormalization()(hidden_dense)
			hidden_dense_encoded = Activation("relu")(hidden_dense_batchnorm)
			z_mean_dense = Dense(self.latent_dim, name="z_mean_dense")(hidden_dense_encoded)
			z_log_var_dense = Dense(self.latent_dim, name="z_log_var_dense")(hidden_dense_encoded)
		
		# Latent representation layers
		z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense)
		self.z_mean_encoded = Activation("relu")(z_mean_dense_batchnorm)

		
		z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense)
		self.z_log_var_encoded = Activation("relu")(z_log_var_dense_batchnorm)

		# Sample z
		self.z = Lambda(self.sampling, output_shape=(self.latent_dim,), name="z")([self.z_mean_encoded, self.z_log_var_encoded])
		self.z_cond = concatenate([self.z, self.input_cond])

	def _build_decoder_layers(self):
		"""
		Function to build encoder part of Conditional Variational Autoencoder
		"""

		if self.depth==1:
			self.decoder_output = Dense(self.original_dim, activation="sigmoid", name="decoder_output")
			self.outputs = self.decoder_output(self.z_cond)

		elif self.depth==2:
			self.decoder_hidden = Dense(self.intermediate_dim, activation="relu", name="decoder_hidden")
			self.decoder_output = Dense(self.original_dim, activation="sigmoid", name="decoder_output")
			self.cvae_decoder_hidden = self.decoder_hidden(self.z_cond)
			self.outputs = self.decoder_output(self.cvae_decoder_hidden)

	def _compile_encoder_decoder(self):
		"""
		Compiles the layers of the encoder and decoder parts, creating the Encoder and Decoder models separately
		"""

		# Compile Encoder
		self.encoder = Model([self.input_data, self.input_cond], self.z_mean_encoded, name="encoder")

		# Compile Decoder
		self.decoder_input = Input(shape=(self.cvae_latent_dim,), name='z_sampling')

		if self.depth==1:
			x_decoded = self.decoder_output(self.decoder_input)
		elif self.depth==2:
			x_hidden = self.decoder_hidden(self.decoder_input)
			x_decoded = self.decoder_output(x_hidden)

		self.decoder = Model(self.decoder_input, x_decoded, name='decoder')

	def _compile_vae(self):
		"""
		Compiles all the layers together, creating the Conditional Variational Autoencoder
		"""

		adam = optimizers.Adam(lr=self.learning_rate)
		self.cvae = Model([self.input_data, self.input_cond], self.outputs, name='cvae')
		self.cvae.compile(optimizer=adam, loss=self.vae_loss)

	def _build_classifier(self):

		fully_con_classifier = Dense(self.latent_dim, activation="relu", name="classifier_fully_con")(self.z_mean_encoded)
		self.classifier_output = Dense(4, activation="softmax", name="classifier_output")(fully_con_classifier)

		self.classifier = Model([self.input_data, self.input_cond], self.classifier_output, name="classifier")
		
		adam = optimizers.Adam(lr=self.learning_rate)
		self.classifier.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])


	def train_cvae(self, train_df, train_cond_df, val_df, val_cond_df, val_flag=True):
		if(val_flag):
			self.train_hist = self.cvae.fit([train_df, train_cond_df], train_df,
							shuffle=True,
							epochs=self.epochs,
							batch_size=self.batch_size,
							callbacks=[tensorboard],
							validation_data=([val_df, val_cond_df], val_df))
		else:
			self.train_hist = self.cvae.fit([train_df, train_cond_df], train_df,
							shuffle=True,
							epochs=self.epochs,
							batch_size=self.batch_size,
							callbacks=[tensorboard])

		self.hist_dataframe = pd.DataFrame(self.train_hist.history)
