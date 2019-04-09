import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

from vae import VAE

# Parameter grids
b_sizes = [50, 100, 150, 200]
epochs = [25, 50, 75, 100]
l_rates = [0.01, 0.001, 0.0005]

X_train = pd.read_csv("../data/tcga_filtered_scaled_all.csv")

#Split validation set
test_set_percent = 0.1
X_autoencoder_val = X_train.sample(frac=test_set_percent)
X_autoencoder_train = X_train.drop(X_autoencoder_val.index)

# Get model parameters

vae_type = sys.argv[1]
intermediate_dim = int(sys.argv[2])
latent_dim = int(sys.argv[3])

if intermediate_dim>0: 
	print("ARCHITECTURE: input-->{}-->{}".format(intermediate_dim, latent_dim))
else: 
	print("ARCHITECTURE: input-->{}".format(latent_dim))

if vae_type=="vae":
	vae = VAE(original_dim=X_train.shape[1], intermediate_dim=intermediate_dim, latent_dim=latent_dim, epochs=epoch, batch_size=b_size, learning_rate=l_rate)

elif vae_type == "cvae":
	vae = CVAE(original_dim=X_train.shape[1], intermediate_dim=intermediate_dim, latent_dim=latent_dim, epochs=epoch, batch_size=b_size, learning_rate=l_rate)

vae.initialize_model()
vae.train_vae(train_df=X_autoencoder_train, val_df=X_autoencoder_val)
vae.visualize_training()

if vae_type=="vae":
	output_filename="parameter_tuning/VAE/paramsweep_"+str(latent_dim)+"emb_"+str(l_rate)+"lr_"+str(b_size)+"bs_"+str(epoch)+"epoch.csv"

elif vae_type=="cvae":
	output_filename="parameter_tuning/CVAE/paramsweep_"+str(latent_dim)+"emb_"+str(l_rate)+"lr_"+str(b_size)+"bs_"+str(epoch)+"epoch.csv"

history_df = vae.hist_dataframe			
history_df = history_df.assign(intermediate_dim=intermediate_dim)
history_df = history_df.assign(latent_dim=latent_dim)
history_df = history_df.assign(learning_rate=learning_rate)
history_df = history_df.assign(batch_size=batch_size)
history_df = history_df.assign(epochs=n_epoch)
history_df.to_csv(output_filename, sep=',')
