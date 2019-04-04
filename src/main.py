import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from vae import VAE

X_train = pd.read_csv("../data/tcga_filtered_scaled_all.csv")

#Split validation set
test_set_percent = 0.1
X_autoencoder_val = X_train.sample(frac=test_set_percent)
X_autoencoder_train = X_train.drop(X_autoencoder_val.index)

vae = VAE(original_dim=X_train.shape[1], intermediate_dim=0, latent_dim=100, epochs=5, batch_size=50, learning_rate=0.01)

vae.initialize_model()
vae.train_vae(train_df=X_autoencoder_train, val_df=X_autoencoder_val)
vae.visualize_training()