import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from vae import VAE

X_tcga_no_brca = pd.read_csv("../data/tcga_filtered_no_brca.csv")

###############
## Load Data ##
###############

X_brca_train = pd.read_csv("../data/ciriello_brca_filtered_train.csv")
X_brca_train = X_brca_train[X_brca_train.Ciriello_subtype != "Normal"]

y_brca_train = X_brca_train["Ciriello_subtype"]

X_brca_train.drop(['Ciriello_subtype'], axis="columns", inplace=True)

# Test data
X_brca_test = pd.read_csv("../data/tcga_brca_filtered_test.csv")
X_brca_test = X_brca_test[X_brca_test.subtype != "Normal"]
y_brca_test = X_brca_test["subtype"]

X_brca_test.drop(['subtype'], axis="columns", inplace=True)

#############################
## 5-Fold Cross Validation ##
#############################

confusion_matrixes = []
validation_set_percent = 0.1


skf = StratifiedKFold(n_splits=5)
i=1
epochs = [25, 50, 75, 100]
classify_df = pd.DataFrame(columns=["epochs_classifier", "accuracy_cv"])

for epoch in epochs:
	scores = []
	for train_index, test_index in skf.split(X_brca_train, y_brca_train):
		print('Fold {} of {}'.format(i, skf.n_splits))

		X_train, X_val = X_brca_train.iloc[train_index], X_brca_train.iloc[test_index]
		y_train, y_val = y_brca_train.iloc[train_index], y_brca_train.iloc[test_index]

		# Prepare data to train Variational Autoencoder (merge dataframes and normalize)
		X_autoencoder = pd.concat([X_train, X_tcga_no_brca], sort=True)
		scaler = MinMaxScaler()
		scaler.fit(X_autoencoder)
		X_autoencoder_scaled = pd.DataFrame(scaler.transform(X_autoencoder), columns=X_autoencoder.columns)

		# Scale logistic regression data
		scaler.fit(X_train)
		X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
		X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

		#Split validation set
		X_autoencoder_val = X_autoencoder_scaled.sample(frac=validation_set_percent)
		X_autoencoder_train = X_autoencoder_scaled.drop(X_autoencoder_val.index)

		# Order the features correctly before training

		X_autoencoder_train = X_autoencoder_train.reindex(sorted(X_autoencoder_train.columns), axis="columns")
		X_autoencoder_val = X_autoencoder_val.reindex(sorted(X_autoencoder_val.columns), axis="columns")
		X_train = X_train.reindex(sorted(X_train.columns), axis="columns")
		X_val = X_val.reindex(sorted(X_val.columns), axis="columns")


		#Train the Model
		vae = VAE(original_dim=X_autoencoder_train.shape[1], intermediate_dim=300, latent_dim=100, epochs=100, batch_size=50, learning_rate=0.001)

		vae.initialize_model()
		vae.train_vae(train_df=X_autoencoder_train, val_df=X_autoencoder_val)

		# Build and train stacked classifier
		enc = OneHotEncoder()
		y_labels_train = enc.fit_transform(y_train.values.reshape(-1, 1))
		y_labels_val = enc.fit_transform(y_val.values.reshape(-1, 1))

		fit_hist = vae.classifier.fit(x=X_train, y=y_labels_train, epochs=epoch)
		score = vae.classifier.evaluate(X_val, y_labels_val)

		print(score)
		scores.append(score)
		i+=1

	print('5-Fold results: {}'.format(scores))
	print('Epochs: {}, Accuracy: {}'.format(format(epoch), np.mean(scores)))

	history_df = pd.DataFrame(fit_hist.history)

	classify_df = classify_df.append({"epochs_classifier":str(epoch), "accuracy_cv":np.mean(scores)}, ignore_index=True)

	classify_df = classify_df.assign(intermediate_dim=vae.intermediate_dim)
	classify_df = classify_df.assign(latent_dim=vae.latent_dim)
	classify_df = classify_df.assign(batch_size=vae.batch_size)
	classify_df = classify_df.assign(epochs_vae=vae.epochs)
	classify_df = classify_df.assign(learning_rate=vae.learning_rate)

	output_filename="../parameter_tuning/tcga_tune_classifier_epochs_"+str(epoch)+".csv"
	classify_df.to_csv(output_filename, sep=',')
	history_df.to_csv("../parameter_tuning/tcga_tune_classifier_epochs_"+str(epoch)+"history.csv", sep=',')

'''
#################################
## Build and train final model ##
#################################

# Prepare data to train Variational Autoencoder (merge dataframes and normalize)
X_autoencoder = pd.concat([X_train, X_brca_train], sort=True)
scaler = MinMaxScaler()
scaler.fit(X_autoencoder)
X_autoencoder_scaled = pd.DataFrame(scaler.transform(X_autoencoder), columns=X_autoencoder.columns)

# Scale logistic regression data
scaler.fit(X_train)
X_brca_train_scaled = pd.DataFrame(scaler.transform(X_brca_train), columns=X_brca_train.columns)
X_brca_test_scaled = pd.DataFrame(scaler.transform(X_brca_test), columns=X_brca_test.columns)

X_autoencoder_scaled = X_autoencoder_scaled.reindex(sorted(X_autoencoder_scaled.columns), axis="columns")
X_brca_train_scaled = X_brca_train_scaled.reindex(sorted(X_brca_train_scaled.columns), axis="columns")
X_brca_test_scaled = X_brca_test_scaled.reindex(sorted(X_brca_test_scaled.columns), axis="columns")

vae = VAE(original_dim=X_autoencoder_scaled.shape[1], intermediate_dim=300, latent_dim=100, epochs=100, batch_size=50, learning_rate=0.001)

vae.initialize_model()
vae.train_vae(train_df=X_autoencoder_scaled, val_flag=False)

enc = OneHotEncoder()
y_labels_train = enc.fit_transform(y_brca_train.values.reshape(-1, 1))
y_labels_test = enc.fit_transform(y_brca_test.values.reshape(-1, 1))

vae.classifier.fit(train_df=X_brca_train_scaled, y_df=y_labels_train, epochs=100)
final_score = vae.classifier.evaluate(X_brca_test_scaled, y_labels_test)

classify_df = pd.DataFrame(data=scores, columns=["Accuracy_CV"])
classify_df = classify_df.assign(average_cv_accuracy=np.mean(scores))
classify_df = classify_df.assign(final_accuracy=final_score)
classify_df = classify_df.assign(intermediate_dim=vae.intermediate_dim)
classify_df = classify_df.assign(latent_dim=vae.latent_dim)
classify_df = classify_df.assign(batch_size=vae.batch_size)
classify_df = classify_df.assign(epochs=vae.epochs)
classify_df = classify_df.assign(learning_rate=vae.learning_rate)

output_filename="../results/VAE/tcga_intermediate_dim_"+str(vae.intermediate_dim)+"latent_dim_"+str(vae.latent_dim)+".csv"

classify_df.to_csv(output_filename, sep=',')'''


