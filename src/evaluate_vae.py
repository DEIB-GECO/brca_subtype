import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping

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
scores = []


skf = StratifiedKFold(n_splits=5)
i=1
classify_df = pd.DataFrame(columns=["Fold", "accuracy"])

for train_index, test_index in skf.split(X_brca_train, y_brca_train):
	print('Fold {} of {}'.format(i, skf.n_splits))

	X_train, X_val = X_brca_train.iloc[train_index], X_brca_train.iloc[test_index]
	y_train, y_val = y_brca_train.iloc[train_index], y_brca_train.iloc[test_index]

	# Prepare data to train Variational Autoencoder (merge dataframes and normalize)
	X_autoencoder = pd.concat([X_train, X_tcga_no_brca], sort=True)
	#scaler = MinMaxScaler()
	#scaler.fit(X_autoencoder)
	#X_autoencoder_scaled = pd.DataFrame(scaler.transform(X_autoencoder), columns=X_autoencoder.columns)

	# Scale logistic regression data
	#scaler.fit(X_train)
	#X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
	#X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

	#Split validation set
	#X_autoencoder_val = X_autoencoder_scaled.sample(frac=validation_set_percent)
	#X_autoencoder_train = X_autoencoder_scaled.drop(X_autoencoder_val.index)
	X_autoencoder_val = X_autoencoder.sample(frac=validation_set_percent)
	X_autoencoder_train = X_autoencoder.drop(X_autoencoder_val.index)

	# Order the features correctly before training

	X_autoencoder_train = X_autoencoder_train.reindex(sorted(X_autoencoder_train.columns), axis="columns")
	X_autoencoder_val = X_autoencoder_val.reindex(sorted(X_autoencoder_val.columns), axis="columns")
	X_train = X_train.reindex(sorted(X_train.columns), axis="columns")
	X_val = X_val.reindex(sorted(X_val.columns), axis="columns")


	#Train the Model
	vae = VAE(original_dim=X_autoencoder_train.shape[1], 
				intermediate_dim=300, 
				latent_dim=100, 
				epochs=100, 
				batch_size=50, 
				learning_rate=0.001, 
				dropout_rate_input=0.2,
				dropout_rate_hidden=0.2)

	vae.initialize_model()
	vae.train_vae(train_df=X_autoencoder_train, val_df=X_autoencoder_val)

	# Build and train stacked classifier
	enc = OneHotEncoder(sparse=False)
	y_labels_train = enc.fit_transform(y_train.values.reshape(-1, 1))
	y_labels_val = enc.fit_transform(y_val.values.reshape(-1, 1))

	X_train_train, X_train_val, y_labels_train_train, y_labels_train_val = train_test_split(X_train, y_labels_train, test_size=0.2, stratify=y_train, random_state=42)

	fit_hist = vae.classifier.fit(x=X_train_train, 
									y=y_labels_train_train, 
									shuffle=True, 
									epochs=100,
									batch_size=50,
									callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
									validation_data=(X_train_val, y_labels_train_val))

	score = vae.classifier.evaluate(X_val, y_labels_val)

	print(score)
	scores.append(score[1])

	classify_df = classify_df.append({"Fold":str(i), "accuracy":score[1]}, ignore_index=True)
	history_df = pd.DataFrame(fit_hist.history)
	history_df.to_csv("../parameter_tuning/tcga_classifier_no_data_norm_dropout_"+str(vae.dropout_rate_input)+"_in_"+str(vae.dropout_rate_hidden)+"_hidden_cv_history_"+str(i)+".csv", sep=',')
	i+=1

print('5-Fold results: {}'.format(scores))
print('Average accuracy: {}'.format(np.mean(scores)))


classify_df = classify_df.assign(mean_accuracy=np.mean(scores))
classify_df = classify_df.assign(intermediate_dim=vae.intermediate_dim)
classify_df = classify_df.assign(latent_dim=vae.latent_dim)
classify_df = classify_df.assign(batch_size=vae.batch_size)
classify_df = classify_df.assign(epochs_vae=vae.epochs)
classify_df = classify_df.assign(learning_rate=vae.learning_rate)
classify_df = classify_df.assign(dropout_input=vae.dropout_rate_input)
classify_df = classify_df.assign(dropout_hidden=vae.dropout_rate_hidden)

output_filename="../parameter_tuning/tcga_classifier_no_data_norm_dropout_"+str(vae.dropout_rate_input)+"_in_"+str(vae.dropout_rate_hidden)+"_hidden_cv.csv"
classify_df.to_csv(output_filename, sep=',')

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

confusion_matrix(y_val_logreg, clf.predict(encoded_val_logreg))

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


