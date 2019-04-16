import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping

from vae import VAE


###############
## Load Data ##
###############

# Training set
X_train = pd.read_csv("../data/swedish_train.csv")
X_train = X_train[X_train.expert_PAM50_subtypes != "Normal"]
y_train = X_train["expert_PAM50_subtypes"]

X_train.drop(['expert_PAM50_subtypes'], axis="columns", inplace=True)
'''
# Final test set
X_test = pd.read_csv("../data/swedish_test.csv")
X_test = X_test[X_test.expert_PAM50_subtypes != "Normal"]
y_test = X_test["expert_PAM50_subtypes"]

X_test.drop(['expert_PAM50_subtypes'], axis="columns", inplace=True)

# Order the features correctly
X_test = X_test.reindex(sorted(X_test.columns), axis="columns")
'''
X_train = X_train.reindex(sorted(X_train.columns), axis="columns")


#############################
## 5-Fold Cross Validation ##
#############################

confusion_matrixes = []
validation_set_percent = 0.1
scores = []


skf = StratifiedKFold(n_splits=5)
i=1
classify_df = pd.DataFrame(columns=["Fold", "accuracy"])

for train_index, test_index in skf.split(X_train, y_train):
	print('Fold {} of {}'.format(i, skf.n_splits))

	X_cv_train, X_cv_val = X_train.iloc[train_index], X_train.iloc[test_index]
	y_cv_train, y_cv_val = y_train.iloc[train_index], y_train.iloc[test_index]

	scaler = MinMaxScaler()
	scaler.fit(X_cv_train)
	X_cv_train = pd.DataFrame(scaler.transform(X_cv_train), columns=X_cv_train.columns)
	X_cv_val = pd.DataFrame(scaler.transform(X_cv_val), columns=X_cv_val.columns)

	#Train the Model
	vae = VAE(original_dim=X_cv_train.shape[1], intermediate_dim=300, latent_dim=100, epochs=100, batch_size=50, learning_rate=0.001)

	vae.initialize_model()
	vae.train_vae(train_df=X_cv_train, val_df=X_cv_val)

	# Build and train stacked classifier
	enc = OneHotEncoder(sparse=False)
	y_labels_train = enc.fit_transform(y_cv_train.values.reshape(-1, 1))
	y_labels_val = enc.fit_transform(y_cv_val.values.reshape(-1, 1))

	X_train_train, X_train_val, y_labels_train_train, y_labels_train_val = train_test_split(X_cv_train, y_labels_train, test_size=0.2, stratify=y_labels_train, random_state=42)

	fit_hist = vae.classifier.fit(x=X_train_train, 
									y=y_labels_train_train, 
									shuffle=True, 
									epochs=100,
									batch_size=50,
									callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
									validation_data=(X_train_val, y_labels_train_val))

	score = vae.classifier.evaluate(X_cv_val, y_labels_val)

	print(score)
	scores.append(score[1])

	classify_df = classify_df.append({"Fold":str(i), "accuracy":score[1]}, ignore_index=True)
	history_df = pd.DataFrame(fit_hist.history)
	history_df.to_csv("../parameter_tuning/swedish_classifier_cv_history_"+str(i)+".csv", sep=',')
	i+=1

print('5-Fold results: {}'.format(scores))
print('Average accuracy: {}'.format(np.mean(scores)))


classify_df = classify_df.assign(mean_accuracy=np.mean(scores))
classify_df = classify_df.assign(intermediate_dim=vae.intermediate_dim)
classify_df = classify_df.assign(latent_dim=vae.latent_dim)
classify_df = classify_df.assign(batch_size=vae.batch_size)
classify_df = classify_df.assign(epochs_vae=vae.epochs)
classify_df = classify_df.assign(learning_rate=vae.learning_rate)

output_filename="../parameter_tuning/swedish_classifier_cv.csv"
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


