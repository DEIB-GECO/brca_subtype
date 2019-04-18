import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from vae import VAE, CVAE


#####################
## Parse arguments ##
#####################

def get_params(param):
	params = parameter_df.loc[param, "value"]
	print(params)
	return params


parser = argparse.ArgumentParser(description="Get VAE arguments")
parser.add_argument("--parameter_file", "--p", help="Location of the file containing the parameters")
parser.add_argument("--hidden_dim", "--h", type=int, help="Dimension of hidden layer, 0 if none")
parser.add_argument("--latent_dim", "--l", type=int, help="Dimension of embedded layer, 0 if none")
parser.add_argument("--batch_size", "--b", type=int, help="Batch size for VAE train")
parser.add_argument("--epochs", "--e", type=int, help="Number of epochs for the VAE")
parser.add_argument("--learning_rate", "--l_rate", type=float, help="Learning rate for the VAE")
parser.add_argument("--dropout_input", "--d_in", type=float, help="Dropout rate of the input layer")
parser.add_argument("--dropout_hidden", "--d_hidden", type=float, help="Dropout rate of the hidden layers")
parser.add_argument("--dropout_decoder", "--d_decoder", type=bool, help="Flag for decoder dropout: 0 for dropout only on encoder, 1 otherwise")
parser.add_argument("--freeze_weights", "--freeze", type=bool, help="Flag that tells whether the Autoencoder weights are frozen or not when training the classifier")

args = parser.parse_args()

if args.parameter_file is not None:
	parameter_df = pd.read_csv(args.parameter_file, index_col=0)

	hidden_dim = int(get_params("hidden_dim"))
	latent_dim = int(get_params("latent_dim"))
	batch_size = int(get_params("batch_size"))
	epochs = int(get_params("epochs"))
	learning_rate = get_params("learning_rate")
	dropout_input = get_params("dropout_input")
	dropout_hidden = get_params("dropout_hidden")
	dropout_decoder = bool(get_params("dropout_decoder"))
	freeze_weights = bool(get_params("freeze_weights"))

else:

	hidden_dim = args.hidden_dim
	latent_dim = args.latent_dim
	batch_size = args.batch_size
	epochs = args.epochs
	learning_rate = args.learning_rate
	dropout_input = args.dropout_input
	dropout_hidden = args.dropout_hidden
	dropout_decoder = args.dropout_decoder
	freeze_weights = args.freeze_weights


###############
## Load Data ##
###############

X_tcga_no_brca = pd.read_pickle("../data/tcga_filtered_no_brca.pkl")
x_tcga_type_no_brca = pd.read_pickle("../data/tcga_tumor_type.pkl")
x_tcga_type_no_brca = x_tcga_type_no_brca[x_tcga_type_no_brca.tumor_type != "BRCA"]

X_brca_train = pd.read_pickle("../data/ciriello_brca_filtered_train.pkl")
X_brca_train = X_brca_train[X_brca_train.Ciriello_subtype != "Normal"]

y_brca_train = X_brca_train["Ciriello_subtype"]

X_brca_train.drop(['Ciriello_subtype'], axis="columns", inplace=True)

# Test data
X_brca_test = pd.read_pickle("../data/tcga_brca_filtered_test.pkl")
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
	X_train_tumor_type = pd.DataFrame(data=["BRCA"]*len(X_train), columns=["tumor_type"])
	X_autoencoder_tumor_type = pd.concat([X_train_tumor_type, x_tcga_type_no_brca], sort=True)

	scaler = MinMaxScaler()
	scaler.fit(X_autoencoder)
	X_autoencoder_scaled = pd.DataFrame(scaler.transform(X_autoencoder), columns=X_autoencoder.columns)

	# Scale logistic regression data
	scaler.fit(X_train)
	X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
	X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

	#Split validation set
	X_autoencoder_train, X_autoencoder_val, X_autoencoder_tumor_type_train, X_autoencoder_tumor_type_val = train_test_split(X_autoencoder_scaled, X_autoencoder_tumor_type, test_size=validation_set_percent, stratify=X_autoencoder_tumor_type, random_state=42)


	# Order the features correctly before training
	X_train = X_train.reindex(sorted(X_train.columns), axis="columns")
	X_val = X_val.reindex(sorted(X_val.columns), axis="columns")


	#Train the Model
	cvae = CVAE(original_dim=X_autoencoder_train.shape[1], 
					intermediate_dim=hidden_dim, 
					latent_dim=latent_dim, 
					cond_dim=35,
					epochs=epochs, 
					batch_size=batch_size, 
					learning_rate=0.learning_rate,
					dropout_rate_input=dropout_input,
					dropout_rate_hidden=dropout_hidden,
					freeze_weights=freeze_weights)

	cvae.initialize_model()
	cvae.train_vae(train_df=X_autoencoder_train, 
					train_cond_df=pd.get_dummies(X_autoencoder_tumor_type_train), 
					val_df=X_autoencoder_val,
					val_cond_df=pd.get_dummies(X_autoencoder_tumor_type_val))


	# Build and train stacked classifier
	enc = OneHotEncoder(sparse=False)
	y_labels_train = enc.fit_transform(y_train.values.reshape(-1, 1))
	y_labels_val = enc.fit_transform(y_val.values.reshape(-1, 1))

	X_train_train, X_train_val, y_labels_train_train, y_labels_train_val = train_test_split(X_train, y_labels_train, test_size=0.2, stratify=y_train, random_state=42)

	tumors = X_autoencoder_tumor_type_train["tumor_type"].unique()


	X_train_train_tumor_type = pd.DataFrame(0, index=np.arange(len(X_train_train)), columns=tumors)
	X_train_train_tumor_type["BRCA"]=1
	X_train_val_tumor_type = pd.DataFrame(0, index=np.arange(len(X_train_val)), columns=tumors)
	X_train_val_tumor_type["BRCA"]=1
    
	X_val_tumor_type = pd.DataFrame(0, index=np.arange(len(X_val)), columns=tumors)
	X_val_tumor_type["BRCA"]=1

	cvae.build_classifier()

	fit_hist = cvae.classifier.fit(x=[X_train_train, X_train_train_tumor_type], 
									y=y_labels_train_train, 
									shuffle=True, 
									epochs=100,
									batch_size=50,
									callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
									validation_data=([X_train_val, X_train_val_tumor_type], y_labels_train_val))

	if(cvae.classifier_use_z)
		X_val_new = pd.DataFrame(np.repeat(X_val.values, 10, axis=0))
		X_val_new.columns = X_val.columns

		X_val_tumor_type_new = pd.DataFrame(np.repeat(X_val_tumor_type.values, 10, axis=0))
		X_val_tumor_type_new.columns = X_val_tumor_type.columns

		y_labels_val_new = pd.DataFrame(np.repeat(y_labels_val.values, 10, axis=0))
		y_labels_val_new.columns = y_labels_val.columns

		score = cvae.classifier.evaluate([X_val_new, X_val_tumor_type_new], y_labels_val_new)
	else:
		score = cvae.classifier.evaluate([X_val, X_val_tumor_type], y_labels_val)

	print(score)
	scores.append(score[1])

	classify_df = classify_df.append({"Fold":str(i), "accuracy":score[1]}, ignore_index=True)
	history_df = pd.DataFrame(fit_hist.history)
	filename = "../parameter_tuning/cvae_tcga_classifier_dropout_{}_in_{}_hidden_cv_frozen_{}_history_{}.csv".format(cvae.dropout_rate_input, cvae.dropout_rate_hidden, cvae.freeze_weights, i)
	history_df.to_csv(filename, sep=',')
	i+=1

print('5-Fold results: {}'.format(scores))
print('Average accuracy: {}'.format(np.mean(scores)))


classify_df = classify_df.assign(mean_accuracy=np.mean(scores))
classify_df = classify_df.assign(intermediate_dim=hidden_dim)
classify_df = classify_df.assign(latent_dim=latent_dim)
classify_df = classify_df.assign(batch_size=batch_size)
classify_df = classify_df.assign(epochs_cvae=epochs)
classify_df = classify_df.assign(learning_rate=learning_rate)
classify_df = classify_df.assign(dropout_input=dropout_input)
classify_df = classify_df.assign(dropout_hidden=dropout_hidden)
classify_df = classify_df.assign(dropout_decoder=dropout_decoder)
classify_df = classify_df.assign(freeze_weights=freeze_weights)
classify_df = classify_df.assign(classifier_use_z=classifier_use_z)

output_filename="../parameter_tuning/cvae_tcga_classifier_dropout_{}_in_{}_hidden_cv_frozen_{}.csv".format(cvae.dropout_rate_input, cvae.dropout_rate_hidden, cvae.freeze_weights)
classify_df.to_csv(output_filename, sep=',')
'''
#################################
## Build and train final model ##
#################################

classify_df = pd.DataFrame(columns=["accuracy", "conf_matrix"])

# Prepare data to train Variational Autoencoder (merge dataframes and normalize)
X_autoencoder = pd.concat([X_brca_train, X_tcga_no_brca], sort=True)
X_brca_tumor_type = pd.DataFrame(data=["BRCA"]*len(X_brca_train), columns=["tumor_type"])
X_autoencoder_tumor_type = pd.concat([X_brca_tumor_type, x_tcga_type_no_brca], sort=True)

scaler = MinMaxScaler()
scaler.fit(X_autoencoder)
X_autoencoder_scaled = pd.DataFrame(scaler.transform(X_autoencoder), columns=X_autoencoder.columns)

# Scale logistic regression data
scaler.fit(X_brca_train)
X_brca_train_scaled = pd.DataFrame(scaler.transform(X_brca_train), columns=X_brca_train.columns)
X_brca_test_scaled = pd.DataFrame(scaler.transform(X_brca_test), columns=X_brca_test.columns)

X_brca_train_scaled = X_brca_train_scaled.reindex(sorted(X_brca_train_scaled.columns), axis="columns")
X_brca_test_scaled = X_brca_test_scaled.reindex(sorted(X_brca_test_scaled.columns), axis="columns")

cvae = ConditionalVAE(original_dim=X_autoencoder_scaled.shape[1], intermediate_dim=300, latent_dim=100, epochs=100, batch_size=50, learning_rate=0.001)
cvae.initialize_model()
cvae.train_cvae(train_df=X_autoencoder_scaled, 
				train_cond_df=pd.get_dummies(X_autoencoder_tumor_type), 
				val_df=pd.DataFrame(),
				val_cond_df=pd.DataFrame(),
				val_flag=False)

enc = OneHotEncoder()
y_labels_train = enc.fit_transform(y_brca_train.values.reshape(-1, 1))
y_labels_test = enc.fit_transform(y_brca_test.values.reshape(-1, 1))

tumors = X_autoencoder_tumor_type["tumor_type"].unique()

X_train_tumor_type = pd.DataFrame(0, index=np.arange(len(X_brca_train_scaled)), columns=tumors)
X_train_tumor_type["BRCA"]=1
X_test_tumor_type = pd.DataFrame(0, index=np.arange(len(X_brca_test_scaled)), columns=tumors)
X_test_tumor_type["BRCA"]=1

fit_hist = cvae.classifier.fit(x=[X_brca_train_scaled, X_train_tumor_type], 
								y=y_labels_train, 
								shuffle=True, 
								epochs=40,
								batch_size=50)

final_score = cvae.classifier.evaluate([X_brca_test_scaled, X_test_tumor_type], y_labels_test)

confusion = confusion_matrix(y_labels_test, cvae.classifier.predict([X_brca_test_scaled, X_test_tumor_type]))

classify_df = classify_df.append({"accuracy":final_score[1], "conf_matrix":confusion}, ignore_index=True)
history_df = pd.DataFrame(fit_hist.history)
history_df.to_csv("../parameter_tuning/cvae_tcga_classifier_history_FINAL.csv", sep=',')
i+=1

print('FINAL ERROR: {}'.format(final_score[0]))
print('ACCURACY: {}'.format(final_score[1]))

classify_df = classify_df.assign(intermediate_dim=cvae.intermediate_dim)
classify_df = classify_df.assign(latent_dim=cvae.latent_dim)
classify_df = classify_df.assign(batch_size=cvae.batch_size)
classify_df = classify_df.assign(epochs_vae=cvae.epochs)
classify_df = classify_df.assign(learning_rate=cvae.learning_rate)

output_filename="../parameter_tuning/cvae_tcga_classifier_FINAL.csv"
classify_df.to_csv(output_filename, sep=',')
'''

