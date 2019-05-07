#!/bin/bash
INPUT=param_config/params.csv
OLDIFS=$IFS
IFS=,
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
{
	read
	while read -r hidden_dim latent_dim batch_size epochs learning_rate dropout_input dropout_hidden dropout_decoder freeze_weights classifier_use_z reconstruction_loss
	do
		echo hidden_dim=$hidden_dim 
		echo latent_dim=$latent_dim 
		echo batch_size=$batch_size 
		echo epochs=$epochs 
		echo l_rate=$learning_rate 
		echo d_in=$dropout_input 
		echo d_hidden=$dropout_hidden 
		echo d_decoder=$dropout_decoder 
		echo freeze=$freeze_weights 
		echo use_z=$classifier_use_z 
		echo rec_loss=$reconstruction_loss
		python src/evaluate_vae_tcga.py --hidden_dim=$hidden_dim --latent_dim=$latent_dim --batch_size=$batch_size --epochs=$epochs --l_rate=$learning_rate --d_in=$dropout_input --d_hidden=$dropout_hidden --d_decoder=$dropout_decoder --freeze=$freeze_weights --use_z=$classifier_use_z --rec_loss=$reconstruction_loss
	done
} < $INPUT
IFS=$OLDIFS
