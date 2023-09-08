# TCR_clustering

The following files are the Contrastive Learning Model for T-cell receptor clustering according to their antigen specificity.
The project includes the following files:
 - main_triplet.py  - the main file to train the contrastive learning model on MHCpTCR dataset (through Unix-based interface)
 - main_autoencoder.py - train TCR autoencoder on a large repertoire data to transfer the trained layers to the contrastive learning model (through Unix-based interface )
 - config_script.py - a file to access training files and folders for and after training
 - enc_decode.py - a file to one-hot encode and decode TCR amino acid sequences
 - model_architecture.py - convolutional neural network and feed-forward neural network architecture for autoencoder and contrastive learning

This is an example of how to use the main_triplet.py file:
python main_triplet.py --model_type CNN --input_type beta --embedding_space 32 --epochs 100 --batch_size 128 --learning_rate 0.001 --triplet_mode naive --patience 20 --triplet_loss semihard --plot_embedding False

Note, that this project has not been published yet, so the TCR data, model analysis and results can't be made publicly accessible yet.
