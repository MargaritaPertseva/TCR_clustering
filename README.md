# TCR_clustering

The following files are the Constrastive Learning with Triplet Loss - based algorithm for T-cell receptors clustering according to their antigen specificity.
The project includes following files:
 - main_triplet.py  - the main file for model training through Unix-based interface
 - config_script.py - a file to access training files and folders for and after training
 - enc_decode.py - a file to one-hot encode and decode TCR amino acid sequences
 - model_architecture.py - convolutional neural network and feed-forward neural network architecture
 - plot_triplet_emb.py  - files to visualise the trained embeddings on a test dataset
 
This is the example of how to use main_triplet.py file:
python main_triplet.py --model_type CNN --input_type beta --data_size small --embedding_space 32 --epochs 200 --batch_size 256 --learning_rate 0.001 --triplet_mode naive --patience 20 --triplet_loss semihard --plot_embedding False

This project is still no publised, so the TCR data can't be made publicly accessible yet.
