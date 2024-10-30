# postprocessing directory

This directory contains the scripts to analyse the performance of the trained VAE model (see train-vae dir): make predictions of te test data.

The principle of the VAE for damage detection:
- The VAE is trained to reproduce its input image, after the input image has passed through the encoder and decoder. The input 'image' is a small cut-out window of a satellite image of Antarctic ice shelf.
- The features of the ice shelf, including damage (fracture) features, are represented in the latent space of the VAE.
- The location (cluster) of values of fracture features in the latentspace are identified using label data.
- To generate damage predictions, the satellite image is split into cut-out windows and fed to the VAE encoder. The cut-out windows located in the desired cluster in the latentspace are given a positive (1) damage prediction; the others a negative (0). The predictions for each cut-out windows are then reassembled to form a 2D prediction map.


## Files:
- plot_history_loss.py          : plots figure of traininig loss per epoch. Expects 1 file with loss values per epoch.
- plot_tiles_latentspace.py     : load encoded data / encode a tile in the latent space; plot and save latent space correlation (for latent space dimensions 2 or 4) to analyse clustering.
- predict_tiles_from_cluster.py : predict damage with trained VAE of which latentspace clusters are identified. Currently implemented for the clusters locations identified in model `model_1684233861_L2_w20_k5_f16_a20`.
- predict_tiles.slurm           : submit predict_tiles_from_cluster as slurm job.
- trained_model_embed_data_incl_labels.py : encode tiles from training and testset for every trained epoch of the VAE model, as well as their labels. Save encoded latent_space_cutouts.npy files for later use.
