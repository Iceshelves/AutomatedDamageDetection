# Train VAE directory

This directory contains the main scripts to read Sentinel-2 data and train an
unsupervised VAE model to predict damage (ice fractures) in an image.

## Main steps:
1. Load data across Antarctic domain, ordered in regular sized tiles
2. Split tiles in (semi-balanced) train/test/validation.
3. Create cut-outs from tiles, feed to VAE
4. Training approach per epoch: 1 epoch views all training data 1 time.
  - The next epoch, re-create cut-outs of the training dataset with an offset
5. Trained VAE model and encoder are saved after every epoch.

## Files
- dataset.py: Prepare TensorFlow Dataset for training. Mask data, normalise data (adaptive Histogram normalisation), generate window cut-outs.
- main.py       : Main script to run and execute all steps (see above)
- tiles.py      : Split train and test data; Option to balance tiles with high/low amount of damage.
- train-vae.ini : config file to specify training settings
- train.slurm   : bash file to submit job to slurm cluster
- VAE.py        : Built model architecture


#### NB:
- `~/outDir` should exist beforehand (or else create it in the feed_loop)
- train.slurm should be run from another dir, because it produces the slurm log files
- the embedding code still to be added (based on the output of training models, the user decides which model to use for embedding,  so it is another script) . Or you can embed based on all models, that is a lot of plots
