# Automated Damage Detection
Prototype for automated damage detection in Sentinel 1 imaging of iceselves using unsupervised learning.

The prototype makes use of a VAE architecture, and employs a bespoke input genertation framework  to create cutouts for network training from larger imaging tiles.  
Notebooks have been used for intial prototyping. These have been converted to scripts for a first production prototype.


## Contents
Within each directory another README file is included provide more details

- files : contains list of data tiles used for training/testing
- notebooks : contains python notebooks for training, pre- and post-processing
- scripts : contains python scripts for traininig, pre- and post-processing
- training: contains one of of the trained VAE models in a .zip, and an overview of the trained models that are available at  /projects/0/einf512/trained_models/
