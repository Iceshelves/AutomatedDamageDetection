#!/usr/bin/env python
# coding: utf-8

import configparser
import datetime
import json
import os
import sys

import geopandas as gpd
from tensorflow import keras

import VAE
import dataset
import tiles


def parse_config(config):
    """ Parse input arguments from dictionary or config file """
    if not isinstance(config, dict):
        parser = configparser.ConfigParser()
        parser.read(config)
        config = parser["train-VAE"]

    catPath = config['catalogPath']
    labPath = config['labelsPath']
    outputDir = config['outputDirectory']
    sizeTestSet = int(config['sizeTestSet'])
    sizeValSet = int(config['sizeValidationSet'])
    roiFile = config['ROIFile']
    bands = [int(i) for i in config['bands'].split(" ")]
    sizeCutOut = int(config['sizeCutOut'])
    nEpochMax = int(config['nEpochMax'])
    sizeStep = int(config['sizeStep'])
    #DATA
    balanceRatio = float(config['balanceRatio'])
    normThreshold = float(config['normalizationThreshold'])    
    # MODEL
    filter1 = int(config['filter1'])
    filter2 = int(config['filter2'])
    kernelSize = int(config['kernelSize'])
    denseSize = int(config['sizeCutOut'])
    latentDim = int(config['latentDim'])
    #vae:
    alpha = 5
    batchSize = int(config['batchSize'])
#     validationSplit = float(config['validationSplit'])
        
    return (catPath, labPath, outputDir, sizeTestSet, sizeValSet, roiFile,
            bands, sizeCutOut, nEpochMax, sizeStep, balanceRatio, normThreshold,
            filter1, filter2, kernelSize, denseSize, latentDim, 
            alpha, batchSize)


def main(config=None):

    # parse input arguments
    config = config if config is not None else "train-vae.ini"
    catPath, labPath, outputDir, sizeTestSet, sizeValSet, roiFile, bands, \
        sizeCutOut, nEpochmax, sizeStep, balanceRatio, normThreshold, \
        filter1, filter2, kernelSize, denseSize, latentDim, \
        alpha, batchSize = parse_config(config)
    

    # using datetime module for naming the current model, so that old models
    # do not get overwritten
    ct = datetime.datetime.now()  # ct stores current time
    ts = ct.timestamp()  # ts store timestamp of current time

    # use the ice-shelves extent as mask for the initial balancing
    mask = gpd.read_file(roiFile)

    # split tiles in training, validation and test sets
    train_set_paths, val_set_paths, test_set_paths = \
        tiles.split_train_and_test(
            catalog_path=catPath,
            test_set_size=sizeTestSet,
            validation_set_size=sizeValSet,
            labels_path=labPath,
            random_state=42  # random state ensures same data sets for each run
        )
    
    # save dataset compositions
    path = outputDir + '/datasets_{}.json'.format(int(ts))
    with open(path, "w") as f:
        json.dump(fp=f, indent=4, obj=dict(training=train_set_paths,
                                           validation=val_set_paths,
                                           test=test_set_paths))

    # I think we could also balance the test set using the mask, t.b.h. as it
    # is a preselection step across all data
    test_set = dataset.Dataset(test_set_paths, sizeCutOut, bands,
                               shuffle_tiles=True,
                               norm_threshold=normThreshold)
    test_set_tf = test_set.to_tf()
    test_set_tf = test_set_tf.batch(64, drop_remainder=True)

    # balanced validation set (i.e. apply mask)
    val_set = dataset.Dataset(val_set_paths, sizeCutOut, bands,
                              shuffle_tiles=True,
                              norm_threshold=normThreshold)
    val_set.set_mask(mask.unary_union, crs=mask.crs)
    val_set_tf = val_set.to_tf()
    val_set_tf = val_set_tf.batch(64, drop_remainder=True)

    # Loop and feed to VAE
    epochcounter = 1  # start at 0 or adjust offset calculation

#     encoder_inputs, encoder, z, z_mean, z_log_var = VAE.make_encoder()
#     decoder = VAE.make_decoder()
#     vae = VAE.make_vae(encoder_inputs, z, z_mean, z_log_var, decoder)
#     vae.compile(optimizer=keras.optimizers.Adam())
    
    encoder_inputs, encoder, z, z_mean, z_log_var = VAE.make_encoder(
                            cutout_size,len(bands),
                            filter_1,kernel_size,
                            filter_2,dense_size,latent_dim)
    decoder = VAE.make_decoder(latent_dim,filter_1,filter_2,kernel_size)
    vae = VAE.make_vae(encoder_inputs, z, z_mean, z_log_var, decoder,alpha)
    vae.compile(optimizer=keras.optimizers.Adam())
    
    path = outputDir + '/model_' + str(int(ts))
    vae.save(os.path.join(path, 'epoch_' + str(epochcounter - 1)))
    encoder.save(os.path.join(path, 'encoder_epoch_' + str(epochcounter - 1) ))
    
    # begin loop
    while epochcounter < nEpochmax:
        offset = (epochcounter - 1)*sizeStep
 
        train_set = dataset.Dataset(train_set_paths, sizeCutOut, bands,
                                    offset=offset, shuffle_tiles=True,
                                    norm_threshold=normThreshold,
                                    balance_ratio = balanceRatio)
        train_set.set_mask(mask.unary_union, crs=mask.crs)
        train_set_tf = train_set.to_tf()
        train_set_tf = train_set_tf.shuffle(buffer_size=2000000)
        train_set_tf = train_set_tf.batch(64, drop_remainder=True)

#         vae = keras.models.load_model(
#             os.path.join(path, 'epoch_' + str(epochcounter - 1))
#         )
    
        history = vae.fit(train_set_tf, epochs=1, validation_data=val_set_tf) 
#         history = vae.fit(x_train, epochs=1, batch_size=batch_size, validation_split=train_val_split) # validatio_split does not work because we loop all data every epoch (and then its not independent validatoin data)

        # change it: make a call to os to create a path
        vae.save(os.path.join(path, 'model_epoch_' + str(epochcounter)))
        encoder.save(os.path.join(path, 'encoder_epoch_' + str(epochcounter - 1) ))
        history.save(os.path.join(path , 'history_epoch_'  + str(epochcounter - 1) ))
        
        epochcounter = epochcounter + 1

    
if __name__ == "__main__":
    # retrieve config file name from command line
    config = sys.argv[1] if len(sys.argv) > 1 else None
    main(config)
