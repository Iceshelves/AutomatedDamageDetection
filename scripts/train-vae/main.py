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

import pandas as pd


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
    #
    bands = [int(i) for i in config['bands'].split(" ")]
    sizeCutOut = int(config['sizeCutOut'])
    sizeStep = int(config['sizeStep'])
    stride = int(config['stride'])
    #DATA
    # balanceRatio = float(config['balanceRatio'])
    file_DMGinfo = config['tiledDamagePixelsCountFile']
    normThreshold = float(config['normalizationThreshold'])
    # MODEL
    filter1 = int(config['filter1'])
    filter2 = int(config['filter2'])
    kernelSize1 = int(config['kernelSize1'])
    kernelSize2 = int(config['kernelSize2'])
    denseSize = int(config['denseSize'])
    latentDim = int(config['latentDim'])
    #vae:
    alpha = 5
    batchSize = int(config['batchSize'])
    nEpochMax = int(config['nEpochData'])
    nEpochTrain = int(config['nEpochTrain'])
#     validationSplit = float(config['validationSplit'])

    return (catPath, labPath, outputDir, sizeTestSet, sizeValSet, roiFile,
            bands, sizeCutOut, nEpochMax, nEpochTrain, sizeStep, stride, file_DMGinfo, normThreshold,
            filter1, filter2, kernelSize1,kernelSize2, denseSize, latentDim,
            alpha, batchSize)


def main(config=None):

    # parse input arguments
    config = config if config is not None else "train-vae.ini"
    catPath, labPath, outputDir, sizeTestSet, sizeValSet, roiFile, bands, \
        sizeCutOut, nEpochmax, nEpochTrain, sizeStep, stride, file_DMGinfo, normThreshold, \
        filter1, filter2, kernelSize1, kernelSize2, denseSize, latentDim, \
        alpha, batchSize = parse_config(config)
    
    # # to do: implement training approach
    # if epoch_type == 'data_epoch': # TO DO: add to config
    #     # loop all data once before next epoch
    #     n_train_epochs = 1; # TO DO: add to config
    # elif epoch_type == 'train_epoch'

    # using datetime module for naming the current model, so that old models
    # do not get overwritten
    ct = datetime.datetime.now()  # ct stores current time
    ts = ct.timestamp()  # ts store timestamp of current time

    # use the ice-shelves extent as mask for the initial balancing
    mask = gpd.read_file(roiFile)

    # # split tiles in training, validation and test sets
    # train_set_paths, val_set_paths, test_set_paths = \
    #     tiles.split_train_and_test(
    #         catalog_path=catPath,
    #         test_set_size=sizeTestSet,
    #         validation_set_size=sizeValSet,
    #         labels_path=labPath,
    #         random_state=42  # random state ensures same data sets for each run
    #     )
    
    # split tiles in training, validation and test sets 
    train_set_paths, val_set_paths, test_set_paths = \
        tiles.split_train_and_test(
                 catalog_path=catPath, 
                 dmg_px_count_file=file_DMGinfo,  
                 select_dmg_quantile=0.9,
                 validation_split=0.2,
                 random_state=42)

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
                            sizeCutOut,len(bands),
                            filter1,filter2,
                            kernelSize1,kernelSize2,
                            denseSize,latentDim)
    decoder = VAE.make_decoder(latentDim,encoder,
                               filter1,filter2,
                               kernelSize1,kernelSize2)
    vae = VAE.make_vae(encoder_inputs, z, z_mean, z_log_var, decoder,alpha)
    vae.compile(optimizer=keras.optimizers.Adam())

    path = outputDir + '/model_' + str(int(ts))
    vae.save(os.path.join(path, 'epoch_' + str(epochcounter - 1)))
    encoder.save(os.path.join(path, 'encoder_epoch_' + str(epochcounter - 1) ))

    # begin loop
    while epochcounter <= nEpochmax:
        offset = (epochcounter - 1)*sizeStep
        while offset >= sizeCutOut: # add offset correction for when offset >= window_size
            offset = offset - sizeCutOut 

        train_set = dataset.Dataset(train_set_paths, sizeCutOut, bands,
                                    offset=offset, 
                                    stride=stride,
                                    shuffle_tiles=True,
                                    norm_threshold=normThreshold)
        train_set.set_mask(mask.unary_union, crs=mask.crs)
        train_set_tf = train_set.to_tf()
        # train_set_tf = train_set_tf.shuffle(buffer_size=2000000)
        train_set_tf = train_set_tf.batch(64, drop_remainder=True)

        history = vae.fit(train_set_tf, epochs=nEpochTrain, validation_data=val_set_tf)
#         history = vae.fit(x_train, epochs=1, batch_size=batch_size, validation_split=train_val_split) # validatio_split does not work because we loop all data every epoch (and then its not independent validatoin data)

        # print('Total loss: {} \n Reconstructiion loss: {} \n K-loss: {}'.format(total_loss, reconstruction_loss, kl_loss)) #
        print('losses: {}'.format(vae.losses) )

        # change it: make a call to os to create a path
        vae.save(os.path.join(path, 'model_epoch_' + str(epochcounter)))
        encoder.save(os.path.join(path, 'encoder_epoch_' + str(epochcounter - 1) ))
        
        # save history dict to jsson
        hist_df = pd.DataFrame(history.history)  
        hist_json_file = os.path.join(path, 'history_epoch_' + str(epochcounter - 1) )
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)

        epochcounter = epochcounter + 1


if __name__ == "__main__":
    # retrieve config file name from command line
    config = sys.argv[1] if len(sys.argv) > 1 else None
    main(config)
