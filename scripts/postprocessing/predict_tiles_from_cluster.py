
import os
import numpy as np
import tensorflow as tf
import glob
import geopandas as gpd
import json
import configparser 
from sklearn.manifold import TSNE
import rioxarray as rioxr
import xarray as xr
# from shapely import geometry
from rasterio.features import shapes, geometry_mask
import pathlib
import pandas as pd
import xarray as xr
import sys

import matplotlib.pyplot as plt
from skimage import exposure as skimage_exposure
import rasterio as rio

# # homedir = '/Users/tud500158/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - TUD500158/'
# workdir = '/gpfs/home3/mizeboud/' #os.path.join(homedir,'github/AutomatedDamageDetection/')
# os.chdir(os.path.join(workdir,'preprocessing/scripts/train-vae/'))
# import dataset
# import tiles as ts

from joblib import Parallel, delayed

print('\n')

def parse_config(config):
    """ Parse input arguments from dictionary or config file """
    if not isinstance(config, dict):
        parser = configparser.ConfigParser(allow_no_value=True)
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
    sizeStep = int(config['sizeStep'])
    stride = int(config['stride'])
    #DATA
    # balanceRatio = float(config['balanceRatio'])
    # file_DMGinfo = config['tiledDamagePixelsCountFile']
    # normThreshold = [float(i) for i in config['normalizationThreshold'].split(" ")]
    normThreshold = config['normalizationThreshold']
    if normThreshold is not None:
        normThreshold = [float(i) for i in normThreshold.split(" ")]
    adaptHist = True if config['adaptHistogramEqual'] == 'True' else False
    

    return (catPath, labPath, outputDir, 
            bands, sizeCutOut, normThreshold, adaptHist,
            )


''' ------------
FUNCTIONS
--------------- '''

def normalise_and_equalise(da,normThreshold=None,equalise=False):
    
    # normalize
    if normThreshold is not None:
        da = (da + 0.1) / (normThreshold + 1)
        da = da.clip(max=1)
    
    if equalise:
        # hist equalist
        n_bands = da['band'].shape[0]
        all_band_eq=np.empty(da.shape)

        for band_i in range(n_bands): # perform adaptive normalisation per band
            band_data = da.isel(band=band_i)
            band_data_eq = skimage_exposure.equalize_adapthist(band_data, clip_limit=0.03)
            all_band_eq[band_i] = np.expand_dims(band_data_eq,axis=0)

        da = da.copy(data=all_band_eq) # overwrite data in dataArray
    
    return da
    
def create_cutouts2(da,cutout_size, normThreshold=None, equalise=False):

    # generate windows
    da = da.rolling(x=cutout_size, y=cutout_size)
    da = da.construct({'x': 'x_win', 'y': 'y_win'}, stride=cutout_size)

    # drop NaN-containing windows
    da = da.stack(sample=('x', 'y'))
    da = da.dropna(dim='sample', how='any')

    # tile_cutouts = da.data.transpose(3, 1, 2, 0) # samples, x_win, y_win, bands: (250000, 20, 20, 3)
    # tile_cutouts_da = da.transpose('sample','x_win','y_win','band')

    tile_cutouts_da = da.transpose('sample','x_win','y_win' ,...) # transpose dimensionos; dataArray does nto necessarily need 'band' as dimension
    tile_cutouts = tile_cutouts_da.data
    
    return tile_cutouts, tile_cutouts_da

def mask_data(data, mask_file):
    mask_poly = gpd.read_file(mask_file).to_crs(epsg=3031)
    # gdf = mask_poly.unary_union 
    # mask = data #.copy(data=np.ones_like(data.values)) # set up img with only 1 vluess
    masked_data = data.rio.clip(mask_poly.unary_union, mask_poly.crs, drop=False, invert=False) # mask (raster)
    return masked_data

  
def plot_latentspace_clusters( embedded_data,labels, clabel,figsize=(8,8) ):    
    marksize = 2
    if len(labels.shape)>1:
        labels = np.squeeze(labels)
    
    # sort data with label value
    embedded_data = embedded_data[ np.argsort(labels),:] #,np.argsort(labels)]
    labels = np.sort(labels,axis=0)
    
    # split data in points with (no) label value
    z0 = embedded_data[:, 0]
    z1 = embedded_data[:, 1]
    
    fig, ax1 = plt.subplots(figsize=figsize )
    s1 = ax1.scatter(z0, z1, c=labels, s=marksize, cmap='cividis_r') 
    ax1.set_xlabel("z[0]"); 
    ax1.set_ylabel("z[1]");
    fig.colorbar(s1,ax=ax1,label=clabel); 
    
    return fig


def plot_latentspace_clusters_3d( embedded_data,labels, clabel='label' ,figsize=(8,8)):    
    marksize = 2
    if len(labels.shape)>1:
        labels = np.squeeze(labels)
    
    # sort data with label value so that labelled pixes will be on top
    embedded_data = embedded_data[ np.argsort(labels),:] #,np.argsort(labels)]
    labels = np.sort(labels,axis=0)
    
    # split data in points with (no) label value
    z0 = embedded_data[:, 0]
    z1 = embedded_data[:, 1]
    z2 = embedded_data[:, 2]
    
    # fig, ax = plt.subplots(1, figsize=figsize )
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    s1 = ax.scatter(z0, z1, z2, c=labels, s=marksize, cmap='cividis_r') # also add size for scatter point
    ax.set_xlabel("z[0]"); 
    ax.set_ylabel("z[1]");
    fig.colorbar(s1,ax=ax,label=clabel,fraction=0.045,shrink=0.6); 
    return fig, ax


def embed_latentspace_2D(encoded_data,
                         latent_dim,perplexity=10, 
                         n_iter=1000,
                         n_iter_without_progress=300):
    ''' Consider using perplexity between 5 and 50 (default 30); larger datasets usually require a larger perplexity.
        n_iter default 1000, should be at least 250
        learning_rate: the learning rate for t-SNE is usually in the range [10.0, 1000.0], default=200; 
        The ‘auto’ option sets the learning_rate to max(N / early_exaggeration / 4, 50)
    '''
    
    z_mean_2D = TSNE(n_components=2,
                     perplexity=perplexity,
                     n_iter=n_iter,
                     learning_rate='auto',
                     init='pca',
                     n_iter_without_progress=n_iter_without_progress).fit_transform(encoded_data);

    return z_mean_2D


''' -------------

   Tile processing function
   
-----------------'''

def predict_and_save_tile( tiles_path, tileNum, 
                            encoder, model_dir, epoch_num, 
                            bands , cutout_size, normThreshold, adaptHist,
                            path2save='./'  ) :

    ## filename of current tile
    tile_file = os.path.join(tiles_path,'S2_composite_2019-11-1_2020-3-1_tile_' + str(tileNum) + '.tif')
    tileName = tile_file.split("/")[-1][:-4] # vb: 'S2_composite_2019-11-1_2020-3-1_tile_124'
    
    ## Filename of prediction
    tilePredict_fileName = '{}_model_{}_epoch{}_predict.tif'.format(tileName,model_dir.split('_')[1],epoch_num)
    
    ## code below moved to main()
    # if os.path.exists( os.path.join(path2save, tilePredict_fileName )):
    #     print('Already predicted tile {}; continue'.format(tileNum))
    #     # continue
    # else: 
    #     print('----\n Processing ' + tile_file )

    #     if not os.path.isfile(tile_file):
    #         print('.. No tile found for {}; continue\n--'.format(tile_file))
    #     else:
    #         process_image( tile_file, tilePredict_fileName,
    #             encoder, model_dir, epoch_num, 
    #             bands , cutout_size, normThreshold, adaptHist,
    #             path2save=path2save  )
    
    return tile_file, tilePredict_fileName

# def define_prediction_filename(image_filepath, model_dir, epoch_num ):
#     # tile_file = os.path.join(tiles_path,'S2_composite_2019-11-1_2020-3-1_tile_' + str(tileNum) + '.tif')
#     # tileName = image_filepath.split("/")[-1][:-4] # vb: 'S2_composite_2019-11-1_2020-3-1_tile_124'
#     image_filename = image_filepath.stem # /path/to/img_file.tif --> img_file
#     model_id = model_dir.split('_')[1] # model_1684233861_L2_w20_k5_f16_a20... --> model_1684233861
    
#     predict_fileName = '{}_model_{}_epoch{}_predict.tif'.format(image_filename,model_id,epoch_num)
    
            
def process_image( image_file, predict_fileName,
                    encoder, model_dir, epoch_num, 
                    bands , cutout_size, normThreshold, adaptHist,
                    path2save='./'  ) :
    
    ''' ----------
    Create cut-outs
        Actually read the tile, make cutouts, linked with labeldata
    ------------'''
    
    # if not os.path.isfile(image_file):
    #     print('----\n No tile found for {}; continue'.format(tileNum))
    #     # continue
    # else:

    with rioxr.open_rasterio(image_file).astype("float32") as da:

        # select bands
        if bands is not None:
            if type(bands) is not list:
                da = da.sel(band=[bands])
            else:
                da = da.sel(band=bands)


        ''' ----------
        Mask ocean
        - not required for predicting -- so do lateron 
        ------------'''

        # mask/clip: if self.mask is not None: [not needed for prediction]
        # da = mask_data(da,oceanmask_file)


        ''' ----------
        Normalise and Equalise
        ------------'''

        da = normalise_and_equalise(da,normThreshold=normThreshold[0],equalise=adaptHist)
        print('.. normalised data')

        ''' ----------
        Create cut-outs
        - First fill NaN values so that they are not dropped during cut-out construction
          This is important to be able to reconstruct spatial image after prediction.
        - Store a mask of the NaN values so that they can be returned later
        ------------'''
        ## Check if data contains a general no-data value instead of NaN
        # (included for exported S2_MGRS tiles)
        da = da.where(da!=-999, other=np.nan)

        ## Create mask -- by omitting the ocean-mask step, there are no NaN values expectedi n the data. 
        # We could mask them, but then also need to aggregate them to window-values to be able to put them on the labels later. Seems like unnecessary steps if it is not needed.
        if np.isnan(da).any():
            mask_nan = np.isnan(da).sum(dim='band') # .astype(bool)
            da = da.fillna(-999)
            # raise ValueError('Tiledata has NaN values -- need to mask these')
            mask_nan_cutouts = create_cutouts2(mask_nan,cutout_size).sum(axis=1).sum(axis=1) # has value > 0 if any px in window has a NaN value
            mask_nan_pred = mask_nan_cutouts.where(mask_nan_cutouts == 0, 1).astype(bool) # yields True (1) for nan-containing windows
        else:
            mask_nan_pred = None

        # generate windows -- cut 
        _, tile_cutouts_da = create_cutouts2(da,cutout_size) # samples, x_win, y_win, bands: (250000, 20, 20, 3)
        # print('cutouts {} '.format(tile_cutouts.shape))


        ''' ----------
        Encode data
        ------------'''

        encoded_data,_,_ = encoder.predict(tile_cutouts_da.data);

        z0 = encoded_data[:,0]
        z1 = encoded_data[:,1]

        ''' ----------------------------------------
        Predict cluster type for all pixels: Threshold
        -----------------------------------------------'''

        ## Sort latent space values so can threshold values of Z1 based on Z0
        i_sort = np.argsort(z0)
        z1_sorted = z1[i_sort]
        z0_sorted = z0[i_sort]

        # set up sequence for x-axis
        z0_seq = np.linspace(np.nanmin(z0),np.nanmax(z0),len(z0) )

        ## Define cluster (all samplesa bove threshold) 

        if model_dir == 'model_1684233861_L2_w20_k5_f16_a20' and epoch_num == '9': # for model_1684233861 (16 may 23)
            dzdz = (0.005)/(1)
            dzdz = (0.06)/2
            c = 0.01 #0 
            z0_seq = np.linspace(np.nanmin(z0),np.nanmax(z0),len(z0) )
            z1_treshold_val = dzdz*z0_seq + c 

            # -- extract samples in cluster
            # idx_cluster_sorted = np.less(z1_sorted, z1_treshold_val).astype(int)
            idx_cluster_sorted = np.greater(z1_sorted, z1_treshold_val).astype(int)
            print('Extract samples GREATER THAN line: Z1 = {:.2f}*Z0 + {:.2f}'.format( dzdz, c ) ) 

        else:
            raise ValueError('Cluster extraction method is not defined for specified model {} -- update code', model_dir)

        ## -- Reverse sort of latent space to recover spatial relations of samples
        idx_cluster = idx_cluster_sorted[np.argsort(i_sort)] # (Nsamples,)


        ''' ----------
        Put predicted labels in dataArray; plot and save
        ------------'''

        ## place the cluster indices in an xarray and unstack to spatial img
        idx_cluster_da = tile_cutouts_da.isel(band=0,x_win=0,y_win=0).copy(deep=True, data=idx_cluster) # (Nsamples)
        cluster_da     = idx_cluster_da.unstack().transpose('y','x') # (Nsamples) --> (x, y) 

        ## Re-apply nan mask
        if mask_nan_pred:
            cluster_da = cluster_da.where(~mask_nan,np.nan)

        # ## Plot
        # fig,ax=plt.subplots(figsize=(6,5))
        # cluster_da.plot.imshow(ax=ax,label=''); ax.set_axis_off(); ax.set_title('');

        ### Save
        cluster_da = cluster_da.astype('uint8') # convert binary data to byte (otherwise loading error from GCS to GEE)
        cluster_da.rio.to_raster(  os.path.join(path2save, predict_fileName ), driver="COG") # use CloudOptimisedGeotiff so it can be used together with GEE
        print('Save predicted tile to {}'.format(predict_fileName))
        # print('----')

''' -------------

   MAIN
   
-----------------'''
def main(encoder_dir, data_dir=None):
    
    print('---- Config -----')
    
    ''' ----------
    Get model to load
    ------------'''
    
    if encoder_dir is None:
        raise NameError('No encoder specified. Run script as "python this_script.py /path/to/model_dir/encoder_dir"')
    
    if encoder_dir[-1] == '/': # trailing slash gives an error somewhere lateron; remove
        encoder_dir = encoder_dir[:-1]
    # Update relative path to full path if needed
    if os.path.isdir(encoder_dir):
        path_to_encoder_epoch = encoder_dir
    else:
        path_to_encoder_epoch = os.path.join(os.path.expanduser('~'), encoder_dir)
    
    ## Retrieve master directory and its name
    path_to_model = os.path.split(path_to_encoder_epoch)[0]
    model_dir = os.path.basename(path_to_model)
    
    ''' ----------
    Get path to data to process; define path to save predictions
    --------------'''
    if data_dir is None:
        data_dir = '/projects/0/einf512/S2_composite_2019-11-1_2020-3-1'
        print('.. No path to data specified; assuming same path as train-data: {}'.format(tiles_path) )
    else:
        print('.. Reading data from     {}'.format(data_dir))

    if data_dir[-1] == '/': # trailing slash gives an error somewhere lateron; remove
        data_dir = data_dir[:-1]
            
    ## Define path to save data
    data_identifier = data_dir.split('/')[-1]
    path2save = os.path.join('/projects/0/einf512/VAE_predictions/',data_identifier)
    if not os.path.isdir(path2save):
        print('.. Creating savepath directory')
        os.makedirs(path2save, exist_ok=False)
    
    # path2save = '/projects/0/einf512/VAE_predictions/S2_composite_2019-11-1_2020-3-1/'
    print('.. Save predictions to:  {}'.format(path2save) )
    tiles_path = data_dir

    ''' ----------
    Parse model input arguments
    ------------'''
    # config = glob.glob(path_to_traindir + "train-vae.ini")
    # config = os.path.join(path_to_model,'train-vae.ini')
    config = glob.glob(os.path.join(path_to_model,'*.ini'))
    
    if not config:
        raise ValueError('Did not find configfile in directory', path_to_model)
    
    catPath, labPath, outputDir, \
            bands, cutout_size, normThreshold, adaptHist = parse_config(config)
    
    ''' ----------
    Load model/encoder
    ------------'''

    print('.. Loaded encoder       {}'.format(path_to_encoder_epoch) )
    
    epoch_num = path_to_encoder_epoch.split('_')[-1]
    encoder = tf.keras.models.load_model(path_to_encoder_epoch,compile=False) # compile=True does not work
    
    # Get latent_dim (of sampling layer)
    latent_dim = encoder.layers[-1].output_shape[-1] 
    
    
    
    ''' ----------
    Get files to process
    --------------'''
    
    file_list = glob.glob(os.path.join(data_dir, '*.tif'))
    
    tile_filelist = [fname for fname in file_list if '_tile_' in fname]
    img_filelist = [fname for fname in file_list if not '_tile_' in fname]
    
    print('.. Tile files in dir: {}, other files: {}'.format(len(tile_filelist), len(img_filelist) ) )
    
    if tile_filelist: # if non-empty list: process tile_files

        # TRAINIING tilenums
        # tile_nums = [102,110,114,123,124,139,140,142,205,206,214,228,238,250,268,273,28,282,285,291,301,302,307,50,68,93]
        # tile_nums = [28, 50,53,123,124,140] # 68, 102,110,114,123,124]
        # tile_nums = [28, 50, 140] # with new manual AND nerd-labels
        # tile_nums = [140] # with new manual AND nerd-labels

        # TEST tilenums
        #tile_nums = [0, 3, 4, 42, 51, 52, 53, 54, 55, 61, 124, 140, 276, 285, 286] # all tiles in DJF 19-20 tht have mnual labels

        # ALL tiles
        tile_nums = np.arange(58,63) # (0,313)

        # Parallel(n_jobs=5)(delayed( predict_and_save_tile)( tiles_path, tileNum, 
        #                         encoder, model_dir, epoch_num, 
        #                         bands , cutout_size, normThreshold, adaptHist,
        #                         path2save=path2save )
        #                     for tileNum in tile_nums )

        print('---- Processing -----')

        for tileNum in tile_nums:
            
            tile_file, tilePredict_fileName = predict_and_save_tile( tiles_path, tileNum, 
                                encoder, model_dir, epoch_num, 
                                bands , cutout_size, normThreshold, adaptHist,
                                path2save=path2save  )
            
            if os.path.exists( os.path.join(path2save, tilePredict_fileName )):
                print('Already predicted tile {}; continue'.format(tileNum))
                # continue
            else: 
                print('----\n Processing ' + tile_file )

                if not os.path.isfile(tile_file):
                    print('.. No tile found for {}; continue\n--'.format(tile_file))
                else:
                    process_image( tile_file, tilePredict_fileName,
                        encoder, model_dir, epoch_num, 
                        bands , cutout_size, normThreshold, adaptHist,
                        path2save=path2save  )

            

    elif img_filelist:
        print('---- Processing -----')
         
        for image_filepath in img_filelist:
            
            image_filename = pathlib.Path(image_filepath).stem # /path/to/img_file.tif --> img_file
            model_id = model_dir.split('_')[1] # model_1684233861_L2_w20_k5_f16_a20... --> model_1684233861

            predict_fileName = '{}_model_{}_epoch{}_predict.tif'.format(image_filename,model_id,epoch_num)
            
            if os.path.exists( os.path.join(path2save, predict_fileName )):
                print('Already predicted file {}; continue'.format(image_filename))
                continue
           
            print('----\n Processing ' + image_filename )

            
            process_image( image_filepath, predict_fileName,
                    encoder, model_dir, epoch_num, 
                    bands , cutout_size, normThreshold, adaptHist,
                    path2save= path2save  ) 
            
        
        
    print('----\nDone')

if __name__ == '__main__':
    #  Run script as "python path/to/script.py /path/to/model/encoder_dir"
    #  Run script as "python path/to/script.py /path/to/model/encoder_dir /path/to/data/to/predict"
        
    # retrieve config filename from command line
    encoder_dir = sys.argv[1] if len(sys.argv) > 1 else None
    data_dir = sys.argv[2] if len(sys.argv) > 2 else None

    # run script
    main(encoder_dir, data_dir)   


