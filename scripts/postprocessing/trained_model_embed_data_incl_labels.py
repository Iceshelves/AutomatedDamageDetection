'''Encode tiles from training and testset for every trained epoch of the VAE model, as well as their labels. 
Save encoded latent_space_cutouts.npy files for later use, separate files for the encoded tiledata as encoded labeldata.
Labeldata exists in two forms:
- manual (drawn by hand on training images)
- nerd (as produced bij the NeRD damage detection method, applied to the training images).
'''

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

import argparse 

import matplotlib.pyplot as plt
from skimage import exposure as skimage_exposure

# # homedir = '/Users/tud500158/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - TUD500158/'
# workdir = '/gpfs/home3/mizeboud/' #os.path.join(homedir,'github/AutomatedDamageDetection/')
# os.chdir(os.path.join(workdir,'preprocessing/scripts/train-vae/'))
# import dataset
import tiles as ts




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
    file_DMGinfo = config['tiledDamagePixelsCountFile']
    # normThreshold = [float(i) for i in config['normalizationThreshold'].split(" ")]
    normThreshold = config['normalizationThreshold']
    if normThreshold is not None:
        normThreshold = [float(i) for i in normThreshold.split(" ")]
    adaptHist = True if config['adaptHistogramEqual'] == 'True' else False
    
    # MODEL
    filter1 = int(config['filter1'])
    filter2 = int(config['filter2'])
    try:
        kernelSize1 = int(config['kernelSize1'])
        kernelSize2 = int(config['kernelSize2'])
    except KeyError: 
        kernelSize1 = int(config['kernelSize']) # old config
        kernelSize2 = kernelSize1
    denseSize = int(config['denseSize'])
    latentDim = int(config['latentDim'])
    #vae:
    alpha = float(config['alpha'])
    batchSize = int(config['batchSize'])
    try:
        nEpochMax = int(config['nEpochData'])
        nEpochTrain = int(config['nEpochTrain'])
        learnRate = float(config['learningRate'])
    except KeyError:
        nEpochMax = int(config['nEpochMax']) # old config files
        nEpochTrain = None
        learnRate = None
        

    return (catPath, labPath, outputDir, sizeTestSet, sizeValSet, roiFile,
            bands, sizeCutOut, nEpochMax, nEpochTrain, sizeStep, stride, file_DMGinfo, normThreshold, adaptHist,
            filter1, filter2, kernelSize1,kernelSize2, denseSize, latentDim,
            alpha, batchSize,learnRate)



'''
FUNCTIONS
'''

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




def parse_cla():
    """
    Command line argument parser

    Accepted arguments:
        Required:
        --year (-y)         : The script will select data for the specified year, and export only that year
        --variable (-var)   : The script selects data for the specified variable. Can be 'dmg', 'emax' and 'REMA' (atm)
        
        Optional:
        --season (-s)       : The script selects the annual data for a specific season. 
                              Options are: SON or DJF
                              Defaults to SON.

    :return args: ArgumentParser return object containing command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", help="Define path to trained model. Should be absolute", type=str)
    parser.add_argument("--label", "-l", help="Specify which type of labeldata to use, manual or NeRD-processed damage",type=str,choices=('manual','nerd','None'))
    parser.add_argument("--epoch", "-e", help="Specify the trained epoch number to process. '-1' will mean last. If not defined, all available epochs will be used",type=int)

    args = parser.parse_args()
    return args 



''' -------------

   MAIN
   
-----------------'''


def main():

    ''' ---------------------------------------------------------------------------
    Command line Configuration
    -------------------------------------------------------------- '''
    # if arguments are not specified in command line, they are set to None
    args = parse_cla()
    model_dir = args.model
    label_type = args.label
    epochs = args.epoch
    if epochs is None:
        epoch_interpret = 'all'
    elif epochs == -1:
        epoch_interpret = 'last'
    else:
        epoch_interpret = epochs
    print('--- \n Loading and encoding model: {} \n ..on epoch(s): {} \n ..with labels from: {}'.format(model_dir, epoch_interpret, label_type))
    
    if os.path.isdir(model_dir):
        path_to_model = model_dir
    else:
        path_to_model = os.path.join(os.path.expanduser('~'), model_dir)
        
        
    ''' ----------
    Parse model input arguments from config file
    ------------'''
    
    config = glob.glob(os.path.join(path_to_model,'*.ini'))
    
    catPath, labPath, outputDir, sizeTestSet, sizeValSet, roiFile, \
            bands, cutout_size, nEpochmax, nEpochTrain, sizeStep, stride, file_DMGinfo, normThreshold,adaptHist, \
            filter1, filter2, kernelSize1, kernelSize2, denseSize, latentDim, \
            alpha, batchSize,learnRate = parse_config(config)

    ''' ----------
    Get tile to process
    --------------'''
    # LOCAL
    # tile_file = os.path.join(homedir,'Data/tiles/training_tiles/S2_composite_2019-11-1_2020-3-1_tile_124.tif')
    # oceanmask_file = os.path.join(homedir,'Data/ne_10m_antarctic_ice_shelves_polys/ne_10m_antarctic_ice_shelves_polys.shp')    

    # SNELLIUS;
    oceanmask_file = roiFile
    catalog_path = catPath
    labels_path = labPath

    tiles_path = '/projects/0/einf512/S2_composite_2019-11-1_2020-3-1/'
    
    # TRAINIING tilenums
    # [102,110,114,123,124,139,140,142,205,206,214,228,238,250,268,273,28,282,285,291,301,302,307,50,68,93]
    tile_nums = [28, 50,53,123,124,140] # 68, 102,110,114,123,124]
    tile_nums = [28, 50, 140] # with new manual AND nerd-labels
    tile_nums = [140] # with new manual AND nerd-labels
    
    # TEST tilenums
    # tile_nums = [0, 3, 4, 42, 51, 52, 53, 54, 55, 61, 124, 140, 276, 285, 286] # all tiles in DJF 19-20 tht have mnual labels

    if label_type is not None:
        if label_type == 'manual':
            ''' Load and encode all tiles that have manual labeldata ''' 
            # tile 61 doesnt exist?
            # tile 0, 3, 4, 42: 0 px in raster? -- labels are of wrong date
            # tile_nums = [0, 3, 4, 42, 51, 52, 53, 54, 55, 61, 124, 140, 276, 285, 286] # all tiles in DJF 19-20 tht have mnual labels

            # tile_nums = [0, 3, 4, 42]#,51, 52, 53, 54, 55] # [102, 140] # selection for development
            # tile_nums = [51, 52, 53, 54, 55, 124, 140] # 
            # tile_nums = [0, 3, 4, 42,276, 285, 286] # to do 
            # tile_nums = [61] # doesnt exist?
            # tile_nums = [124,140,285]; # all TRAINING tiles that also have manual labels
            # tile_nums = [ 124, 140, 276, 285, 286]
            # tile_nums = [276, 285, 286]

            # TRAINIING tilenums
            # [102,110,114,123,124,139,140,142,205,206,214,228,238,250,268,273,28,282,285,291,301,302,307,50,68,93]
            # tile_nums = [102,110,114,123,124]

            ''' ----------
            Load Labels: shapefiles
            ------------'''

            # # read tile catalog to be able to get CRS and filter labels to same date
            catalog = ts._read_tile_catalog(catalog_path)
            tiles = ts._catalog_to_geodataframe(catalog)

            labels = ts._read_labels(labels_path, verbose=True)
            labels = labels.to_crs(tiles.crs)  # make sure same CRS is used. NB: at this stage, 
            
            # IF LABELS HAVE NO DATE (new version --> to update this in the shapefile): assume 2020-01-01
            # try:
            #     labels.Date
            # except AttributeError:
            #     labels['Date'] = pd.to_datetime('2020-01-01')
            idx_NaT = ~labels['Date'].notnull() # all NaT indices are 'True'
            labels.loc[idx_NaT,'Date'] = pd.to_datetime('2020-01-01')

            # select the only labels matching the tiles timespan
            labels = ts._filter_labels(labels,
                                    tiles.start_datetime.min(),
                                    tiles.end_datetime.max())

            ## Create GeoSeries from labels.geometry
            label_polys = gpd.GeoSeries(labels.geometry,crs=labels.crs) # Combine all geometries to one GeoSeries, with the 

        if label_type == 'nerd':
            ''' load and encode the dmg for a specified tileNumber '''

            # tileNum = 140;
            # tile_nums = [tileNum];
            # tile_nums = [124,140];
            # # tile_nums = [0, 3, 4, 42, 51, 52, 53, 54, 55, 61, 124, 140, 276, 285, 286] # all tiles in DJF 19-20 tht have mnual labels
            # tile_nums = [ 124, 140, 276, 285, 286]

            dmg_path = '/projects/0/einf512/NERD/damage_detection/'
            # dmg_file = os.path.join(dmg_path,'S2_composite_2019-11-1_2020-3-1_tile_' + str(tileNum) + '_300m_damageContin.tif')
            # dmg_path = os.path.join(os.path.expanduser('~'), 'data/S1_dmg/')
    
    ''' ---------
    
    Load and encode tiledata based on selected tile number(s)
    
    ------------- '''
 

    for tileNum in tile_nums:
        
        tile_file = os.path.join(tiles_path,'S2_composite_2019-11-1_2020-3-1_tile_' + str(tileNum) + '.tif')
        tileName = tile_file.split("/")[-1][:-4] # vb: 'S2_composite_2019-11-1_2020-3-1_tile_124'
        print('----\n Processing ' + tileName +'\n')

        
        ''' ----------
        Create cut-outs
            Actually read the tile, make cutouts, linked with labeldata

        Update: do not link shapefile laabldata; now read same-tile NERD output
        ------------'''

        # read tile - floats are required to mask with NaN's
        da = rioxr.open_rasterio(tile_file).astype("float32")

        # select bands
        if bands is not None:
            if type(bands) is not list:
                da = da.sel(band=[bands])
            else:
                da = da.sel(band=bands)


        ''' ----------
        Mask ocean
        ------------'''

        # mask/clip: if self.mask is not None: [removed; see _generate_cutouts]
        da = mask_data(da,oceanmask_file)


        ''' ----------
        Normalise and Equalise
        ------------'''
        
        da = normalise_and_equalise(da,normThreshold=normThreshold[0],equalise=adaptHist)


        ''' ----------
        Plot tile (for test phase)
        ------------'''
        figpath_tile = os.path.join(path_to_model, 'tile_' + str(tileNum) + '_normalised' )
        if ~os.path.exists(figpath_tile): 
            if da.max() > 1:
                print('Img is not normalised!! so plot is wrong')
            fig,ax= plt.subplots(1,figsize=(7,8))
            da.attrs['long_name']='imgbands';
            da.plot.imshow(ax=ax,rgb='band', vmin=0,vmax=1)
            ax.set_title('tile_' + str(tileNum) + ' normalised')
            ax.set_aspect('equal')
            # fig.savefig(os.path.join(path_to_traindir , model_dir, tileNum + '_normalised_histEq'))
            fig.savefig(figpath_tile)


        ''' ----------
        Create cut-outs
        ------------'''

        # generate windows -- cut 
        # TO UPDATE: create_cutouts2 drops NaN containing windows -- either fill NaN with -999 or some other value when embedding tiles for predictions
        tile_cutouts, tile_cutouts_da = create_cutouts2(da,cutout_size) # samples, x_win, y_win, bands: (250000, 20, 20, 3)
        # label_cutouts, __ = create_cutouts(tile_dmg_int, cutout_size)
        print('cutouts {} '.format(tile_cutouts.shape))


        ''' ----------
        Add labeldata to cut-outs
        ------------'''
        if label_type is not None:
            if label_type == 'nerd':
                ''' ----------
                Load Labels: processed dmg
                - load NeRD-detected damage tile and interpolate to same grid. 
                  Every pixel has a value 0 to 1
                - Create matching cutouts of damage. 
                  Get a single damage value per cutout by summing all pixel values within the cutout.
                - The damage value can be added as new band to each cutout
                ------------'''
                dmg_file = os.path.join(dmg_path,'S2_composite_2019-11-1_2020-3-1_tile_' + str(tileNum) + '_300m_damageContin.tif')
                crevSig_file = os.path.join(dmg_path,'S2_composite_2019-11-1_2020-3-1_tile_' + str(tileNum) + '_output_10px_crevSig.tif')
                # dmg_file = os.path.join(dmg_path,'dmg_SON_2019_tile_' + str(tileNum) + '.tif')
                
                # if DMG file not availalbe, try crevSig file. For 30m:10px , threshold =0.04
                # ...
                if os.path.exists(dmg_file):
                    data_dmg = rioxr.open_rasterio(dmg_file).astype("float32")
                elif os.path.isfile(crevSig_file):
                    data_crevSig = rioxr.open_rasterio(crevSig_file).astype("float32")
                    data_dmg = data_crevSig - 0.04   #threshold for S2 30m-10px
                    data_dmg = data_dmg.where(data_dmg>0, other=0) # only keep positive values
                else:
                    print('No dmg or crevSig file found:\n {} \n {}'.format(dmg_file, crevSig_file))

                # interpolate dmg to same resolution as tile (to link as labels)
                tile_dmg = data_dmg.isel(band=0).interp_like(da.isel(band=0))
                tile_dmg = tile_dmg.fillna(0).expand_dims(dim='band') # fill nan with zeros and add band-dim 

            if label_type == 'manual':

                # # rasterize labels: create mask 
                # labels_tileraster = geometry_mask(label_polys,
                #                                   out_shape=(len(da.y),len(da.x)),
                #                                   transform=da.rio.transform(),invert=True)
                # labels_tileraster = labels_tileraster.astype(np.dtype('uint16')) # np ndarray (x , y)
                # labels_tileraster = np.expand_dims(labels_tileraster,axis=0) # ndarray (1 , x , y) because tiledata shape (3,x,y)
                # # create dataArray from np ndarray
                # tile_dmg = xr.DataArray(
                #             data=labels_tileraster,
                #             dims=["band","x", "y"])

                # remove any empty geometries
                label_polys= label_polys.loc[~label_polys.is_empty]
                
                # create polygon mask
                labels_tileraster = geometry_mask( label_polys.to_crs(da.rio.crs).geometry,
                            out_shape=da.isel(band=0).shape, #ds.geobox.shape,
                            transform=da.rio.transform(), # ds.geobox.affine,
                            all_touched=True,
                            invert=True)
                labels_tileraster = labels_tileraster.astype(np.dtype('uint16')) # np ndarray (x , y)

                tile_dmg = da.isel(band=0).copy( data=labels_tileraster ).expand_dims(dim='band')  # ( y, x)
                tile_dmg.attrs['name']='label'
                tile_dmg.attrs['long_name']='manual_label'

                print('Manual Labels burnt into raster, {}px.'.format(tile_dmg.data.sum())) 

            # Plot labeldata as raster to perform visual check
            label_raster_file = os.path.join(os.path.expanduser('~'),'data/labels/', 
                                            'labels_'+ label_type + '_w' + str(cutout_size) + '_' + tileName + '.png')
            if ~os.path.exists(label_raster_file): 
                fig,ax= plt.subplots(1,figsize=(7,8))
                tile_dmg.isel(band=0).plot.imshow(ax=ax, vmin=0,vmax=1,cbar_kwargs={'fraction':0.045})
                ax.set_title('tile_' + str(tileNum) + ' ' + label_type + ' labels')
                ax.set_aspect('equal')
                # fig.savefig(os.path.join(path_to_traindir , model_dir, tileNum + '_normalised_histEq'))
                fig.savefig(label_raster_file)

            # Shape labels to latent-dim space
            label_cutouts, _ = create_cutouts2(tile_dmg, cutout_size)  
            labels_ldim = label_cutouts.sum(axis=1).sum(axis=1) # (Nsamples, x_win, y_win, 1) to (Nsamples,1)      
            print('labels {}, min {}, max {}'.format(labels_ldim.shape,np.min(labels_ldim),np.max(labels_ldim)))

            # Save encoded labeldata
            label_file = os.path.join(labels_path,'labels_encoded', 'labels_'+ label_type + '_w' + str(cutout_size) + '_' + tileName + '_ldim.npy')
            if ~os.path.exists(label_file):
                # label_file = os.path.join(path_to_model, tileName + "_labels_ldim.npy")
                np.save(label_file , labels_ldim) # save 'encoded' labels for later use.
                print('..saved labels at {}\n'.format(labels_path))
        
        
        
        ''' ----------
        Load model/encoder
        ------------'''

        # -- laod model
        # model_dirs = glob.glob(path_to_model +'/model_epoch*' )
        # model_dirs.sort()
        # path_to_model_epoch = model_dirs[-1]
        # model = tf.keras.models.load_model(path_to_model_epoch ,compile=False)

        # -- load encoder
        encoder_dirs = glob.glob(os.path.join(path_to_model,'encoder*') )
        encoder_dirs.sort()
        # -- if specified, select a specific (e.g. the last) encoder and redefine the list
        if epochs is not None:
            # path_to_encoder_epoch = encoder_dirs[-1]
            encoder_dirs = [encoder_dirs[epochs]]  

        for path_to_encoder_epoch in encoder_dirs:
            print('Loading encoder {}'.format(path_to_encoder_epoch))

            epoch_num = path_to_encoder_epoch.split('_')[-1]
            encoder = tf.keras.models.load_model(path_to_encoder_epoch,compile=False) # compile=True does not work
            # Get latent_dim (of sampling layer)
            latent_dim = encoder.layers[-1].output_shape[-1] 

            # print('----\n loaded model {} and encoder {}'.format(os.path.split(path_to_model_epoch)[-1] ,os.path.split(path_to_encoder_epoch)[-1] ) )
            # print('----\n loaded encoder {}'.format(os.path.split(path_to_encoder_epoch)[-1] ) )

            ''' ----------
            Encode input; load or save data
            TO DO: consider saving encoded data as dataArray -- can add labels into datafile that way
                   (encoder.predict can handle dataArrays) 
                   --> save as netcdf
            TO DO: encoder requires 3-band input, 
                   and output is encoded data per LDIM (enc_data1, enc_data2, enc_data3) 
                   Currently, i am only reading one of them. (no something else happens here)
            ------------'''

            encoded_data_file = os.path.join(path_to_model, tileName + "_encoded" + '_epoch'+str(epoch_num) +".npy")

            if os.path.exists(encoded_data_file):
                # read file
                # encoded_data = np.load(encoded_data_file)
                print('---- existing encoded data epoch {}; size: {}'.format(epoch_num, encoded_data.shape))
            else: 
                # encode data and save file
                encoded_data,_,_ = encoder.predict(tile_cutouts);
                np.save(encoded_data_file, encoded_data) # save encoded data for later use.
                print('---- succesfully encoded data epoch {}; size: {}'.format(epoch_num, encoded_data.shape))
            

    print('Done')
        

if __name__ == '__main__':
    ''' running script using named command line arguments '''
    #  Run script as "python path/to/script.py --model path_to_model"
    #  Optional: add --label label_type , or -e epoch_num
    #  Run script as "python path/to/script.py -m path_to_model -l label_type -e epoch_num"
    main()
    