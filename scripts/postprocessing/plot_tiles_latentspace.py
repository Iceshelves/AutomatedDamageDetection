import numpy as np
import tensorflow as tf
import glob
import geopandas as gpd
import json
import configparser 
from sklearn.manifold import TSNE
import rioxarray as rioxr
import xarray as xr

import matplotlib.pyplot as plt
import os
from skimage import exposure as skimage_exposure

# homedir = '/Users/tud500158/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - TUD500158/'
workdir = '/gpfs/home3/mizeboud/' #os.path.join(homedir,'github/AutomatedDamageDetection/')
os.chdir(os.path.join(workdir,'preprocessing/scripts/train-vae/'))
import dataset
import tiles as ts

# from shapely import geometry
from rasterio.features import shapes, geometry_mask
import pathlib
import pandas as pd
import xarray as xr
import sys


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
            bands, sizeCutOut, nEpochMax, nEpochTrain, sizeStep, stride, file_DMGinfo, normThreshold,
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

    tile_cutouts = da.data.transpose(3, 1, 2, 0) # samples, x_win, y_win, bands: (250000, 20, 20, 3)
    tile_cutouts_da = da.transpose('sample','x_win','y_win','band')

    return tile_cutouts, tile_cutouts_da

def mask_data(data, mask_file):
    mask_poly = gpd.read_file(mask_file).to_crs(epsg=3031)
    # gdf = mask_poly.unary_union 
    # mask = data #.copy(data=np.ones_like(data.values)) # set up img with only 1 vluess
    masked_data = data.rio.clip(mask_poly.unary_union, mask_poly.crs, drop=False, invert=False) # mask (raster)
    return masked_data



def main(model_dir):

    ''' ----------
    Define model to load
    ------------'''
    
    if model_dir is None:
        raise NameError('No config file specified. Run script as "python this_script.py /path/to/model_dir/"')
    
    if os.path.isdir(model_dir):
        path_to_model = model_dir
    else:
        path_to_model = os.path.join(os.path.expanduser('~'), model_dir)
        
        
    # path_to_traindir = os.path.join(workdir,'training/2022-10/2022-10-04/') 
    # # model_dir = 'model_1664891181' # model 4/oct/22, alpha=2000; incl hist.eq.  
    # model_dir = 'model_1664876184' # model 4/oct/22, alpha=200; incl hist.eq
    # # 'model_1665487140' # model 10/oct/22, alpha=2000; incl hist.eq. So should be same(similar) as model_1664891181..? The difference is that I applied the test-encoding now on the img instead of windows.

    # path_to_traindir = os.path.join(workdir,'training/2022-11/2022-11-03-vrlab/') 
    # model_dir = 'model_1667468050' # model 03/nov/22, alpha=200; incl hist.eq


    # path_to_traindir = os.path.join(workdir,'train','2022-11')
    # model_dir = 'model_1667487343' # 3 nov // cannot read sample layer
    # model_dir = 'model_1668077484' # 10 nov; adapthist & alpha=200


    # ------
    # path_to_model = os.path.join(path_to_traindir, model_dir)

    # print('Loading from model_dir {}'.format(path_to_traindir+'/'+model_dir))

    ''' ----------
    Parse input arguments
    ------------'''
    # config = glob.glob(path_to_traindir + "train-vae.ini")
    # config = os.path.join(path_to_model,'train-vae.ini')
    config = glob.glob(os.path.join(path_to_model,'*.ini'))
    
    catPath, labPath, outputDir, sizeTestSet, sizeValSet, roiFile, \
            bands, cutout_size, nEpochmax, nEpochTrain, sizeStep, stride, file_DMGinfo, normThreshold, \
            filter1, filter2, kernelSize1, kernelSize2, denseSize, latentDim, \
            alpha, batchSize,learnRate = parse_config(config)

    ''' ----------
    Get tile to process
    --------------'''
    # LOCAL
    # tile_file = os.path.join(homedir,'Data/tiles/training_tiles/S2_composite_2019-11-1_2020-3-1_tile_124.tif')
    # mask_file = os.path.join(homedir,'Data/ne_10m_antarctic_ice_shelves_polys/ne_10m_antarctic_ice_shelves_polys.shp')    

    # SNELLIUS;
    mask_file = roiFile
    catalog_path = catPath
    
    # tile_file = '/projects/0/einf512/S2_composite_2019-11-1_2020-3-1/S2_composite_2019-11-1_2020-3-1_tile_124.tif'
    tile_file = '/projects/0/einf512/S2_composite_2019-11-1_2020-3-1/S2_composite_2019-11-1_2020-3-1_tile_140.tif'


    ''' ----------
    Get testdata: one tile

    Instead of loading tiles directly as tf Dataset with dataset.Dataset(), load the tile as xarray to be able to link it to labels
    ------------'''

    # get info on which tiles are assigned as test tiles
    # datasets_json = glob.glob(path_to_traindir + 'datasets*.json')
    datasets_json = glob.glob(os.path.join(path_to_model , 'datasets*.json') )

    with open(datasets_json[0]) as file:        # Opening JSON file
        datasets_dict = json.loads(file.read()) # load data

    test_set_paths = datasets_dict['test']   
    test_set_paths

    # test: 1 tile
    # num_tiles = 1 # number of tiles to load
    # tile_list = test_set_paths[:num_tiles];
    # print('tilelist: ' , tile_list)

    tile_list = test_set_paths

    ''' ----------
    Get Labels
    ------------'''

    # # # read tile catalog to be able to get CRS and filter labels to same date
    # catalog = ts._read_tile_catalog(catalog_path)
    # tiles = ts._catalog_to_geodataframe(catalog)

    # labels = ts._read_labels(labels_path, verbose=True)
    # labels = labels.to_crs(tiles.crs)  # make sure same CRS is used
    # # select the only labels matching the tiles timespan
    # labels = ts._filter_labels(labels,
    #                         tiles.start_datetime.min(),
    #                         tiles.end_datetime.max())


    # ## Create GeoSeries from labels.geometry
    # label_polys = gpd.GeoSeries(labels.geometry,crs=labels.crs) # Combine all geometries to one GeoSeries, with the 



    ''' ----------
    Create cut-outs
        Actually read the tile, make cutouts, plot latent space
    ------------'''

    # for tile in tile_list: # from dataset.py _generate_cutouts


    tileName = tile_file.split("/")[-1][:-4] # vb: 'S2_composite_2019-11-1_2020-3-1_tile_124'
    tileNum='tile_'+tileName.split('tile_')[-1]

    print('----\n Processing ' + tileName +'\n')

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
    # da = mask_data(da,mask_file)


    ''' ----------
    Normalise and Equalise
    ------------'''

    # da = normalise_and_equalise(da,normThreshold=normThreshold[0],equalise=True)
    da = normalise_and_equalise(da,normThreshold=normThreshold[0],equalise=False)


    ''' ----------
    Plot tile (for test phase)
    ------------'''
    try: 
        fig,ax= plt.subplots(1,figsize=(7,8))
        da.attrs['long_name']='imgbands';
        da.plot.imshow(ax=ax,rgb='band', vmin=0,vmax=1)
        ax.set_title(tileNum + ' normalised')
        ax.set_aspect('equal')
        # fig.savefig(os.path.join(path_to_traindir , model_dir, tileNum + '_normalised_histEq'))
        fig.savefig(os.path.join(path_to_model, tileNum + '_normalised' ))
    except Exception:
        print('Error for plotting image -- skipped')
        pass

    ''' ----------
    Create cut-outs
    ------------'''

    # generate windows -- cut 

    tile_cutouts, tile_cutouts_da = create_cutouts2(da,cutout_size) # samples, x_win, y_win, bands: (250000, 20, 20, 1)
    # label_cutouts, __ = create_cutouts(tile_dmg_int, cutout_size)
    print('cutouts {} // {}'.format(tile_cutouts.shape,tile_cutouts_da.shape))

    
    
    ''' ----------
    Load model/encoder
    ------------'''

    # -- laod model
    # model_dirs = glob.glob(path_to_model +'/model_epoch*' )
    # model_dirs.sort()
    # path_to_model_epoch = model_dirs[-1]
    # model = tf.keras.models.load_model(path_to_model_epoch ,compile=False)
    
    # -- load encoder
    encoder_dirs = glob.glob(path_to_model+'/encoder*')
    encoder_dirs.sort()
    
    # -- (A) select the last encoder
    
    # path_to_encoder_epoch = encoder_dirs[-1]
    
    # -- (B) do all epochs
    for path_to_encoder_epoch in encoder_dirs:
        
        epoch_num = path_to_encoder_epoch.split('_')[-1]
        encoder = tf.keras.models.load_model(path_to_encoder_epoch,compile=False) # compile=True does not work
        # Get latent_dim (of sampling layer)
        latent_dim = encoder.layers[-1].output_shape[-1] 

        # print('----\n loaded model {} and encoder {}'.format(os.path.split(path_to_model_epoch)[-1] ,os.path.split(path_to_encoder_epoch)[-1] ) )
        print('----\n loaded encoder {}'.format(os.path.split(path_to_encoder_epoch)[-1] ) )

    
        ''' ----------
        (Load) Encode input 
        ------------'''

        encoded_data_file = os.path.join(path_to_model, tileName + "_encoded" + '_epoch'+str(epoch_num) +".npy")

        if os.path.exists(encoded_data_file):
            # read file
            encoded_data = np.load(encoded_data_file)
            print('---- loaded encoded data epoch {}; size: {}'.format(epoch_num, encoded_data.shape))
        else: 
            # encode data and save file
            encoded_data,_,_ = encoder.predict(tile_cutouts);
            np.save(encoded_data_file, encoded_data) # save encoded data for later use.
            # np.save(tileName + "_labels.npy", label_cutouts) # save encoded data for later use.
            print('---- succesfully encoded data epoch {}; size: {}'.format(epoch_num, encoded_data.shape))



        ''' ----------
        Plot latent space spatially Ldim=4
        ------------'''

        if latent_dim == 4:
            z0 = encoded_data[:,0]
            z1 = encoded_data[:,1]
            z2 = encoded_data[:,2]
            z3 = encoded_data[:,3]



            # add z0 only to (sample,x,y. )
            L_space_xy_z0 =  tile_cutouts_da.isel(band=[0],x_win=0,y_win=0).copy(deep=True, data=np.expand_dims(z0,axis=1) ) #.unstack()
            L_space_xy_z1 =  tile_cutouts_da.isel(band=[0],x_win=0,y_win=0).copy(deep=True, data=np.expand_dims(z1,axis=1) ) #.unstack()
            L_space_xy_z2 =  tile_cutouts_da.isel(band=[0],x_win=0,y_win=0).copy(deep=True, data=np.expand_dims(z2,axis=1) ) #.unstack()
            L_space_xy_z3 =  tile_cutouts_da.isel(band=[0],x_win=0,y_win=0).copy(deep=True, data=np.expand_dims(z3,axis=1) ) #.unstack()


            fig,axes = plt.subplots(2,2,figsize=(12,10)) # (20,10)

            L_space_xy_z0.unstack().isel(band=0).plot.imshow(ax=axes[0,0],x='x',cmap='RdBu') #,vmin=-0.1,vmax=0.1, # vmin=-2 vmax=2
            axes[0,0].set_title('z0')
            axes[0,0].set_aspect('equal')


            L_space_xy_z1.unstack().isel(band=0).plot.imshow(ax=axes[0,1],x='x',cmap='RdBu') # vmin=-0.1,vmax=0.1,
            axes[0,1].set_title('z1')
            axes[0,1].set_aspect('equal')


            L_space_xy_z2.unstack().isel(band=0).plot.imshow(ax=axes[1,0],x='x',cmap='RdBu') # vmin=-2 vmax=2
            axes[1,0].set_title('z2')
            axes[1,0].set_aspect('equal')


            L_space_xy_z3.unstack().isel(band=0).plot.imshow(ax=axes[1,1],x='x',cmap='RdBu') # vmin=-2 vmax=2
            axes[1,1].set_title('z3')
            axes[1,1].set_aspect('equal')

            # fig.savefig(os.path.join(path_to_traindir , model_dir, tileNum + '_spatial_lspace'))
            fig.savefig(os.path.join(path_to_model, tileNum + '_spatial_lspace' + '_epoch'+str(epoch_num) ))


        ''' ----------
        Plot latent space spatially Ldim=2
        ------------'''

        if latent_dim == 2:
            z0 = encoded_data[:,0]
            z1 = encoded_data[:,1]

            # add z0 only to (sample,x,y. )
            L_space_xy_z0 =  tile_cutouts_da.isel(band=[0],x_win=0,y_win=0).copy(deep=True, data=np.expand_dims(z0,axis=1) ) #.unstack()
            L_space_xy_z1 =  tile_cutouts_da.isel(band=[0],x_win=0,y_win=0).copy(deep=True, data=np.expand_dims(z1,axis=1) ) #.unstack()

            fig,axes = plt.subplots(1,2,figsize=(10,6))

            L_space_xy_z0.unstack().isel(band=0).plot.imshow(ax=axes[0],x='x',cmap='RdBu') #,vmin=-0.1,vmax=0.1, # vmin=-2 vmax=2
            axes[0].set_title('z0')
            axes[0].set_aspect('equal')

            L_space_xy_z1.unstack().isel(band=0).plot.imshow(ax=axes[1],x='x',cmap='RdBu') # vmin=-0.1,vmax=0.1,
            axes[1].set_title('z1')
            axes[1].set_aspect('equal')

            # fig.savefig(os.path.join(path_to_traindir , model_dir, tileNum + '_spatial_lspace'))
            fig.savefig(os.path.join(path_to_model, tileNum + '_spatial_lspace' + '_epoch'+str(epoch_num) ))
    

if __name__ == '__main__':
    #  Run script as "python path/to/script.py /path/to/model_dir"
        
    # retrieve config filename from command line
    model_dir = sys.argv[1] if len(sys.argv) > 1 else None

    # run script
    main(model_dir)   

    
    
'''
ARCIHVED:
'''


# # def embed_latentspace_2D(encoded_data,
# #                          latent_dim,perplexity=10, 
# #                          n_iter=1000,
# #                          n_iter_without_progress=300):
# #     ''' Consider using perplexity between 5 and 50 (default 30); larger datasets usually require a larger perplexity.
# #         n_iter default 1000, should be at least 250
# #         learning_rate: the learning rate for t-SNE is usually in the range [10.0, 1000.0], default=200; The ‘auto’ option sets the learning_rate to max(N / early_exaggeration / 4, 50)
# #     '''
    
# #     if latent_dim > 2: # only embed if latent_dim is higher than 2D (otherwise plot 2D)
# #         z_mean_2D = TSNE(n_components=2,
# #                          perplexity=perplexity,
# #                          n_iter=n_iter,
# #                          learning_rate='auto',
# #                          init='pca',
# #                          n_iter_without_progress=n_iter_without_progress).fit_transform(encoded_data);
# #     else: # no embedding needed
# #         z_mean_2D = encoded_data  # TO DO: check if this outputs the same shape as the embedded z_mean_2D
        
# #     return z_mean_2D

# # Nsamples = encoded_data.shape[0] #100000 
# # encoded_2D_testdata = embed_latentspace_2D(encoded_data, # sample selection
# #                                            latent_dim,perplexity=40,
# #                                            n_iter=250,
# #                                            n_iter_without_progress=100)
# # np.save("embedded_data_N{:d}_tileX.npy".format(Nsamples),reconstructed_window)
# z_mean_2D = TSNE(n_components=2,
#                  perplexity=30,
#                  n_iter=250,
#                  init='pca',
#                  n_iter_without_progress=100,
#                  n_jobs=4).fit_transform(encoded_data);
# np.save("embedded_data_N_tileX.npy",encoded_2D_testdata)
# print('---- succesfully embedded data to 2D')
 

    
# print('\n')

# ''' ----------
# Plot clustering
# ------------'''
  
    

# def plot_latentspace_clusters_no_labels( embedded_data ):#,labels):
#     ''' Embedded data should have (N,2)'''
#     plt.figure(figsize=(8, 8))
#     plt.scatter(embedded_data[:, 0], embedded_data[:, 1])#, c=labels,s=2,cmap='winter') # also add size for scatter point
#     plt.colorbar()
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.show()
#     return fig
    
    
# def plot_latentspace_clusters( embedded_data,labels ):    
#     marksize = 1
    
#     fig, ax1 = plt.subplots(figsize=(8,8) )
#     s1 = ax1.scatter(embedded_data[:, 0], embedded_data[:, 1], c=labels, s=marksize, cmap='winter',vmin=0, vmax=1) # also add size for scatter point
#     ax1.set_xlabel("z[0]"); 
#     ax1.set_ylabel("z[1]");
#     fig.colorbar(s1,ax=ax1); 
#     return fig

# fig = plot_latentspace_clusters_no_labels( encoded_2D_testdata )
# fig.savefig('embedded_testdata_nolabels_tileX')


# # fig = plot_latentspace_clusters( testdata_encoded_2D , testdata_labels )
    
