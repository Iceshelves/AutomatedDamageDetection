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

homedir = '/Users/tud500158/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - TUD500158/'
workdir = os.path.join(homedir,'github/AutomatedDamageDetection/')
os.chdir(os.path.join(workdir,'scripts/train-vae/'))
import dataset
import tiles as ts

# from shapely import geometry
from rasterio.features import shapes, geometry_mask
import pathlib
import pandas as pd
import xarray as xr



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
    kernelSize1 = int(config['kernelSize1'])
    kernelSize2 = int(config['kernelSize2'])
    denseSize = int(config['denseSize'])
    latentDim = int(config['latentDim'])
    #vae:
    alpha = float(config['alpha'])
    batchSize = int(config['batchSize'])
    nEpochMax = int(config['nEpochData'])
    nEpochTrain = int(config['nEpochTrain'])
    learnRate = float(config['learningRate'])

    return (catPath, labPath, outputDir, sizeTestSet, sizeValSet, roiFile,
            bands, sizeCutOut, nEpochMax, nEpochTrain, sizeStep, stride, file_DMGinfo, normThreshold,
            filter1, filter2, kernelSize1,kernelSize2, denseSize, latentDim,
            alpha, batchSize,learnRate)

''' ----------
Define model to load
------------'''


# path_to_traindir = os.path.join(workdir,'training/2022-10/2022-10-04/') 
# model_dir = 'model_1664891181' # model 4/oct/22, alpha=2000; incl hist.eq.  
# # model_dir = 'model_1664876184' # model 4/oct/22, alpha=200; incl hist.eq
# # 'model_1665487140' # model 10/oct/22, alpha=2000; incl hist.eq. So should be same(similar) as model_1664891181..? The difference is that I applied the test-encoding now on the img instead of windows.


path_to_traindir = os.path.join(workdir,'training/2022-11/2022-11-03-vrlab/') 
# model_dir = 'model_1667468050' # model 03/nov/22, alpha=200; incl hist.eq || CANNOT OPEN
model_dir = 'model_1667483594' # model 03/nov/22, alpha=200; excl.hist.eq 


path_to_model = os.path.join(path_to_traindir, model_dir)

# get tile name
tile_file = os.path.join(homedir,'Data/tiles/training_tiles/S2_composite_2019-11-1_2020-3-1_tile_124.tif')
mask_file = os.path.join(homedir,'Data/ne_10m_antarctic_ice_shelves_polys/ne_10m_antarctic_ice_shelves_polys.shp')    

print('Loading from model_dir {}'.format(path_to_traindir+model_dir))

''' ----------
Parse input arguments
------------'''
# config = glob.glob(path_to_traindir + "train-vae.ini")
config = os.path.join(path_to_model,'train-vae.ini')

catPath, labPath, outputDir, sizeTestSet, sizeValSet, roiFile, \
        bands, cutout_size, nEpochmax, nEpochTrain, sizeStep, stride, file_DMGinfo, normThreshold, \
        filter1, filter2, kernelSize1, kernelSize2, denseSize, latentDim, \
        alpha, batchSize,learnRate = parse_config(config)


''' ----------
Load model
------------'''


# path_to_model = glob.glob(path_to_traindir + 'model*')
# print(path_to_model)
# path_to_model = path_to_model[-1]

epoch_dirs = glob.glob(path_to_model +'/model_epoch*' )
encoder_dirs = glob.glob(path_to_model+'/encoder*')
path_to_model_epoch = epoch_dirs[-1]
path_to_encoder_epoch = encoder_dirs[-1]

try:
    model = tf.keras.models.load_model(path_to_model_epoch ,compile=False)
    encoder = tf.keras.models.load_model(path_to_encoder_epoch,compile=False)
except ValueError: # when mdoel saved in h5 format, try different laod approach 
    model = tf.keras.models.load_model(path_to_model_epoch)
    encoder = tf.keras.models.load_model(path_to_encoder_epoch)

# Get latent_dim (of sampling layer)
latent_dim = encoder.layers[-1].output_shape[-1] # is 4


print('---- loaded model, encoder and .ini')

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

''' ----------
Get testdata: one tile

Instead of loading tiles directly as tf Dataset with dataset.Dataset(), load the tile as xarray to be able to link it to labels
------------'''

# get info on which tiles are assigned as test tiles
datasets_json = glob.glob(path_to_traindir + 'datasets*.json')

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
    Actually read the tile, make cutouts, linked with labeldata
    
Update: do not link laabldata; now read same-tile NERD output
------------'''

# for tile in tile_list: # from dataset.py _generate_cutouts



tileName = tile_file.split("/")[-1][:-4] # vb: 'S2_composite_2019-11-1_2020-3-1_tile_124'
tileNum='tile_'+tileName.split('tile_')[-1]

print('\n----\n Processing ' + tileName +'\n')

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
da = mask_data(da,mask_file)


''' ----------
Normalise and Equalise
------------'''

da = normalise_and_equalise(da,normThreshold=normThreshold[0],equalise=True)


''' ----------
Plot tile (for test phase)
------------'''
try: 
    fig,ax= plt.subplots(1,figsize=(10,8))
    da.attrs['long_name']='imgbands';
    # da.isel(band=0).plot.imshow(ax=ax,vmin=0,vmax=1,cbar_kwargs={"fraction": 0.046})
    da.plot.imshow(ax=ax,rgb='band', vmin=0,vmax=1)#,cbar_kwargs={"fraction": 0.046})
    ax.set_title(tileNum + ' normalised and equalised')
    ax.set_aspect('equal')
    fig.savefig(os.path.join(path_to_traindir , model_dir, tileNum + '_normalised_histEq'))
except Exception:
    print('Error for plotting image -- skipped')
    pass

''' ----------
Create cut-outs
------------'''

# generate windows -- cut 

tile_cutouts, tile_cutouts_da = create_cutouts2(da,cutout_size) # samples, x_win, y_win, bands: (250000, 20, 20, 1)
# label_cutouts, __ = create_cutouts(tile_dmg_int, cutout_size)


''' ----------
Encode input 
------------'''
encoded_data_file = os.path.join(path_to_model, tileName + "_encoded.npy")
if os.path.exists(encoded_data_file):
    # read file
    encoded_data = np.load(encoded_data_file)
    print('---- loaded encoded data; size: ', encoded_data.shape)
else: 
    # encode data and save file
    encoded_data,_,_ = encoder.predict(tile_cutouts);
    np.save(encoded_data_file, encoded_data) # save encoded data for later use.
    # np.save(tileName + "_labels.npy", label_cutouts) # save encoded data for later use.
    print('---- succesfully encoded data; size: ', encoded_data.shape)


''' ----------
predict output  
------------'''   
#     # predicted_data = model.predict(test_set_tf); # reconstruct images (windows):
#     predicted_data = model.predict(tile_cutouts); # reconstruct images (windows):
#     np.save(tileName + "_predicted.npy",predicted_data)
#     print('---- succesfully predicted data')


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


    fig,axes = plt.subplots(2,2,figsize=(20,10))

    # axes[0,0].imshow(tile_da[0])
    # axes[0,0].set_aspect('equal')
    # tile_data.isel(band=0).plot.imshow()#ax=axes[0],x='x',add_label=False)

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

    fig.savefig(os.path.join(path_to_traindir , model_dir, tileNum + '_spatial_lspace'))


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
    
