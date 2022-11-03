import numpy as np
import tensorflow as tf
import glob
import geopandas as gpd
import json
import configparser 
from sklearn.manifold import TSNE
import rioxarray as rioxr
import xarray as xr


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

print('---- modules imported')


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
    normThreshold = float(config['normalizationThreshold'])
        
    return (catPath, labPath, outputDir, sizeTestSet, sizeValSet, roiFile,
            bands, sizeCutOut, nEpochMax, sizeStep, normThreshold)


# path_to_traindir = './model_v0/train_epoch_2/' # path on local computer
# path_to_traindir = '../train/model_v0/train_epoch_2/' # path on cartesius
path_to_traindir = os.path.join(workdir,'training//2022-10/') # path on cartesius
model_dir = 'model_1665487140'

path_to_model = os.path.join(path_to_traindir, model_dir)

# parse input arguments
# config = glob.glob(path_to_traindir + "train-vae.ini")
config = os.path.join(path_to_model,'train-vae.ini')
catalog_path, labels_path, outputDir, sizeTestSet, sizeValSet, roiFile, bands, \
    cutout_size, nEpochmax, sizeStep, normThreshold = parse_config(config)


''' ----------
Load model
------------'''


# path_to_model = glob.glob(path_to_traindir + 'model*')
# print(path_to_model)
# path_to_model = path_to_model[-1]

epoch_dirs = glob.glob(path_to_model +'/epoch*' )
encoder_dirs = glob.glob(path_to_model+'/encoder*')
path_to_model_epoch = epoch_dirs[-1]
path_to_encoder_epoch = encoder_dirs[-1]


model = tf.keras.models.load_model(path_to_model_epoch ,compile=False)
encoder = tf.keras.models.load_model(path_to_encoder_epoch,compile=False)

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

# get tile name

tile_file = os.path.join(homedir,'Data/tiles/training_tiles/S2_composite_2019-11-1_2020-3-1_tile_124.tif')
mask_file = os.path.join(homedir,'Data/ne_10m_antarctic_ice_shelves_polys/ne_10m_antarctic_ice_shelves_polys.shp')    
# mask_file = '..'


tileName = tile_file.split("/")[-1][:-4] # vb: 'S2_composite_2019-11-1_2020-3-1_tile_124'
print('\n----\n Processing ' + tileName +'\n')

# read tile - floats are required to mask with NaN's
da = rioxr.open_rasterio(tile_file).astype("float32")

# select bands
if bands is not None:
    if type(bands) is not list:
        da = da.sel(band=[bands])
    else:
        da = da.sel(band=bands)

# mask/clip: if self.mask is not None: [removed; see _generate_cutouts]
da = mask_data(da,mask_file)



# generate windows -- normalise and equalise

# tile_cutouts, tile_cutouts_da = create_cutouts(da,cutout_size, normThreshold=normThreshold[0],equalise=True) # samples, x_win, y_win, bands: (250000, 20, 20, 1)

img_equal = normalise_and_equalise(da,normThreshold=normThreshold[0],equalise=True)
tile_cutouts, tile_cutouts_da, img_equal = create_cutouts2(da,cutout_size) # samples, x_win, y_win, bands: (250000, 20, 20, 1)
# label_cutouts, __ = create_cutouts(tile_dmg_int, cutout_size)

# print(tile_cutouts.shape , label_cutouts.shape)
print(tile_cutouts.shape)





# ''' ----------
# Create cut-outs
#     Actually read the tile, make cutouts
# ------------'''



# for tile in tile_list: # from dataset.py _generate_cutouts
#     # get tile name
#     tileName = tile.split("/")[-1][:-4] # vb: 'S2_composite_2019-11-1_2020-3-1_tile_124'
#     print('\n----\n Processing ' + tileName +'\n')
    
#     # read tile - floats are required to mask with NaN's
#     da = rioxr.open_rasterio(tile).astype("float32")

#     # select bands
#     if bands is not None:
#         if type(bands) is not list:
#             da = da.sel(band=[bands])
#         else:
#             da = da.sel(band=bands)

#     # mask/clip: if self.mask is not None: [removed; see _generate_cutouts]

#     # apply offset [removed: only relevant for training, not for testing]
    
#     # rasterize labels: create mask on tile raster
#     labels_tileraster = geometry_mask(label_polys,
#                                       out_shape=(len(da.y),len(da.x)),
#                                       transform=da.rio.transform(),invert=True)
#     labels_tileraster = labels_tileraster.astype(np.dtype('uint16')) # np ndarray (x , y)
#     labels_tileraster = np.expand_dims(labels_tileraster,axis=0) # ndarray (1 , x , y) because tiledata shape (3,x,y)
#     # create dataArray from np ndarray
#     da_label = xr.DataArray(
#                 data=labels_tileraster,
#                 dims=["band","x", "y"])
    

#     # generate windows
#     da = da.rolling(x=cutout_size, y=cutout_size)
#     da = da.construct({'x': 'x_win', 'y': 'y_win'}, stride=cutout_size)

#     # drop NaN-containing windows
#     da = da.stack(sample=('x', 'y'))
#     da = da.dropna(dim='sample', how='any')

#     # normalize
#     if normThreshold is not None:
#         da = (da + 0.1) / (normThreshold + 1)
#         da = da.clip(max=1)

#     tile_cutouts = da.data.transpose(3, 1, 2, 0) # samples, x_win, y_win, bands: (250000, 20, 20, 3)
    
#     # same for labels:
#     da_label = da_label.rolling(x=cutout_size, y=cutout_size)
#     da_label = da_label.construct({'x': 'x_win', 'y': 'y_win'}, stride=cutout_size)
#     da_label = da_label.stack(sample=('x', 'y'))
#     da_label = da_label.dropna(dim='sample', how='any')
#     # no need to normalize
#     label_cutouts = da_label.data.transpose(3, 1, 2, 0)  # samples, x_win, y_win, bands: (250000, 20, 20, 1)


#     print(tile_cutouts.shape , label_cutouts.shape)



#     ''' ----------
#     Encode input 
#     ------------'''


#     # encoded_data,_,_ = encoder.predict(test_set_tf);
#     encoded_data,_,_ = encoder.predict(tile_cutouts);
#     np.save(tileName + "_encoded.npy", encoded_data) # save encoded data for later use.
#     np.save(tileName + "_labels.npy", label_cutouts) # save encoded data for later use.
#     print('---- succesfully encoded data; size: ', encoded_data.shape)


#     # predicted_data = model.predict(test_set_tf); # reconstruct images (windows):
#     predicted_data = model.predict(tile_cutouts); # reconstruct images (windows):
#     np.save(tileName + "_predicted.npy",predicted_data)
#     print('---- succesfully predicted data')

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
    
