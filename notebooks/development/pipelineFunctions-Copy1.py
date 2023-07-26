from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pathlib
import geopandas as gpd
import pandas as pd
from rasterio.features import geometry_mask
import numpy as np
import xarray as xr



### COPIED FROM tiles.py
def _read_labels(labels_path, verbose=True):
    """ Read all labels, and merge them in a single GeoDataFrame """
    labels_path = pathlib.Path(labels_path)
    labels = [gpd.read_file(p) for p in labels_path.glob("*.geojson")]
    if verbose:
        print("Labels successfully read from {} files".format(len(labels)))

    crs = labels[0].crs
    assert all([label.crs == crs for label in labels])
    labels = pd.concat(labels)

    # fix datetimes' type
    labels.Date = pd.to_datetime(labels.Date)
    return labels



def _filter_labels(labels, start_datetime, end_datetime, verbose=True):
    """ Select the labels whose date in the provided datetime range """
    mask = (labels.Date >= start_datetime) & (labels.Date <= end_datetime)
    if verbose:
        print("Selecting {} out of {} labels".format(mask.sum(), len(labels)))
    return labels[mask]

def mask_labels_tile(labels, tile_data):
    # create mask
    label_polys = gpd.GeoSeries(labels.geometry,crs=labels.crs) # Combine all geometries to one GeoSeries, with the correct projection; s = s.to_crs(...)
    label_raster = geometry_mask(label_polys,out_shape=(len(tile_data.y),len(tile_data.x)),transform=tile_data.rio.transform(),invert=True)
    # Inspect data type of mask -> ndarray
    label_raster = np.expand_dims(label_raster,axis=0)
    label_raster = label_raster.astype(np.dtype('uint16'))
    return label_raster

def balance_windowdata(tile_cutouts, balance_ratio):
    ''' Balance the ratio of labelled/unlabelled windows
    - balance_ratio = N_labelled / N_unlabelled such that a balance of 1 means equal numbers of (un)labelled data windows
    '''
    idx_y1 = tile_cutouts.isel(band=-1) == 1 #[:,:,:,-1] == 1 # all labelled pixels
    count_y1_in_window = np.sum(idx_y1,axis=(1,2)) # labels summed per window
    idx_windows_y1     = count_y1_in_window > 0 # boolean: all windows that have at least one labelled pixel

    idx_windows_y1
    # separate labelled and unlabelled windows
    cutouts_label_1 = tile_cutouts.isel(sample=idx_windows_y1.values) #[idx_windows_y1,:,:,:] # labelled
    cutouts_label_0 = tile_cutouts.isel(sample=~idx_windows_y1.values)# [~idx_windows_y1,:,:,:]# unlabelled

    N_labeled_windows = len(cutouts_label_1)
    N_train_windows_y1 = N_labeled_windows #- N_test_windows_y1

    # calculate number of unlabelledwindows for train (& test) data based on the desired balance ratio
    N_train_windows_y0 = int(1/balance_ratio * N_train_windows_y1)

    # # select the windows according to the ratio
    # # TO DO: select windows in a random order
    data_train_label_1 = cutouts_label_1[:N_train_windows_y1 ,:,:,:]
    data_train_label_0 = cutouts_label_0[:N_train_windows_y0 ,:,:,:] 
    # train_data = np.concatenate((data_train_label_1,data_train_label_0))
    train_data = xr.concat((data_train_label_1,data_train_label_0),dim='sample')
    return train_data


def embed_latentspace_2D(encoded_data,
                         latent_dim,
                         perplexity=10, 
                         n_iter=1000,
                         n_iter_without_progress=300):
    ''' Encoded data should be shape (n_samples, n_features=latent_dim)
        Consider using perplexity between 5 and 50 (default 30); larger datasets usually require a larger perplexity.
        n_iter default 1000, should be at least 250
        learning_rate: the learning rate for t-SNE is usually in the range [10.0, 1000.0], default=200; The ‘auto’ option sets the learning_rate to max(N / early_exaggeration / 4, 50)
    '''
    
    if latent_dim > 2: # only embed if latent_dim is higher than 2D (otherwise plot 2D)
        print('..embedding {}D to 2D..'.format(latent_dim) )
        z_mean_2D = TSNE(n_components=2,
                         perplexity=30,
                         n_iter=250,
                         init='pca',
                         n_iter_without_progress=100,
                         n_jobs=4).fit_transform(encoded_data)
    else: # no embedding needed
        z_mean_2D = encoded_data  # TO DO: check if this outputs the same shape as the embedded z_mean_2D
        
    return z_mean_2D


def plot_latentspace_clusters( embedded_data,labels ):    
    marksize = 100
    marksize = 15

    idx_labelled = labels > 0
    data_labelled = embedded_data[idx_labelled,:]
    data_unlabelled = embedded_data[~idx_labelled,:]
    
    
    fig, ax1 = plt.subplots(figsize=(8,8) )
#     s1 = ax1.scatter(embedded_data[:, 0], embedded_data[:, 1], c=labels, s=marksize*labels,vmin=0, vmax=labels.max()) # also add size for scatter point
#     s1 = ax1.scatter(embedded_data[::-1, 0], embedded_data[::-1, 1], c=labels[::-1], s=marksize,vmin=0, vmax=labels.max()) # also add size for scatter point
    s1 = ax1.scatter(data_unlabelled[:, 0], data_unlabelled[:, 1], c=labels[~idx_labelled], s=marksize,vmin=0, vmax=labels.max()) # also add size for scatter point
    s2 = ax1.scatter(data_labelled[:, 0], data_labelled[:, 1], c=labels[idx_labelled], s=marksize,vmin=0, vmax=labels.max()) # also add size for scatter point
    
    ax1.set_xlabel("z[0]"); 
    ax1.set_ylabel("z[1]");
    fig.colorbar(s1,ax=ax1); 
    return fig, ax1