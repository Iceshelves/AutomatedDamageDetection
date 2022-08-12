import gc
import random

import geopandas as gpd
import rioxarray as rioxr
import tensorflow as tf
import tiles


class Dataset:
    """
    Split tiles into square cutouts and return these as a tensorflow Dataset.
    """
    def __init__(self, tile_list, cutout_size, bands, offset=0, stride=None,
                 num_tiles=None, shuffle_tiles=False, norm_threshold=None,
                 return_coordinates=False,
                 balance_ratio=0):
        """
        :param tile_list: list of tile paths
        :param cutout_size: length of the cutout side (number of pixels)
        :param bands: list of band indices to be considered
        :param offset: lateral (X and Y) cutout offset (number of pixels)
        :param stride: cutout stride (number of pixels)
        :param num_tiles: only use a selection of the provided tile list
        :param shuffle_tiles: if True, shuffle tiles
        :param norm_threshold: normalize image intensities using the provided
            threshold value
        :param return_coordinates: if True, return the (X, Y) coordinates
            together with the cutouts
        """
        self.cutout_size = cutout_size
        self.bands = bands
        self.stride = stride if stride is not None else self.cutout_size
        _num_tiles = num_tiles if num_tiles is not None else len(tile_list)
        self.tiles = random.sample(tile_list, _num_tiles) \
            if shuffle_tiles else tile_list[:_num_tiles]
        if offset >= cutout_size:
            raise ValueError(
                "offset ({}) larger than window size ({}) - set "
                "offset to {}".format(offset,cutout_size,cutout_size % offset)
            )
        self.offset = offset
        self.balance_ratio = balance_ratio

        self.mask = None
        self.buffer = None
        self.invert = None
        self.all_touched = None
        self.norm_threshold = norm_threshold
        self.return_coordinates = return_coordinates

    def set_mask(self, geometry, crs, buffer=None, invert=False,
                 all_touched=False):
        """
        Clip the tiles with the provided geometry before generating the
        cutouts.

        :param geometry: geometry
        :param crs: coordinate reference system (geopandas compatible)
        :param buffer: apply provided buffer to the geometry
        :param invert: clip the tiles with the inverted mask
        :param all_touched: see `rioxarray.clip`
        """
        self.mask = gpd.GeoSeries({"geometry": geometry}, crs=crs)
        self.buffer = buffer if buffer is not None else 0
        self.invert = invert
        self.all_touched = all_touched

    def to_tf(self):
        """ Generate cut-outs and Return dataset as a tensorflow `Dataset` object. """
        ds = tf.data.Dataset.from_generator(
            self._generate_cutouts,
            output_types=(tf.float64, tf.float64, tf.float32),
            output_shapes=(
                None,  # x
                None,  # y
                # samples, x_win, y_win, bands
                (None, self.cutout_size, self.cutout_size, None)
            )
        )
        if not self.return_coordinates:
            ds = ds.map(lambda x, y, cutout: cutout)  # only return cutout
        # remove the outer dimension of the array if not return_coordinates
        return ds.flat_map(
            lambda *x: tf.data.Dataset.from_tensor_slices(
                x if len(x) > 1 else x[0]
            )
        )


#     def add_labels_band(tile_data, labels_path='/projects/0/einf512/labels',catalog_path='/projects/0/einf512/S2_composite_catalog'):  # [DEV]

#         # --- [DOUBLE WORK same as in tiles.py; put it somewhere else] ----
#         # read tile catalog
#         catalog = tiles._read_tile_catalog(catalog_path)
#         tiles_all = tiles._catalog_to_geodataframe(catalog)

#         # load labels for the dataset [DUBBEL WERK; al in tiles.py]
#         labels = tiles._read_labels(labels_path, verbose=True)
#         labels = labels.to_crs(tiles.crs)  # make sure same CRS is used
#         # select labels matching the tiles timespan
#         labels = tiles._filter_labels(labels,
#                                 tiles_all.start_datetime.min(),
#                                 tiles_all.start_datetime.min())
#         # --- [DOUBLE WORK (end)]  ----

#         # create label as raster
#         label_polys = gpd.GeoSeries(labels.geometry,crs=labels.crs) # Combine all geometries to one GeoSeries, with the correct projection; s = s.to_crs(...)
#         label_raster = geometry_mask(label_polys,out_shape=(len(tile_data.y),len(tile_data.x)),transform=tile_data.rio.transform(),invert=True)
#         # Inspect data type of mask -> ndarray
#         label_raster = np.expand_dims(label_raster,axis=0)
#         label_raster = label_raster.astype(np.dtype('uint16'))
#         # return label_raster

#         # add labelraster to tile
#         tile_data_np = tile_data.values
#         # tile_data_np = np.concatenate((tile_data_np[0:3],labels_raster));
#         tile_data_np = np.concatenate((tile_data_np,labels_raster));

#         tile = xr.DataArray(data=tile_data_np,dims=['band','y','x'],
#                             coords={ #'band':tile_data.coords['band'],
#                                     'y':tile_data[0].coords['y'],'x':tile_data[0].coords['x']})
#         return tile

    def _generate_cutouts(self):
        """
        Iterate over (a selection of) the tiles yielding all cutouts for each
        of them.
        Apply balancing to cutouts
        """

        for tile in self.tiles:
            gc.collect()  # solves memory leak when dataset is used within fit

            # read tile - floats are required to mask with NaN's
            da = rioxr.open_rasterio(tile).astype("float32")

            # mask ROI (ocean)
            if self.mask is not None:
                mask = self.mask.to_crs(da.spatial_ref.crs_wkt)
                geometry = mask.unary_union.buffer(self.buffer)
                da = da.rio.clip([geometry], drop=True, invert=self.invert,
                                 all_touched=self.all_touched)

            # TO DO: add labels as new band (for balancing)
            # if self.balance_ratio >0:
            #     da = add_labels_band(da)

            # apply offset
            da = da.shift(x=self.offset, y=self.offset)
            da['x'] = da.x.shift(x=self.offset)
            da['y'] = da.y.shift(y=self.offset)

            # generate windows
            da = da.rolling(x=self.cutout_size, y=self.cutout_size)
            da = da.construct({'x': 'x_win', 'y': 'y_win'}, stride=self.stride)

            # drop NaN-containing windows
            da = da.stack(sample=('x', 'y'))
            da = da.dropna(dim='sample', how='any') # (band, x_win, y_win, sample)

            # balance the ratio of labelled/unlabelled windows
            #   balance_ratio = N_labelled / N_unlabelled  (0 = no blancing, 1 = equal balance)
#             if self.balance_ratio > 0:

#                 # TO DO: make sure labels are included as a band in the DA
#                 # tile_cutouts = da

#                 idx_labels = da.isel(band=-1) == 1   # all labelled pixels (labels are added as last band)
#                 labelled_windows = idx_labels.sum(('x_win','y_win')) > 0 # boolean: identify all windows that have at least one labelled

#                 # separate labelled and unlabelled windows
#                 cutouts_label_1 = da.isel(sample=labelled_windows.values) # labelled
#                 cutouts_label_0 = da.isel(sample=~labelled_windows.values)# unlabelled

#                 N_label_1 = cutouts_label_1.isel(band=0,x_win=0,y_win=0).shape[0] # number of labelled windows (band=0 could be any band)
#                 N_label_0 = int(1/self.balance_ratio * N_label_1) # number of unlabelled windows dependingn on balance ratio

#                 # # select the windows according to the ratio
#                 # # TO DO: select windows in a random order
#                 data_train_label_1 = cutouts_label_1.isel(sample=np.arange(N_label_1))
#                 data_train_label_0 = cutouts_label_0.isel(sample=np.arange(N_label_0))

#                 da = xr.concat((data_train_label_1,data_train_label_0),dim='sample')


            # select bands
            if self.bands is not None:
                if type(self.bands) is not list:
                    da = da.sel(band=[self.bands])
                else:
                    da = da.sel(band=self.bands)
            # elif self.balance_ratio > 0: # remove added label-band
            #     da = da.sel(band=list(range(da.sizes['band']-1)) )

            # normalize
            if self.norm_threshold is not None:
                # da = (da + 0.1) / (self.norm_threshold + 1) # normalises to [0 1] using  v_max= norm_threshold nd v_min=0, but with a small addition to omit 0/x
                # da = da.clip(max=1)
                if len(self.norm_threshold) ==1:
                    norm_min=0
                    norm_max=self.norm_threshold
                elif len(self.norm_threshold) == 2:
                    norm_min, norm_max = self.norm_threshold
                    
                # normliise using norm_min nd norm_max values
                da = (da - norm_min) / (norm_max - norm_min)
                da = da.clip(min=0,max=1)
                print('Normalised to {:.1f}-{:.1f}'.format(da.min().values, da.max().values) )

            yield (
                da.sample.coords['x'],
                da.sample.coords['y'],
                da.data.transpose(3, 1, 2, 0)  # samples, x_win, y_win, bands
            )
