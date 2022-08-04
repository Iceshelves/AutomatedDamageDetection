import pathlib

import geopandas as gpd
import pandas as pd
import pystac


def _read_tile_catalog(catalog_path):
    """ Read the tile catalog """
    catalog_path = pathlib.Path(catalog_path)
    catalog_path = catalog_path / "catalog.json"
    return pystac.Catalog.from_file(catalog_path.as_posix())


def _catalog_to_geodataframe(catalog, crs="WGS84"):
    """ Convert STAC catalog to a GeoDataFrame object """
    features = {item.id: item.to_dict() for item in catalog.get_all_items()}
    gdf = gpd.GeoDataFrame.from_features(features.values())
    gdf.index = features.keys()
    for column in gdf.columns:
        if 'datetime' in column:
            gdf[column] = pd.to_datetime(gdf[column])
    gdf = gdf.set_crs(crs)
    return gdf


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


def _get_tile_paths(catalog, item_ids, asset_key):
    """ Extract the asset paths from the catalog """
    items = (catalog.get_item(item_id, recursive=True) for item_id in item_ids)
    assets = (item.assets[asset_key] for item in items)
    return [asset.get_absolute_href() for asset in assets]


def _filter_labels(labels, start_datetime, end_datetime, verbose=True):
    """ Select the labels whose date in the provided datetime range """
    mask = (labels.Date >= start_datetime) & (labels.Date <= end_datetime)
    if verbose:
        print("Selecting {} out of {} labels".format(mask.sum(), len(labels)))
    return labels[mask]


# def split_train_and_test(catalog_path, test_set_size, labels_path=None,
#                          validation_set_size=None, random_state=None,
#                          verbose=True):
#     """
#     The tiles in the provided STAC catalog are split in test, validation and
#     training sets.

#     :param catalog_path: STAC catalog path
#     :param test_set_size: size of the test set
#     :param labels_path: path to the labels. If provided, all tiles overlapping
#         with the labels will be included in the test set
#     :param validation_set_size: size of the validation set
#     :param random_state: random state for the data set sampling
#     :param verbose: if True, print info to stdout
#     """

#     # read tile catalog
#     catalog = _read_tile_catalog(catalog_path)
#     tiles = _catalog_to_geodataframe(catalog)

#     # read labels and reserve the labeled tiles for the test set
#     test_set = gpd.GeoDataFrame()
#     if labels_path is not None:
#         labels = _read_labels(labels_path, verbose)
#         labels = labels.to_crs(tiles.crs)  # make sure same CRS is used

#         # select the only labels matching the tiles timespan
#         labels = _filter_labels(labels,
#                                 tiles.start_datetime.min(),
#                                 tiles.end_datetime.max())

#         # add the tiles overlapping with the labels to the test set
#         labels = labels[labels.is_valid] # remove invalid geometries
#         mask = tiles.intersects(labels.unary_union) # boolean
#         test_set = test_set.append(tiles[mask])

#         if verbose:
#             print("{} tiles overlap with labels: ".format(len(test_set)))
#             for tile in test_set.index:
#                 print(tile)

#         if len(test_set) > test_set_size:
#             raise ValueError(
#                 "Labels overlap with {} tiles while a test set size of {} was "
#                 "selected - please increase `test_set_size` to >= {}.".format(
#                     len(test_set),
#                     test_set_size,
#                     len(test_set)
#                 )
#             )

#         tiles = tiles[~mask]

#     # pick additional unlabeled tiles for the test set
#     test_set_unlabeled = tiles.sample(test_set_size - len(test_set),
#                                       random_state=random_state)
#     test_set = test_set.append(test_set_unlabeled)
#     test_set_paths = _get_tile_paths(catalog, test_set.index, "B2-B3-B4-B11")

#     train_set = tiles.index.difference(test_set_unlabeled.index)
#     train_set = tiles.loc[train_set]

#     # split validation set and training set
#     val_set = gpd.GeoDataFrame()
#     if validation_set_size is not None:
#         val_set = val_set.append(
#             train_set.sample(validation_set_size, random_state=random_state)
#         )
#         train_set = train_set.index.difference(val_set.index)
#         train_set = tiles.loc[train_set]

#     train_set_paths = _get_tile_paths(catalog, train_set.index, "B2-B3-B4-B11")
#     val_set_paths = _get_tile_paths(catalog, val_set.index, "B2-B3-B4-B11")

#     return train_set_paths, val_set_paths, test_set_paths


# def split_train_and_test(catalog_path, test_set_size, labels_path=None,
#                          validation_set_size=None, random_state=None,
#                          verbose=True):
#     """
#     The tiles in the provided STAC catalog are split in test, validation and
#     training sets.

#     :param catalog_path: STAC catalog path
#     :param test_set_size: size of the test set
#     :param labels_path: path to the labels. If provided, all tiles overlapping
#         with the labels will be included in the TRAINING set (to balance the data)
#     :param validation_set_size: size of the validation set
#     :param random_state: random state for the data set sampling
#     :param verbose: if True, print info to stdout
#     """

#     # read tile catalog
#     catalog = _read_tile_catalog(catalog_path)
#     tiles = _catalog_to_geodataframe(catalog)

#     # read labels and reserve the labeled tiles for the test set # Changed: use them for training set, to get a better balanced traainiing set.
#     train_set = gpd.GeoDataFrame()
#     test_set = gpd.GeoDataFrame()
#     if labels_path is not None:
#         labels = _read_labels(labels_path, verbose)
#         labels = labels.to_crs(tiles.crs)  # make sure same CRS is used

#         # select the only labels matching the tiles timespan
#         labels = _filter_labels(labels,
#                                 tiles.start_datetime.min(),
#                                 tiles.end_datetime.max())

#         # add the tiles overlapping with the labels to the test set
#         labels = labels[labels.is_valid] # remove invalid geometries
#         mask = tiles.intersects(labels.unary_union) # boolean
#         train_set = train_set.append(tiles[mask])

#         if verbose:
#             print("{} tiles overlap with labels: ".format(len(train_set)))
#             for tile in train_set.index:
#                 print(tile)

#         if len(train_set) > test_set_size:
#             raise ValueError(
#                 "Labels overlap with {} tiles while a test set size of {} was "
#                 "selected - please increase `test_set_size` to >= {}.".format(
#                     len(train_set),
#                     test_set_size,
#                     len(train_set)
#                 )
#             )

#         test_tiles = tiles[~mask]

#     # pick additional unlabeled tiles for the test set # Chaanged: for the training set
#     train_set_unlabeled = tiles.sample(test_set_size - len(train_set),
#                                       random_state=random_state)
#     train_set = train_set.append(train_set_unlabeled)
    

#     # test_set = tiles.index.difference(test_set_unlabeled.index)
#     test_set = test_tiles.sample(test_set_size,random_state=random_state) # pick N tiles for test set.
#     test_set = test_tiles.append(test_set)

#     # split validation set and training set
#     val_set = gpd.GeoDataFrame()
#     if validation_set_size is not None:
#         val_set = val_set.append(
#             train_set.sample(validation_set_size, random_state=random_state)
#         )
#         train_set = train_set.index.difference(val_set.index)
#         train_set = tiles.loc[train_set]

#     train_set_paths = _get_tile_paths(catalog, train_set.index, "B2-B3-B4-B11")
#     test_set_paths = _get_tile_paths(catalog, test_set.index, "B2-B3-B4-B11")
#     val_set_paths = _get_tile_paths(catalog, val_set.index, "B2-B3-B4-B11")

#     return train_set_paths, val_set_paths, test_set_paths

def split_train_and_test(catalog_path, 
                         dmg_px_count_file=None, select_dmg_quantile=0.9,
                         validation_split=0.2,
                         random_state=None, 
                         verbose=True):
    
    """
    The tiles in the provided STAC catalog are split in test, validation and
    training sets.

    :param catalog_path: STAC catalog path
    :param test_set_size: size of the test set
    :param labels_path: path to the labels. If provided, all tiles overlapping
        with the labels will be included in the TRAINING set (to balance the data)
    :param random_state: random state for the data set sampling
    :param verbose: if True, print info to stdout
    
    UPDATE: balance using a dmg indication, couonting damage pixels from NeRD ouotput on RAAMP mosaic
    :param select_dmg_qntl: select upper quantile of the dmg_px_count information for training data
    :param dmg_px_count_file: pickle file (.pkl) containing pre-calculated indicator for amount of dmg on each tile
    :param validation_split: provide a training:validation split ratio instead of set size 
    """

    
    # read tile catalog
    catalog = _read_tile_catalog(catalog_path)
    tiles = _catalog_to_geodataframe(catalog) # gpd f
    tilelist = tiles.index.values.tolist()    # tilenames in list (excluding .tif)
    
    # remove ROI tiles from tilelist, before dividing training/test/val
    tileNums_test_ROI = [112,122,126,139,140,141,142,143,151,152,153,154]
    tilelist_ROI = [item for item in tilelist if int(item.split('_')[-1]) in tileNums_test_ROI]
    # make SET from ROI tiles
    ROI_set = tiles.loc[tilelist_ROI]
    # remove these from tilelist
    tiles_set = tiles.index.difference(tilelist_ROI) 
    tiles_set = tiles.loc[tiles_set]
    tiles = tiles_set
    
    
    # read the dmg-indicator file for all tiles
    train_set = gpd.GeoDataFrame()
    if dmg_px_count_file is not None: # dmg file: "RAMP_tiled_dmg_px_count .pkl or .csv"
        # tiled_dmg_px_count = pd.read_pickle(dmg_px_count_file)
        tiled_dmg_px_count = pd.read_csv(dmg_px_count_file, index_col=0) 
        print(tiled_dmg_px_count)
        
        # remove tiles that are in test_ROI also from DMG count
        ROI_tiles = [True if item not in tileNums_test_ROI else False for item in tiled_dmg_px_count['Tile_Num'].values] # TRUE if tile can be kept, FALSE if
        tiled_dmg_px_count = tiled_dmg_px_count[ROI_tiles]
        

        # select tiles with most damage
        df_mask= tiled_dmg_px_count['Dmg_px_count']>=tiled_dmg_px_count['Dmg_px_count'].quantile( select_dmg_quantile )
        tiles_heavydmg = tiled_dmg_px_count[df_mask]
        tile_nums_heavydmg = tiles_heavydmg['Tile_Num'].values # array with strings 'N'
        
        for tile_N in tile_nums_heavydmg:
            # tile_N = 'tile_' + tile_N   # add prefix to string to differentiate number '10' from '100' and '110' etc
            tile_N = 'tile_' + str(tile_N)
            corresponding_file = [file  for file in tilelist if file.endswith(tile_N) ]
            
            # add to train_set
            train_set = pd.concat([train_set, tiles.loc[corresponding_file]])
            
    elif dmg_px_count_file is None: # no dmg information (for balance), choose random tiles
        train_set_size = int(0.7*len(tiles)) # get 70% of all tiles for training
        train_set = tiles.sample( train_set_size , random_state=random_state)
    
    # select testing data: tiles that are not selected for train/val              
    test_set = tiles.index.difference(train_set.index) 
    test_set = tiles.loc[test_set] 
    test_set_paths = _get_tile_paths(catalog, test_set.index, "B2-B3-B4-B11")                      
  
    # split train_set in training & validation set
    val_set = gpd.GeoDataFrame()
    
    if validation_split > 0.5: # revise ratio: training:validation should be ~80:20 (corresponds to validation_split=0.2) 
        validation_split = 1-validation_split
    validation_set_size = int(validation_split * len(train_set ) ) # round to nearest int
    
    val_set = pd.concat([val_set, 
                         train_set.sample( validation_set_size, random_state=random_state)] )

    train_set = train_set.index.difference(val_set.index)
    train_set = tiles.loc[train_set]

    train_set_paths = _get_tile_paths(catalog, train_set.index, "B2-B3-B4-B11")
    val_set_paths = _get_tile_paths(catalog, val_set.index, "B2-B3-B4-B11")    
        
    if verbose:
        print("{} high-dmg tiles selected for training: ".format(len(train_set) ) )
        for tile in train_set.index:
            print(tile)
        print("{} high-dmg tiles selected for validation: ".format(len(val_set) ) )
        for tile in val_set.index:
            print(tile)

    return train_set_paths, val_set_paths, test_set_paths