import ee

import configparser
import os
import sys
import json
import ast 
# ee.Authenticate()
ee.Initialize()


# Load libraries and assets
gridTiles_iceShelves = ee.FeatureCollection('projects/ee-izeboudmaaike/assets/gridTiles_iceShelves')
iceShelves = ee.FeatureCollection('users/izeboudmaaike/ne_10m_antarctic_ice_shelves_polys')


''' -----------------
    Functions
---------------------'''

def maskS2clouds(image):
    qa = ee.Image(image).select('QA60')
    
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
        .And(qa.bitwiseAnd(cirrusBitMask).eq(0)) \
        .focal_min(radius=20e3, units='meters') # Add (generous) buffer to mask
    return image.updateMask(mask)



def main(configFile):
    '''This function exports Sentinel-1 data from the Google Earth Engine (GEE) to the Google Cloud Storage Bucket (GCS).
    It exports individual relative-orbit files.
    A configuration file is used to specify user input, such as:
    - Which Sentinel-1 subset to use (orbit properties, temporal range, optional spatial range)
    - The path of the GCS bucket
    - Options to clip to coastline.
    '''


    ''' ---------------------------------------------------------------------------
            Configuration
    -------------------------------------------------------------- '''

    if configFile is None:
        raise NameError('No config file specified. Run script as "python this_script.py /path/to/config_file.ini"')
    else:
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(os.path.join(configFile))

    my_bucket = config['PATHS']['gcloud_bucket']        # full path:   e.g. ee-export_s1_relorbs/path/to/dir
    bucket_base   = my_bucket.split('/')[0]             # main bucket: e.g. ee-export_s1_relorbs
    bucket_subdir = my_bucket.replace(bucket_base,'').lstrip('/')    # e.g. path/to/dir/
    if bucket_subdir: # if string is not empty, make sure subdir ends with a trailing "/"
        if bucket_subdir[-1] != '/':
            bucket_subdir += '/' # add trailing "/" if not present

    t_strt = config['DATA']['t_strt']
    t_end = config['DATA']['t_end']
    bnds = config['DATA']['bnds'] 
    CRS = config['DATA']['CRS']
    scale = int(config['DATA']['imRes']) # test scale
    start_export = True if config['DATA']['start_export'] == 'True' else False

    filename_prefix = 'S2_MGRStile_' 

    ## Filter info
    meta_filters_lt = config._sections['METAFILTERS_less_than'] # reads items to dict; but reads items as string isntead of values
    for key in meta_filters_lt.keys():
        meta_filters_lt[key] = ast.literal_eval(meta_filters_lt[key]) # use ast instead of json, as this has no issues with loading "None" or boolean values
    print(meta_filters_lt) 

    meta_filters_gt = config._sections['METAFILTERS_greater_than']
    for key in meta_filters_gt.keys():
        meta_filters_gt[key] = ast.literal_eval(meta_filters_gt[key]) # use ast instead of json, as this has no issues with loading "None" or boolean values
    print(meta_filters_gt)

    # print('Bucket:  {}'.format(bucket_base))
    # print('Bucket subdir:  {}'.format(bucket_subdir))

    ''' ---------------------------------------------------------------------------
            Select all images
    -------------------------------------------------------------- '''

    # Load collection & apply standard metadata filters
    imCol = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(gridTiles_iceShelves) \
            .filterDate(t_strt, t_end) \
            # .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 20) \
            # .sort('CLOUDY_PIXEL_PERCENTAGE', True)

    # Further improve image collection filters
    # col_improved = col.filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 1) \
    #                 .filterMetadata('THIN_CIRRUS_PERCENTAGE', 'less_than', 1) \
    #                 .filterMetadata('DARK_FEATURES_PERCENTAGE', 'less_than', 10) \
    #                 .filterMetadata('NODATA_PIXEL_PERCENTAGE', 'less_than', 20) \
    #                 .filterMetadata('SNOW_ICE_PERCENTAGE', 'greater_than', 10)
    
    for filter_property in meta_filters_lt.keys():
        filter_value = meta_filters_lt[filter_property]
        imCol = imCol.filterMetadata(filter_property.upper(), 'less_than', filter_value )
    

    for filter_property in meta_filters_gt.keys():
        filter_value = meta_filters_gt[filter_property]
        imCol = imCol.filterMetadata(filter_property.upper(), 'greater_than', filter_value )

    imCol = imCol.map(maskS2clouds)

    # col_distinct = ee.ImageCollection(col_best.distinct('MGRS_TILE'))
    fCol_MGRStiles = imCol.distinct('MGRS_TILE')
    imgs_list = fCol_MGRStiles.aggregate_array('system:index').getInfo() # featCol to list
    
        
    ''' ---------------------------------------------------------------------------
            Export images to Cloud Bucket
    -------------------------------------------------------------- '''

        
    im_task_list = []

    print('.. Export {} MGRS tiles ({}m) to gcloud bucket {} '.format(len(imgs_list), scale,  my_bucket ))
    # for i in range(0,5): #len(imgs_list)):
    for i in range(5,len(imgs_list)):
        # get img
        imName = imgs_list[i] # select single img id from orbits_to_use
        eeImg = ee.Image('COPERNICUS/S2_SR_HARMONIZED/' + imName).select(['B4','B3','B2','B11'])

        # eeImg_meta = eeImg.getInfo() # reads metadata    
        
        # export filename
        file_name = filename_prefix + imName 

        print(file_name)
        
        export_geometry = eeImg.geometry()
            
        im_task = ee.batch.Export.image.toCloudStorage(
                image = eeImg, # do not export toByte; then NaN mask will be converted to 0
                description = 'export_'+file_name.split('/')[-1],
                fileNamePrefix = bucket_subdir + file_name,
                scale= scale,
                bucket= bucket_base,
                crs=CRS,
                maxPixels=1e10,
                region=export_geometry
        )

        # -- start tasks at GEE editor
        if start_export:
            im_task.start()
        
        # store img tasks in list to check status later
        im_task_list.append(im_task)
         
    if not start_export:
        print('.. Did not start export tasks; set start_task to True')      
    print('Done')


    ''' ---------------------------------------------------------------------------
            Images are uploading to gcloud bucket.
            This might take a while.
    -------------------------------------------------------------- '''

if __name__ == '__main__':
    #  Run script as "python path/to/script.py /path/to/config_file.ini"
        
    # retrieve config filename from command line
    config = sys.argv[1] if len(sys.argv) > 1 else None

    # run script
    main(config)   