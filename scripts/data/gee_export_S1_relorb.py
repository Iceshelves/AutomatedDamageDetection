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

# Functions
def add_rel_orb_num_cycle(eeImage):
    relorb_slice = ee.Number(eeImage.get('relativeOrbitNumber_start')).format('%.0f').cat('_').cat(ee.Number(eeImage.get('sliceNumber')).format('%.0f'))
    return eeImage.set('relorb_slice', relorb_slice)

def get_imCol_relorbs(image):
    s1_relorb = ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.eq("system:index", image.getString('relorb_id')))
    return ee.Image(s1_relorb.first())

def merge_slices_per_day(imCol, datelist):
    day_mosaic_list = []
    for day in datelist:
        day_slices = imCol.filterDate(ee.Date(day), ee.Date(day).advance(1,'day'))
        slice_numbers = day_slices.aggregate_array('sliceNumber').distinct()
        properties = {'date': ee.Date(day).format('YYYY-MM-dd'),
                      'slice_numbers': slice_numbers.getInfo(),
                      'slice_ids': day_slices.aggregate_array('system:index').distinct().getInfo()
                      }
        day_mosaic_list.append(day_slices.mosaic().set(properties) )
    return day_mosaic_list

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

    scale = int(config['DATA']['imRes']) # test scale
    mode = config['DATA']['mode']
    orbit = config['DATA']['orbitPass']
    t_strt = config['DATA']['t_strt']
    t_end = config['DATA']['t_end']
    bnds = config['DATA']['bnds'] 
    CRS = config['DATA']['CRS']
    start_export = True if config['DATA']['start_export'] == 'True' else False
    
    relorb_num = int(config['DATA']['relorbNumber'])

    # Select ROI
    PineIsland = ee.Geometry.Polygon(
        [[[-102.02909017852227, -74.76139200530943],
        [-102.02909017852227, -75.46684365487192],
        [-98.44754720977227, -75.46684365487192],
        [-98.44754720977227, -74.76139200530943]]], None, False)
    export_geometry = PineIsland

    print('Loaded settings: \n \
        mode:        {}\n \
        bucket:      {}\n \
        orbitPass:   {}\n \
        bands:       {}\n \
        dateRange:   {} to {}\n \
        relorb_num:  {} \
        '.format(mode,my_bucket,orbit, bnds,t_strt, t_end ,relorb_num))  

    ''' ---------------------------------------------------------------------------
            Select all images
    -------------------------------------------------------------- '''

    # Load collection & apply standard metadata filters
    imCol = (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterDate(t_strt, t_end)
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', bnds))
            .filterMetadata('instrumentMode', 'equals', mode)
            .filter(ee.Filter.eq('orbitProperties_pass', orbit))
            .filterBounds(iceShelves)
            .map(add_rel_orb_num_cycle))

    # select relorb number
    imCol_relorb_N = imCol.filterMetadata('relativeOrbitNumber_start', 'equals', relorb_num)

    print(".. number of imgs: ",  imCol_relorb_N.size().getInfo() )

    dateList = (imCol_relorb_N.aggregate_array('system:time_start')
                .map(lambda dateMillis: ee.Date(dateMillis).format('YYYY-MM-dd')) # convert to readable format
                .distinct()).getInfo()
    # print('.. distinct dates: ', dateList.sort().size().getInfo(), dateList.getInfo() )
    print('.. distinct dates: ', len(dateList), dateList )

    # Merge slices for distinct dates
    imgs_list = merge_slices_per_day( imCol_relorb_N, dateList )
    # print('.. e.g. merged slices of orbit first day: ', imgs_list[0].get('slice_numbers').getInfo())

    ''' ---------------------------------------------------------------------------
            Export images to Cloud Bucket
    -------------------------------------------------------------- '''
    # raise RuntimeError('Stop here - dev')
    
    im_task_list = []

    print('.. Export {} relorbs ({}m) to gcloud bucket {} '.format(len(imgs_list), scale,  my_bucket ))

    for i in range(0,len(imgs_list)):

        eeImg =imgs_list[i].select(bnds)

        # export filename
        slice_ids = eeImg.get('slice_ids').getInfo() # list with relorb IDs
        img_date_base = slice_ids[0].split('T')[0] # get common substring
        imName = img_date_base + '_orbit-'+ str(relorb_num) +'_sliceMosaic_' + str(scale) + 'm' 
        file_name = imName 

        print(file_name)
        # print('.. merged slices of orbit for this date: ', eeImg.get('slice_numbers').getInfo())
        
        # export_geometry =eeImg.geometry()
            
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