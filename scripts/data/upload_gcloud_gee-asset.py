#!/usr/bin/env python3.9
'''
upload files from gcloud bucket to GEE asset

'''

import sys
import configparser
import os
import subprocess
import ee
import re
# import rioxarray as rioxr
# import geopandas as gpd

from datetime import datetime

ee.Initialize()

''' ----------------------------------------------------------
         Functions
-------------------------------------------------------------- '''
def is_datetime(datetime_str,date_format='%Y%m%dT%H%M%S'):
    try:
        return datetime.strptime(datetime_str,date_format )
    except ValueError:
        # skip parts that are not datetime
        pass

''' ----------------------------------------------------------
         Main
-------------------------------------------------------------- '''

def main(configFile):

    ''' -------
    Configuration 
    -----------'''
    
    if configFile is None:
        raise NameError('No config file specified. Run script as "python this_script.py /path/to/config_file.ini"')
    else:
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(os.path.join(configFile))

    gcloud_dir = config['ASSET']['gcloud_bucket']
    asset_ID = config['ASSET']['gee_assetID'] 
    descript = asset_ID.split('/')[-1] # 'S1_PiG_40m_toAsset'
    # include_vars = [varName.strip() for varName in config['ASSET']['include_vars'].split(',')] # ['crevSig','alphaC','dmg'] 
    imRes = int(config['DATA']['imRes'])
    # Npix = int(config['DATA']['nPix'])
    CRS = config['DATA']['CRS']
    start_upload = True if config['DATA']['start_upload'] == 'True' else False

    # -- settings following from config

    pattern ='*.tif' # '*'+str(imRes)+'m*'+str(Npix)+'px_'+'*' # e.g.: '*40m*10px*'
    export_scale = imRes

    ''' ----------------------------------------------------------
            Find all files with pattern in gcloud
    -------------------------------------------------------------- '''

    try:
        content_clouddir = subprocess.run('gsutil ls ' + gcloud_dir + pattern ,shell=True , check=True, capture_output=True, text=True).stdout.splitlines()
    except subprocess.CalledProcessError: 
        # error raised when Command 'gsutil ls gs://dir/*' returned non-zero exit status 1.
        # pattern_oldfmt = '*_'+str(imRes*Npix)+'m*' # e.g.: '*40m*10px*' # RAMP / OLD FORMAT
        print('No files found for pattern {}')# ; try old format pattern {}'.format(pattern,pattern_oldfmt))
        # content_clouddir = subprocess.run('gsutil ls ' + gcloud_dir + pattern_oldfmt ,shell=True , check=True, capture_output=True, text=True).stdout.splitlines()

    # all imgs
    gcloud_tif_list = [file for file in content_clouddir if file.endswith('.tif')]
    # gcloud_tif_crevSig = gcloud_tif_list
    
    n_files = len(gcloud_tif_list)

    print('.. Found {} files in gcloud {}'.format(n_files,gcloud_dir) )
    

    ''' ----------------------------------------------------------
            Check asset list for already uploaded imgs
    -------------------------------------------------------------- '''    

    collection_path = asset_ID
    assets_list = ee.data.getList(params={"id": collection_path}) # list with dictionaries {'type':.., 'id':...}
    assets_list  = [d['id'] for d in assets_list] # get list of id's
    
    # assetst are uploaded as  'assetId': asset_ID + '/' + fileName  
    # where fileName is e.g. relorb_S1A_EW_GRDH_1SSH_20190908T233549_20190908T233649_028936_0347F1_B030_40m_output_10px
    assetIDs_uploaded = [ os.path.basename(assetID) for assetID in  assets_list] # contains all imgs in collection

    # get gcFiles, strip path fileName so that it can be compared to the available asset IDs 
    
    ## remove uploaded imgs: if imCol img name is same as name in GCS
    gcloud_tif_toUpload = [gcFile for gcFile in gcloud_tif_list if os.path.basename(gcFile).removesuffix('.tif') not in assetIDs_uploaded]
    
    ## remove uploaded imgs: for predicted imCol, the imCol img name is different than on GCS
    if "S2" in os.path.basename(gcloud_tif_list[0]) :
        if "predict" in os.path.basename(gcloud_tif_list[0]):
            # emulate the to-be asset-names based on the GCS file
            gc_to_asset_fnames = [gcFile.split('_tile')[0] + '_VAE-predicted-dmg_' + 'tile_' + re.search('tile_(.+?)_', gcFile).group(1) 
                                    for gcFile in gcloud_tif_list]
            # look for existing names in asset, and keep respective list of GCS files
            gcloud_tif_toUpload = [gcFile for (gcFile, gcAsset) in zip(gcloud_tif_list,gc_to_asset_fnames) if os.path.basename(gcAsset).removesuffix('.tif') not in assetIDs_uploaded]
            print('example GCS file: ', gcloud_tif_toUpload[0])
    print('.. of which to upload: {}'.format(len(gcloud_tif_toUpload)))
    
    ''' ----------------------------------------------------------
            Export all images to Asset
    -------------------------------------------------------------- '''
    # if not start_upload:
    #     print('.. Did not start upload tasks; set start_task to True')   
    #     exit()

    # files_to_upload = gcloud_tif_crevSig # maybe filter this list for files that already exist? -- if img aalready exists in imCol the task yields an error
    print('.. Uploading to Asset: {}'.format(asset_ID))
    counter = 1
    for gcFile in gcloud_tif_toUpload:
        # print('upload {}/{}'.format(counter, n_files ) )

        ''' -----------------
        Load data for all variables
        ---------------------'''

        # -- load img
        cloudImage = ee.Image.loadGeoTIFF(gcFile)

        bandNames = cloudImage.bandNames().getInfo()
        if len(bandNames) == 1: # if img has 1 band, assume it is predicted dmg band
            cloudImage = cloudImage.rename('dmg')
            bandName = 'dmg'

        
        ''' -----------------
        Add metadata to img
        ---------------------'''

        img_name = os.path.basename(gcFile)

        if "S2_composite" in img_name:
            if "predict" in img_name :
                # print(img_name.split('_'))
                # print([is_datetime(part,date_format='%Y-%m-%d') for part in img_name.split('_')])

                # infer information from img name
                img_dates = [is_datetime(part,date_format='%Y-%m-%d').strftime("%Y-%m-%d") \
                            for part in img_name.split('_') \
                                if is_datetime(part,date_format='%Y-%m-%d') is not None] # date in 'YYYY-mm-dd'

                tileNum = 'tile_' + re.search('tile_(.+?)_', img_name).group(1)

                model_id = 'model_'+ re.search('model_(.+?)_', img_name).group(1) 
                epoch_num = 'epoch_'+re.search('epoch(.+?)_', img_name).group(1) 
                img_date_start = img_dates[0]
                img_date_end = img_dates[1]

                cloudImage = cloudImage.set({'system:time_start':ee.Date(img_date_start).millis(), 
                                            'system:time_end':ee.Date(img_date_end).millis(), 
                                            'tileNum':tileNum,
                                            'model_info':{'model_id':model_id,
                                                        'epoch_num':epoch_num}
                                            })   

                fileName = img_name.split('_tile')[0] + '_VAE-predicted-dmg_' + tileNum  
        
            else: # handling original data tile
                fileName = img_name.split('.')[0] # remove .tif extension
                print(fileName)
        elif img_name.startswith("S2_MGRStile"): # in img_name:
            ee_img_id = img_name.replace('S2_MGRStile_','')
            # define filename for in Asest Collection
            fileName = img_name.split('.')[0]  # remove .tif extension

            # infer information from img name
            img_dates = [is_datetime(part,date_format='%Y%m%dT%H%M%S').strftime("%Y-%m-%d") \
                            for part in img_name.split('_') \
                                if is_datetime(part,date_format='%Y%m%dT%H%M%S') is not None] # date in 'YYYY-mm-dd'

            model_id = 'model_'+ re.search('model_(.+?)_', img_name).group(1) 
            epoch_num = 'epoch_'+re.search('epoch(.+?)_', img_name).group(1) 

            img_date_start = img_dates[0]
            img_date_end = img_dates[1]

            cloudImage = cloudImage.set({'system:time_start':ee.Date(img_date_start).millis(), 
                                         'system:time_end':ee.Date(img_date_end).millis(), 
                                         'input_img_source':'S2_SR_HARMONIZED',
                                         'input_img_ID':ee_img_id,
                                         'model_info':{ 'model_id':model_id,
                                                        'epoch_num':epoch_num},
                                        })   

        # TMP: export mask separately. TO DO: include this in uploading S2_MGRStile as a band
        if img_name.startswith('mask_S2_MGRStile'):
            ee_img_id = img_name.replace('mask_S2_MGRStile_','')
            cloudImage = cloudImage.rename('mask')
            bandName = 'mask'
            # define filename for in Asest Collection
            fileName = img_name.split('.')[0]  # remove .tif extension

            # infer information from img name
            img_dates = [is_datetime(part,date_format='%Y%m%dT%H%M%S').strftime("%Y-%m-%d") \
                            for part in img_name.split('_') \
                                if is_datetime(part,date_format='%Y%m%dT%H%M%S') is not None] # date in 'YYYY-mm-dd'
            img_date_start = img_dates[0]
            img_date_end = img_dates[1]
            cloudImage = cloudImage.set({'system:time_start':ee.Date(img_date_start).millis(), 
                                         'system:time_end':ee.Date(img_date_end).millis(), 
                                         'input_img_source':'S2_SR_HARMONIZED',
                                         'input_img_ID':ee_img_id,
                                         'bandNames':bandName,
                                        })   

        else:
            raise Exception('Could not include metadata; only accounted for predicted S2_composites / S2_MGRStiles ' )

        ''' -----------------
        Create upload task
        ---------------------'''

        # fileName = img_name = img_name.split('_tile')[0] + '_VAE-predicted-dmg_' + tileNum  # img_name.replace('_crevSig','').split('.')[0] # filename without varName and file extention
        
        # print('.. asset name: ', fileName) 
        # print('.. upload descript: ', descript+str(counter) )
        
        # TO DO: handle already uploaded imgs in imCol
        task = ee.batch.Export.image.toAsset(**{
            'image': cloudImage,
            'description': descript+'-'+str(counter),
            'assetId': asset_ID + '/' + fileName , # if img already exists in imCol, the task yields an error
            'scale': export_scale,
            'crs': CRS,
            'region':cloudImage.geometry()
        })
        
        if start_upload:
            print('.. started upload: ', fileName)
            task.start()
        counter=counter+1

    if not start_upload:
        print('.. Did not start upload tasks; set start_task to True')    
    print('Done')


if __name__ == '__main__':
    #  Run script as "python path/to/script.py /path/to/config_file.ini"
        
    # retrieve config filename from command line
    config = sys.argv[1] if len(sys.argv) > 1 else None

    # run script
    main(config)    