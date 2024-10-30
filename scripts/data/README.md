# Data directory

This directory contains the scripts used to transfer data between platforms.

## Used platforms:
- dcache : SURF dCache storage
- Google Cloud Storage (GCS)
- Google Earth Engine (GEE) Assets

## File pipeline:
1. Filter and select satellite images in the Google Earth Engine; these are exported to GCS. Script: https://code.earthengine.google.com/d82dedd43de7f0b77cb2e0160fc2198d , gee_export_S2_MGRS_tiles.py and gee_export_S1_relorb.py.
2. Move images from GCS to dCache. Script: gcs-to-dcache.py
3. Acces images on dCache from computing cluster to perform VAE training
4. When predictions of damage detection are created, upload to GCS (using `gsutil`).
5. Upload prediction images from GCS to GEE Asset for easy viewing in GEE. Script: upload_GCS-to-GEEasset.py
6. View output at GEE: https://code.earthengine.google.com/9af710497fac6b76c710873839e797b6

All *.ini files in this directory convey the locations of the data in the storage.
Apart from the GEE Asset files, the data storages are not publicly accessible.

## GEE Assets IDs:
- Datatiles of Sentinel-2 composite images (from Dec-Jan-Feb 2019-2020): `users/izeboudmaaike/damage_detection/VAE_damage_detection/S2_composite_2019-11-1_2020-3-1_dataTiles`
- Data of Sentinel-2 MGRS images (individual acquisitions) used to apply VAE to: `users/izeboudmaaike/damage_detection/VAE_damage_detection/S2_MGRStiles_2019-11-01_2020-04-01`
- Predicted damage on Sentinel-2 MGRS & Sentinel-2 composite images: `users/izeboudmaaike/damage_detection/VAE_damage_detection/S2_composite_2019-11-1_2020-3-1_predicted_dmg`
