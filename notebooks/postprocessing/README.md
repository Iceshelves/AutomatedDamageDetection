
**NB**: as these notebooks serve as a reference to insight in development steps, it is advised not to run the notbeooks anew (which won't work) but to view the visible output (plots etc) withiin the notebook. 

## Files

- extract_lspace_clusters_L2.ipynb : notebook to visualise the latent space and explore the extraction of clusters. Notebook contains some code that is left-over from earlier tests as reference (it's a bit of a mess).
- extract_lspace_clusters_model_1670420388.ipynb : similar notebook as the other, but displays the cluster extraction threshold for the specified model.
- load_history.ipynb : read and plot performance of VAE training history per epoch; notebook used for the development of scripts/postprocessng/plot_history_loss.py  
- plot_windows_labels.ipynb : visualise manual labels on a tile and how this translates to the window cut-outs that are fed into the VAE
- predict_tiles_tryout.ipynb : test code to create dmg prediction that is used for the development of scripts/postprocessing/predict_tiles_from_cluster.py , applied to VAE model_1684233861_L2_w20_k5_f16_a20
- tmp_view_predicted_MGRS-tiles_artifacts.ipynb : visualise dmg prediction tiles and do some tests to filter artifacts out.
