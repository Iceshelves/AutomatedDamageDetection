# Train VAE

~/outDir should exist beforehad (or create it in the feed_loop if it does not exist)
train.slurm should be run from another dir (please change accordinlgly the path inside train.slurm), because it produces the slurm log files

the embedding code still to be added (based on the output of training models, the user decides which model to use for embedding,  so it is another script) . Or you can embed based on all models, that is a lot of plots

Tensorboard can be used to monitor the training process. This can be done by connecting Tensorboard to the log directory. By default this is `./logs/fit`. You can simply run:

``` bash
tensorboard --logdir logs/fit --port=6006 
```

Then Tensorboard will be running on port 6006. To view the training process, you can run the following command on your local, and replace `USERNAME` with your username to snellius:

``` bash
ssh -NfL 16006:localhost:6006 USERNAME@snellius.surf.nl
```

In this way Tensorboard will be forwarded from the Snellius to your local. Then you can open your browser, type in localhost:16006 to view the training process.