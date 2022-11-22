import os
import sys
import matplotlib.pyplot as plt
import json
import numpy as np    


def main(model_dir=None):
    # get model_dir from terminal input
    
    ''' ----------
    Define model to load
    ------------'''
    
    if model_dir is None:
        raise NameError('No config file specified. Run script as "python this_script.py /path/to/model_dir/"')
    
    if os.path.isdir(model_dir):
        path_to_model = model_dir
    else:
        path_to_model = os.path.join(os.path.expanduser('~'), model_dir)
        
    
    ''' ----------
    Find and read history file(s) -- expect one file per epoch
    ------------'''    

    history_files = [file for file in os.listdir(path_to_model) if 'history' in file]
    history_files.sort() # sorts 1, 10, 11, 2 , 3 etc -- so use manual numbers rather than sorted list

    if len(history_files) > 1:
        loss = []
        val_loss = []
        for epoch_num in range(0,len(history_files)+1):
            hfile = 'history_epoch_' + str(epoch_num)
        # for hfile in history_files:
            try:
                with open(os.path.join(path_to_model,hfile)) as hf:
                    data = hf.read()
                    data = json.loads(data)
                    loss.append(data['loss']['0'])
                    val_loss.append(data['val_loss']['0'])
                    del data 
            except: # fill with NaN if file for epoch does not exist
                loss.append(np.nan)
                val_loss.append(np.nan)
                
        # print(loss)
    else:
        hfile=history_files[0]
        with open(os.path.join(path_to_model,hfile)) as hf:
            data = hf.read()
            data = json.loads(data)
            loss = np.array(list(data['loss'].values()))
            val_loss = np.array(list(data['val_loss'].values()))

    # print(data)
    print('-- Loss for {} epochs: {}'.format(len(history_files), loss) )
    
            
    ''' ----------
    Save model loss figure
    ------------'''


    # summarize history for loss
    fig, ax = plt.subplots()
    ax.plot(loss)
    ax.plot(val_loss)
    ax.set_title('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'val'], loc='upper right')

    fig.savefig(os.path.join(path_to_model  , 'model_loss' ) )
    
    

if __name__ == '__main__':
    #  Run script as "python path/to/script.py /path/to/model_dir"
        
    # retrieve config filename from command line
    model_dir = sys.argv[1] if len(sys.argv) > 1 else None

    # run script
    main(model_dir)   
    