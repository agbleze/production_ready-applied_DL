#%%
import wandb
from wandb.keras import WandbCallback
from tensorflow import keras
from tensorflow.keras import layers

#%%
run = wandb.init(project="example-DL-Book", name="run-1")
wandb.config = {"learning_rate":0.001,
                "epochs": 50,
                "batch_size": 128
                }

import os

file_path = os.getcwd()



#%%
wandb.log_artifact(file_path, name='new_artifact', type='my_dataset')

artifact = wandb.Artifact('new_artifact', type='my_dataset')
artifact.add_dir(file_path)
wandb.log_artifact(artifact)


run.use_artifact('datasiast/example-DL-Book/new_artifact:v3', type="my_dataset")
artifact_dir = artifact.download()
print(artifact_dir)

#%%


#%%
model = keras.Sequential()

logging_callback = WandbCallback(log_evaluation=True)

model.fit(x=x_train, y=y_train, batch_size=wandb.config['batch_size'],
          epochs=wandb.config['epochs'],
          verbose="auto", validation_data=(x_valid, y_valid),
          callbacks=[logging_callback]
          )





