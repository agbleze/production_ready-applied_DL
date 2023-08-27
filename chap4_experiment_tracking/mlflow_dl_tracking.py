
#%%
import os
import mlflow
from mlflow import log_metric, log_param, log_artifact

#%%
log_param("epochs", 30)
log_metric("custom", 0.6)

log_metric("custom", 0.75)

#%%

if not os.path.exists("artifacts_dir"):
    os.makedirs("artifacts_dir")

with open("artifacts_dir/test.txt", "w") as f:
    f.write("simple example")

log_artifact("artifacts_dir")

# %%
mlflow.end_run()
# %%
exp_id = mlflow.create_experiment("DLBook_1")
exp = mlflow.get_experiment(exp_id)

with mlflow.start_run(experiment_id=exp.experiment_id, run_name="run_1") as run:
    mlflow.set_tag("model_name", "model1_dev")
    log_param("epochs", 30)
    log_metric("custom", 0.6)
    
    
 #%%
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name('DLBook1').experiment_id
print(experiment)

#%%
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_USER

parent_run = client.create_run(experiment_id=experiment,
                               tags={MLFLOW_RUN_NAME: "run_2",
                                     MLFLOW_USER: "datasiast"}) 
 
client.log_param(parent_run.info.run_id, "who", "parent")
client.log_param(parent_run.info.run_id, "Description", "main run for set of smaller run")
 
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

child_run = client.create_run(experiment_id=experiment,
                              tags={MLFLOW_RUN_NAME: "run_2",
                                    MLFLOW_USER: 'datasiast',
                                    MLFLOW_PARENT_RUN_ID: parent_run.info.run_id
                                    }
                              )


client.log_param(child_run.info.run_id, "who", "child")
client.log_param(child_run.info.run_id, "run_id", "run_2_1")
client.log_param(child_run.info.run_id, "Description", "First subrun of run_2")

#%% integrate mlflow inzo keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np

df = pd.read_csv('data')
x_data = df[['x']]
y_data = df[['y']]

model = keras.Sequential([keras.Input(shape=10),
                          layers.Dense(128, activation="relu", name="layer1"),
                          layers.Dense(64, activation="relu", name="layer2"),
                          layers.Dense(1, activation="sigmoid", name="layer3")
                          ]
                        ) 
loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()
model.compile(loss=loss, optimizer=optimizer)

model.fit(x=x_data, y=y_data, batch_size=16, epochs=5, validation_split=0.2)

with mlflow.start_run() as run:
    mlflow.keras.log_model(model, "models")    