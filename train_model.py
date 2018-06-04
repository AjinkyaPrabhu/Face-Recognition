from AlexNet import alexNet

import numpy as np
width = 128
height = 128
file_name = "training_set.npy"

LR=0.01
EPOCHS = 10
MODEL_NAME="face-detect-{}-{}-{}".format("Ajinkya",EPOCHS,LR)
model = alexNet(width,height,LR)

data = np.load(file_name)
train = data


train_x = np.array([i[0] for i in train]).reshape(-1,width,height,1)
train_y = [i[1] for i in train]



model.fit(train_x,train_y,validation_set = 0.1,
          shuffle=True,snapshot_step=100,show_metric=True,run_id=MODEL_NAME,batch_size=100)
#tensorboard --logdir=C:\tmp\tflearn_logs\face-detect-Ajinkya-10-0.01
