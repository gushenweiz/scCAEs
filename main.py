import sys
sys.path.insert(0, '/content/drive/My Drive/Colab/DynAE')
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from DynAE import DynAE
from datasets import load_data
import metrics

#%%

from DynAE import Selecting_highly_variable_genes
sys.path.insert(0, 'F:/ZQ/SingleCell1')
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from DynAE import DynAE
from datasets import load_data
import metrics
import os
#%%
os.environ['CUDA_VISIBLE_SEVICES']='1'

dataset ='Zeisel'
loss_weight_lambda=0.3
save_dir='D:/ZQ/SingleCell1/DynAE/results'
visualisation_dir='D:/ZQ/SingleCell1/DynAE/visualisation'
data_path ='D:/ZQ/SingleCell1/data'
batch_size=125
maxiter_pretraining=2e4
maxiter_clustering=1e4
tol=0.0001
optimizer1=SGD(0.001,0.9)
optimizer2=tf.train.AdamOptimizer(0.001)
kappa = 2
ws=0.3
hs=0.3
rot=20
scale=0.
gamma=10
highly_genes=1601
#%%
import pandas as pd
x, y = load_data(dataset, data_path)
x=Selecting_highly_variable_genes(x, highly_genes)
x=x[:,:1600]
x = np.asarray(x).astype('float64')
# Aaa
#print(x[1])
print("x的大小是"+str(x.size))
print(x.shape)
print(x.shape[-1])
n_clusters = len(np.unique(y))
print("n_clusters的大小是"+str(n_clusters))
'''model = DynAE(batch_size=batch_size, dataset=dataset, dims=[x.shape[-1], 1000, 1000, 2000, 9], loss_weight=loss_weight_lambda, gamma=gamma, n_clusters=n_clusters, visualisation_dir=visualisation_dir, ws=ws, hs=hs, rot=rot, scale=scale)'''
model = DynAE(batch_size=batch_size, dataset=dataset, dims=[x.shape[-1],9], loss_weight=loss_weight_lambda, gamma=gamma, n_clusters=n_clusters, visualisation_dir=visualisation_dir, ws=ws, hs=hs, rot=rot,scale=scale)
model.compile_dynAE(optimizer=optimizer1)
model.compile_disc(optimizer=optimizer2)
model.compile_aci_ae(optimizer=optimizer2)

#Pretraining phase

model.train_aci_ae(x, y, maxiter=maxiter_pretraining, batch_size=batch_size, validate_interval=1000, save_interval=1000,save_dir=save_dir, verbose=1, aug_train=True)

#Save the pretraining weights if you do not want to pretrain your model again


model.ae.save_weights(save_dir + '/' + dataset + '/pretrain/ae_weights.h5')
model.critic.save_weights(save_dir + '/' + dataset + '/pretrain/critic_weights.h5')

#Load the pretraining weights if you have already pretrained your network
model.ae.load_weights(save_dir + '/' + dataset + '/pretrain/ae_weights.h5')
model.critic.load_weights(save_dir + '/' + dataset + '/pretrain/critic_weights.h5')

#%%




#%%



#%%

#Clustering phase

y_pred = model.train_dynAE(x=x, y=y, kappa=kappa, n_clusters=n_clusters, maxiter=maxiter_clustering, batch_size=batch_size, tol=tol, validate_interval=140, show_interval=None, save_interval=2800, save_dir=save_dir, aug_train=True)

#%%

#Save the clustering weights

model.ae.save_weights(save_dir + '/' + dataset + '/cluster/ae_weights.h5')

#%%

#Print ACC and NMI

model.compute_acc_and_nmi(x, y)
