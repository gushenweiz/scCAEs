import os
import pandas as pd
import numpy as np
from scipy.sparse import issparse
from scipy.io import mmread
from sklearn import preprocessing
import time

import argparse
from functions import *
import socket
theano.config.floatX= 'float32'
############################## settings ##############################
parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=42)
parser.add_argument('--dataset', default='Zeisel')
parser.add_argument('--data_type', default='csv')
parser.add_argument("--highly_genes", default=1601)
parser.add_argument('--continue_training', action='store_true', default=False)
parser.add_argument('--datasets_path', default='./datasets/')
parser.add_argument('--feature_map_sizes', default=[50, 50, 10])
parser.add_argument('--dropouts', default=[0.1, 0.1, 0.0])
parser.add_argument('--batch_size', default=100)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--num_epochs', default=800)
parser.add_argument('--reconstruct_hyperparam', default=1.)
parser.add_argument('--cluster_hyperparam', default=1.)
parser.add_argument('--regularization_hyperparam', default=0.0001)
parser.add_argument('--initialization_is_done', default=False)
parser.add_argument('--do_soft_k_means', default=True)
parser.add_argument('--architecture_visualization_flag', default=1)
parser.add_argument('--loss_acc_plt_flag', default=1)
parser.add_argument('--verbose', default=2)
args = parser.parse_args()

############################## Logging ##############################

output_path = './results/' + os.path.basename(__file__).split('.')[0] + '/' + time.strftime("%d-%m-%Y_") + \
              time.strftime("%H-%M-%S") + '_' + args.dataset +'_' + str(args.regularization_hyperparam) + '_' + '_' + socket.gethostname()
pyscript_name = os.path.basename(__file__)
create_result_dirs(output_path, pyscript_name)
sys.stdout = Logger(output_path)

print(args)
print('----------')
print(sys.argv)

# fixed random seeds
seed = args.seed
np.random.seed(args.seed)
rng = np.random.RandomState(seed)
theano_rng = MRG_RandomStreams(seed)
lasagne.random.set_rng(np.random.RandomState(seed))
learning_rate = args.learning_rate
dataset = args.dataset
data_type = args.data_type
datasets_path = args.datasets_path
dropouts = args.dropouts
feature_map_sizes = args.feature_map_sizes
num_epochs = args.num_epochs
batch_size = args.batch_size
cluster_hyperparam = args.cluster_hyperparam
reconstruct_hyperparam = args.reconstruct_hyperparam
verbose = args.verbose
regularization_hyperparam = args.regularization_hyperparam
initialization_is_done = args.initialization_is_done
do_soft_k_means = args.do_soft_k_means


############################## Load Data And Preprocessing ##############################

data, y = read_data(filename=args.dataset, data_type=data_type)
data = np.array(data).astype('float32')
y = np.array(y)
X, y = preprocessing(data, y, highly_genes=args.highly_genes)
print(X.shape)
print(y.shape)

num_clusters = len(np.unique(y))
num_samples = len(y)
dimensions = [X.shape[1], X.shape[2], X.shape[3]]
print('dataset: %s \tnum_samples: %d \tnum_clusters: %d \tdimensions: %s'
      % (dataset, num_samples, num_clusters, str(dimensions)))

feature_map_sizes[-1] = num_clusters
input_var = T.tensor4('inputs')
kernel_sizes, strides, paddings, test_batch_size = dataset_settings(dataset)
print(
    '\n... build soft_K_means model...\nfeature_map_sizes: %s \tdropouts: %s \tkernel_sizes: %s \tstrides: %s \tpaddings: %s \tseed: %s'
    % (str(feature_map_sizes), str(dropouts), str(kernel_sizes), str(strides), str(paddings), str(seed)))

##############################  Build soft_K_means Model  ##############################
encoder, decoder, loss_recons, loss_recons_clean, loss3 = build_depict(input_var, n_in=dimensions,
                                                                feature_map_sizes=feature_map_sizes,
                                                                dropouts=dropouts, kernel_sizes=kernel_sizes,
                                                                strides=strides,
                                                                paddings=paddings)

start_de = time.time()

if initialization_is_done == False:
    ############################## Pre-train soft_K_means Model: auto-encoder initialization   ##############################
    print("\n...Start AutoEncoder training...")
    initial_time = timeit.default_timer()

    train_depict_ae(dataset, X, y, input_var, decoder, encoder, loss_recons, loss_recons_clean, num_clusters, output_path,
                    batch_size=100, test_batch_size=100, num_epochs=4000, learning_rate=1e-4,
                    verbose=verbose, seed=seed, continue_training=args.continue_training)

############################## Clustering Pre-trained soft_K_means Features   ##############################
y_pred, centroids = clustering(dataset, X, y, input_var, encoder, decoder, num_clusters, output_path,
                              test_batch_size=100, seed=seed, continue_training=args.continue_training)



if do_soft_k_means == True:
    ############################## Train soft_K_means Model  ##############################
    train_soft_k_means(dataset, X, y, input_var, decoder, encoder, loss_recons,loss_recons_clean, loss3, num_clusters, output_path,
        batch_size=100, test_batch_size=100, num_epochs=100,
        learning_rate=1e-3, rec_mult=reconstruct_hyperparam, clus_mult=cluster_hyperparam,
        reg_lambda=regularization_hyperparam, centroids=centroids,  continue_training=args.continue_training, seed=seed)


