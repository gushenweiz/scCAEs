# ScCAEs: Deep clustering single-cell RNA-seq via convolutional autoencoder embedding and soft K-means
ScCAEs first learned the nonlinear mapping from original scRNA-seq to low-dimensional features through a convolutional autoencoder, and then iteratively completed cell clustering through the regularized soft Kmeans algorithm with KL divergence. 

## Architecture
![model](https://github.com/gushenweiz/scCAEs/blob/master/Architecture/model.png)
## Install

To use scCAEs you must make sure that your python version is greater than 3.6. If you don’t know the version of python you can check it by:
```python
python
>>> import platform
>>> platform.python_version()
'3.6.13'
```

* PyPI  
Directly install the required packages from PyPI.

```bash
pip install scipy==1.4.1
pip install scikit_learn==0.22.1
pip install scanpy==1.4.6
pip install python-igraph==0.8.2
```
We recommend using Anaconda (see [Installing Anaconda](https://docs.anaconda.com/anaconda/install/)) to install Theano:
```bash
conda install theano==1.0.4
```
You need to complete Theano's GPU configuration by yourself (see [GPU supprot](https://lasagne.readthedocs.io/en/latest/user/installation.html#gpu-support)), otherwise scCAEs will not work as expected.
Finally, you should install version 0.2.dev1 of Lasagne.
```bash
git clone https://github.com/Lasagne/Lasagne.git
cd Lasagne
pip install --editable .
```

## Data availability
In the "datasets" folder, we provide the 10X data format of the PBMC dataset in the "10X" folder, so that you can quickly start scCAEs. In addition, we also provide the compressed format of all the datasets used in the paper. If you want to use them, please unzip them first. 

## Usage
### Quick start
We use the dataset 4K PBMC from a Healthy Donor to illustrate an example. You should first enter the "datasets" folder and unzip the "PBMC.zip" file. Then you just need to go back to the scCAEs file directory and run the following code:

```bash
python scCAEs.py --dataset PBMC
```

Then you will get the cluster result of "PBMC" dataset using scCAEs method. The final output reports the clustering performance, here is an example on 10X PBMC scRNA-seq data:

Final: NMI= 0.8188, ARI= 0.8299, ACC= 0.8811.

In addition, you will also get an output file named "cluster_PBMC.csv". In the file, the first column will be the cell name, the second column will be the clustering result. 

### Other Datesets

If you want to run other datasets, you should first generate two files "data.csv" (gene count matrix, where rows represent cells and columns represent genes) and "label.csv" (true label). Then you can put your dataset folder containing the above two files ("data.csv" and "label.csv") into "datasets" and run the following code: 

```bash
python scCAEs.py --dataset folder_name
```

The folder_name represents the name of your dataset folder. We use the Zeisel dataset as an example. Then you can generate the gene expression matrix file "data.csv" and the cell label file "label.csv", and put them in the folder "/datasets/Zeisel". Finally you can run the following code:

```bash
python scCAEs.py --dataset Zeisel
```
Then you will get the cluster result of "Zeisel" dataset using scCAEs method. The final output reports the clustering performance, here is an example on Zeisel dataset:

Final: NMI= 0.7483, ARI= 0.8057, ACC= 0.8895.

In addition, you will also get an output file named "cluster_Zeisel.csv". In the file, the first column will be the cell name, the second column will be the clustering result. 

## Plots
### Cluster
We show an example on how to create two tSNE plots with the clustering result and the true cell types.The R command can be found in the "analysis" folder.<br> 
In the following two images, the first image is colored with clustering results, and the second image is colored with true labels.<br>
![Zeisel_clusters](https://github.com/gushenweiz/scCAEs/blob/master/analysis/Zeisel_clusters.png)![Zeisel_truelabel](https://github.com/gushenweiz/scCAEs/blob/master/analysis/Zeisel_truelabel.png)
### The inference of cellular trajectory
We also show the results of reconstructed trajectories using Monocle3 on the Petropoulos dataset.The R command can be found in the "analysis" folder. In the following two images, the first image shows the results of Monocle3 reconstructed trajectories using raw data as input, and the second image shows the results of Monocle3 reconstructed trajectories using low-dimensional representation from scCAEs as input.<br>
![raw_trajectory](https://github.com/gushenweiz/scCAEs/blob/master/analysis/raw_trajectory.png)![scCAEs_trajectory](https://github.com/gushenweiz/scCAEs/blob/master/analysis/scCAEs_trajectory.png)<br>

