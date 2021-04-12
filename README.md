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


## Usage
### Quick start
We use the dataset 4K PBMC from a Healthy Donor (download [here](https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/pbmc4k)) to illustrate an example. You just run the following code:

```bash
python scCAEs.py --data_type 10X
```

Then you will get the cluster result of "PBMC" dataset using scCAEs method. The values of NMI, ARI and Accuracy are 0.8110, 0.8246 and 0.8763, respectively. 

### Other Datesets

If you want to run other 10X type datasets, you can put your 10X type dataset files and cell label file into "/datasets/10X". In other words, "/Datasets/10X" should contain three 10X type files: "barcodes.tsv", "genes.tsv", "matrix.mtx", and a cell label file "label.csv".

Then you can run the following code in your command lines:

```bash
python scCAEs.py --data_type 10X
```

If you want to run other types of datasets, you should first generate two files "data.csv" (gene count matrix, where rows represent cells and columns represent genes) and "label.csv" (true label). Then you can put your dataset folder containing the above two files ("data.csv" and "label.csv") into "datasets" and run the following code: 

```bash
python scCAEs.py --dataset folder_name
```

The fold_name represents the name of your dataset folder. We use the Zeisel (download [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60361)) dataset as an example. Then you can generate the gene expression matrix file "data.csv" and the cell label file "label.csv", and put them in the folder "/datasets/Zeisel". Finally you can run the following code:

```bash
python scCAEs.py --dataset Zeisel
```
Then you will get the cluster result of "Zeisel" dataset using scCAEs method. The values of NMI, ARI and Accuracy are 0.7636, 0.8107 and 0.8932, respectively. 



