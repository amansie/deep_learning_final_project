# Inferring Transcription Factor-Gene Interactions with Deep Learning

Our project applies deep learning to single-cell RNA-seq profiles to determine which transcription factors influence expression of downstream genes. We are currently applying different models to predict gene expression based on the levels of 1,421 transcription factors.

## Dataset

We used the Allen Institute cell atlas dataset, accessible from the [Gene Expression Omnibus](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115746). This dataset profiles ~45,000 genes from ~15,000 cells from mouse neocortex.

## Data Preprocessing

[This notebook](load_data/data_loader_v2.ipynb) extracts the relevant transcription factors (listed [here](load_data/input_genes.txt)) and genes to predict (listed [here](load_data/output_genes-1.txt)), normalizes gene expression counts as fractions of total cell counts (see [here](load_data/cell_expr_sums.csv) for counts), and performs an 4:1 train/test split of cells ([train](load_data/train_cells.txt), [test](load_data/test_cells.txt)). The resulting dataframes are saved as `.csv` files. The summary statistics of the output gene expressions can be generated with [this script](output_dataset_describe.py).

## Models

### Linear Regression

[This script](linear_regression_ka2461.py) fits linear regressions for each output gene in the preprocessed data.

### Fully-Connected Neural Network

[This folder](DNN) implements grid search to find an optimal dense neural network.

### Convolutional Neural Network

We have implemented two CNNs (see this [folder](cnn)). The [1-D convolutional network](cnn/1d_cnn.ipynb) performs convolutions on a 1-D vector of gene expressions, while [VGG16](cnn/vgg16.ipynb) arranges the input data as a 37x37 matrix to feed into the VGG16 model.

### Graph Convolutional Network
[This folder](GCN) implemets GCN on graph data set generated from our gene data set. We used graph convolution layers for node feature updating. We isolate the target nodes' feature vectors after graph convolution, and uesd that as input for subsequence MLP regression.
