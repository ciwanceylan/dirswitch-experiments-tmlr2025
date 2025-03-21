# DirSwitch Experiments

This repo contains all the source code for the experiments in the TMLR submission "Disobeying Directions:  Switching Random Walk Filters for Unsupervised Node Embedding Learning on Directed Graphs".

Source code for the ReachNEs framework and DirSwitch is located under [reachnes-tmlr2025/](reachnes-tmlr2025/)

The purpose of this package is to benchmark node embedding models.
Researchers have released various embedding models over the  past 10-15 years.
These have been implemented using different package versions, Python versions, and even different programming languages.
To be able to benchmark them with a unified setup, this package used conda environments to isolate different node embedding models.



## Installation

Installation via conda is required as not all packages are available via pip. 
However, some packages are more easily installed via pip, so the recommendation is to use a combination.
Below are full install instructions:



#### CPU installation
```shell
conda env create --file environment_cpu_slim_env.yml
conda activate reachnes_main_env
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install pyg_lib==0.3.1 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/repo.html
pip install GitPython
pip install -e reachnes-tmlr2025/torch-sprsvd/
pip install -e reachnes-tmlr2025/
pip install -e structfeatures/
pip install -e .
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib
```

#### GPU installation
```shell
conda env create --file environment_cuda_slim_env.yml
conda activate reachnes_main_env
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib==0.3.1 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
pip install GitPython
pip install -e reachnes-tmlr2025/torch-sprsvd/
pip install -e reachnes-tmlr2025/
pip install -e structfeatures/
pip install -e .
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib
```


For the conda environment there are 4 files available:

- `environment_cuda_slim_env.yml`
- `environment_cuda_full_env.yml`
- `environment_cpu_slim_env.yml`
- `environment_cpu_full_env.yml`

Use the 'cuda' files for Nvidia GPU compatible installation. Otherwise use the 'CPU' files.

Use the file with the 'full' suffix to include jupyter and plotting libraries (not necessary for the experiments).

Setting LD_LIBRARY_PATH to point to the conda lib/ directory is usually necessary to avoid `ImportError` from the `dgl` package.


### Index datasets

The repo is constructed so that datasets can reside outside the repo if necessary.
For this reason, the dataset must be indexed before experiments can be run.
Given that the dataset are in their default locations inside [data/](data/), they can be indexed by running
```shell
python index_datasets.py
``` 

### Install conda environments for baseline methods

APP, DGGAN, HOPE and NERD have their own conda environments.
The environment specification files can be found in each respective folder under [methods/positional](methods/positional).

For APP and NERD, the code has to be compiled. 
The files named `install_cmd` in their respective directories contain the commands to compile the models.




## Datasets

The 11 smallest dataset are included in the repo under [data/inrepo/](data/inrepo/).

The 5 remaining datasets can be downloaded following the references provided in the paper.


## Reproducing experiments

Scripts for reproducing the experimental results of the paper are located in [scripts/](scripts/) directory.
The scripts default is to run just one dataset, either Cora-ML or Fly Larva. 

- [run_dispersal_experiments.sh](scripts/run_dispersal_experiments.sh) is used for the entropy measuring experiments (Figure 5 and 6). Modify the file [experiments/dispersal_analysis.py](experiments/dispersal_analysis.py) to change the datasets.
- [run_graph_alignment_experiments.sh](scripts/run_graph_alignment_experiments.sh) Runs the graph alignment experiments (Figure 7).
- [run_dirswitch_message_passing_experiments.sh](scripts/run_dirswitch_message_passing_experiments.sh) is for the results in Table 4, 5, 12 and 13.
- [run_dirswirch_proximity_experiments.sh](scripts/run_dirswirch_proximity_experiments.sh) is for the results in Tables 6 and 7, 14 and 15.
- [run_hp_tuning.sh](scripts/run_hp_tuning.sh) runs the hyperparameter tuning for the proximity embedding comparison.
- [run_proximity_node_classification_benchmark.sh](scripts/run_proximity_node_classification_benchmark.sh) runs proximity embedding comparison using the best hyperparameters (stored in [algs_best_hps.json](src/nebtools/algs/algs_best_hps.json).
- [run_ssgnn_experiment.sh](scripts/run_ssgnn_experiment.sh) Runs the comparison using self-supervised graph neural networks.
- [run_ssgnn_experiment.sh](scripts/run_pokec_investigation.sh) Runs the additional Pokec investigation (Table 16)

## Troubleshooting

The conda environment uses an upgraded version of OpenSSL.
You may therefore cncounter this error when using GitHub:
```shell
OpenSSL version mismatch. Built against 30000020, you have 30200010
```
If this happens, simply deactivate the conda environment and proceed as normal.
