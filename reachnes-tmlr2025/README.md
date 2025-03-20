# ReachNEs and DirSwitch

## Install instructions

Clone the repo.

Install the reachnes dependencies, using either pip.

#### Using pip
```bash
pip install -r requirements_<cpu/cuda>.txt
```

#### Install sp-rsvd
```bash
pip install -e ./torch-sprsvd/
```

#### Install reachnes
```bash
pip install ./reachnes
```

## Run from command line

Use the following command template to run Reachnes from the terminal:
```bash
python run.py <path_to_data.npz> <path_to_output.npy> <reduction_model> <emb_dim> <coefficients> <order> <normalization_sequence>
```

There are a number of other flags to use, use --help to see them. See the examples below.

To get structural ECF embeddings run:
```bash
python run.py datasets/telegram.npz test_embeddings.npy ecf 8 '[["geometric", {"alpha": 0.4}]]' 5 O
```

To get proximal SVD embeddings using two different coefficient sets returned as seperate embeddings run:
```bash
python run.py datasets/telegram.npz test_embeddings.npy fsvd 8 '[["geometric", {"alpha": 0.4}], ["poisson", {"tau": 2.0}]]' 5 O --as-series
```

To run using multiple GPUs in parallel with SPRSVD with oversampling 12
To get proximal SVD embeddings using two different coefficient sets returned as seperate embeddings run:
```bash
python run.py datasets/telegram.npz test_embeddings.npy rsvd 8 '[["geometric", {"alpha": 0.4}]]' 5 O --ddp --reduction_args '{"num_oversampling": 12}' 
```


## Run tests
Install additional test dependencies

#### Using pip
```bash
pip install networkx pytest
```

#### Run tests
```bash
pytest ./reachnes/tests/
```