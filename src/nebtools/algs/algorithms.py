import dataclasses as dc
from nebtools.algs.utils import EmbeddingAlgSpec, AlgGraphSupport


@dc.dataclass(frozen=True)
class NodeAttributesOnlyAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=False,
        weighted=False,
        node_attributed=False,
        edge_attributed=False,
        dynamic=False,
    )
    name: str = "node_attribute_only"
    path: str = ""
    env_name: str = ""


@dc.dataclass(frozen=True)
class NodeAttributesAndStructAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=False,
        weighted=False,
        node_attributed=False,
        edge_attributed=False,
        dynamic=False,
    )
    name: str = "node_attributes_and_structural"
    path: str = ""
    env_name: str = ""


@dc.dataclass(frozen=True)
class AppAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=False,
        edge_attributed=False,
        dynamic=False,
    )
    dimensions: int = 128  # Dimension of the concatenated proximal embeddings.
    jump_factor: float = 0.5  # Jumping factor of the APP method.
    steps: int = 3  # Maximum number of random walk steps.
    name: str = "app"
    path: str = "methods/positional/app/run.py"
    env_name: str = "neb_app_env"


@dc.dataclass(frozen=True)
class BladeAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=False,
        edge_attributed=False,
        dynamic=False,
    )
    num_epochs: int = 30  # Number of training epochs
    num_layers: int = 3  # Number of layers.
    dimensions: int = 256  # Embedding dimension.
    lr: float = 1e-4  # Learning rate
    neg_per_pos: int = 1  # Number negative samples per positive
    use_pos_edge_score: bool = True  # Use edge weights to adjust loss function
    init_method: str = "normal"  # Embedding initialization.
    use_cpu: bool = False  # Force use CPU.
    name: str = "blade"
    path: str = "methods/positional/blade/run_blade.py"
    env_name: str = "reachnes_main_env"


@dc.dataclass(frozen=True)
class DgganAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=False,
        edge_attributed=False,
        dynamic=False,
    )
    dimensions: int = 128  # Embedding dimension.
    batch_size: int = 128  # Batch size
    lmbda: float = 1e-5  # Lambda value
    lr: float = 1e-4  # Lambda value
    name: str = "dggan"
    path: str = "methods/positional/dggan/run.py"
    env_name: str = "neb_dggan_env"


@dc.dataclass(frozen=True)
class HopeAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=True,
        node_attributed=False,
        edge_attributed=False,
        dynamic=False,
    )
    dimensions: int = 128  # Num embedding dims
    beta_multiplier: float = 0.5  # Multiplied by (1/lambda_max) where lambda_max is the largest eigenvalue of the adjcency matrix.
    name: str = "hope"
    path: str = "methods/positional/hope/run.py"
    env_name: str = "neb_hope_env"


@dc.dataclass(frozen=True)
class NerdAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=False,
        edge_attributed=False,
        dynamic=False,
    )
    dimensions: int = 128  # Dimension of the concatenated proximal embeddings.
    walk_size: int = 3  # The walk size.
    samples: int = 1  # Millions of training samples.
    name: str = "nerd"
    path: str = "methods/positional/nerd/run.py"
    env_name: str = "neb_nerd_env"


@dc.dataclass(frozen=True)
class ReachnesxAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=True,
        node_attributed=True,
        edge_attributed=False,
        dynamic=False,
    )
    dimensions: int = 128  # Embedding dimension.
    adj_seq: str = (
        "OU::IU"  # Sequence of adjacency sequences for alternating neighbourhoods.
    )
    rw_distribution: str = "poisson"  # Str to specify random-walk length distribution.
    tau: str = "2.0"  # Mean of the distribution
    loc: str = "0"  # Translation of random-walk length distribution.
    order: int = 10  # Maximum walk length / (number of sum terms - 1).
    filter: str = "none"  # Which reachability filter to use.
    use_degree: bool = False  # Use the degree as initial feature.
    use_lcc: bool = False  # Use the local clustering coefficient features.
    standardize_input: bool = False  # Standardize the input
    use_float64: bool = False  # Use double precision.
    name: str = "reachnesx"
    path: str = "methods/reachnes/reachnes/run_reachnesx.py"
    env_name: str = "reachnes_main_env"


@dc.dataclass(frozen=True)
class Reachnes_rsvdAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=True,
        node_attributed=False,
        edge_attributed=False,
        dynamic=False,
    )
    dimensions: int = 128  # Embedding dimension.
    adj_seq: str = (
        "OU::IU"  # Sequence of adjacency sequences for alternating neighbourhoods.
    )
    rw_distribution: str = (
        "geometric"  # Str to specify random-walk length distribution.
    )
    tau: str = "1.0"  # Mean of the distribution
    loc: str = "1"  # Translation of random-walk length distribution.
    order: int = 10  # Maximum walk length / (number of sum terms - 1).
    num_oversampling: int = 8  # Number of oversampling dimensions for RSVD.
    filter: str = "log"  # Which reachability filter to use.
    batch_size: str = "auto"  # Batch size. Either 'auto' or an integer.
    use_float64: bool = False  # Use double precision.
    name: str = "reachnes_rsvd"
    path: str = "methods/reachnes/reachnes/run_reachnes_rsvd.py"
    env_name: str = "reachnes_main_env"


@dc.dataclass(frozen=True)
class CcassgAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=True,
        edge_attributed=False,
        dynamic=False,
    )
    num_epochs: int = 100  # Number of training epochs
    num_layers: int = 2  # Number of layers.
    dimensions: int = 512  # Embedding dimension.
    add_degree: bool = False  # Use the node degree features
    add_lcc: bool = False  # Use the local clustering coefficient features
    standardize: bool = True  # Standardize the input attributes.
    use_cpu: bool = False  # Force use CPU.
    lr: float = 1e-3  # Learning rate.
    wd: float = 0.0  # Weight decay.
    lambd: float = 1e-3  # Lambda loss parameter.
    dfr: float = 0.2  # feature drop out ratio.
    der: float = 0.2  # edge drop out ratio.
    dep1: float = 0.25  # feature drop out 1.
    encoder: str = "gcn_rossi"  # Which encoder GNN to use.
    dir_seqs: str = "U"  # Edge direction sequences for Switch model
    name: str = "ccassg"
    path: str = "methods/structural/ssgnn/run_ccassg.py"
    env_name: str = "reachnes_main_env"


@dc.dataclass(frozen=True)
class Ccassg_switchAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=True,
        edge_attributed=False,
        dynamic=False,
    )
    num_epochs: int = 100  # Number of training epochs
    num_layers: int = 2  # Number of layers.
    dimensions: int = 512  # Embedding dimension.
    add_degree: bool = False  # Use the node degree features
    add_lcc: bool = False  # Use the local clustering coefficient features
    standardize: bool = True  # Standardize the input attributes.
    use_cpu: bool = False  # Force use CPU.
    lr: float = 1e-3  # Learning rate.
    wd: float = 0.0  # Weight decay.
    lambd: float = 1e-3  # Lambda loss parameter.
    dfr: float = 0.2  # feature drop out ratio.
    der: float = 0.2  # edge drop out ratio.
    dep1: float = 0.25  # feature drop out 1.
    encoder: str = "gcn_switch"  # Which encoder GNN to use.
    dir_seqs: str = "OU::IU"  # Edge direction sequences for Switch model
    name: str = "ccassg_switch"
    path: str = "methods/structural/ssgnn/run_ccassg.py"
    env_name: str = "reachnes_main_env"


@dc.dataclass(frozen=True)
class Graphmae2Alg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=True,
        edge_attributed=False,
        dynamic=False,
    )
    num_epochs: int = 1000  # Number of training epochs
    num_layers: int = 2  # Number of layers.
    dimensions: int = 512  # Embedding dimension.
    add_degree: bool = False  # Use the node degree features
    add_lcc: bool = False  # Use the local clustering coefficient features
    standardize: bool = True  # Standardize the input attributes.
    use_cpu: bool = False  # Force use CPU.
    lr: float = 1e-3  # Learning rate.
    wd: float = 1e-4  # Weight decay.
    num_heads: int = 8  # Number of heads.
    mask_rate: float = 0.5  # Mask ratio.
    replace_rate: float = 0.05  # Replacement rate.
    alpha_l: int = 3  # Loss exponent parameter.
    lam: float = 1.0  # Loss weighting parameter.
    encoder: str = "gat_rossi"  # Which encoder to use.
    dir_seqs: str = "none"  # Which edge directions to use.
    name: str = "graphmae2"
    path: str = "methods/structural/ssgnn/run_graphmae2.py"
    env_name: str = "reachnes_main_env"


@dc.dataclass(frozen=True)
class Graphmae2_switchAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=True,
        edge_attributed=False,
        dynamic=False,
    )
    num_epochs: int = 1000  # Number of training epochs
    num_layers: int = 4  # Number of layers.
    dimensions: int = 512  # Embedding dimension.
    add_degree: bool = False  # Use the node degree features
    add_lcc: bool = False  # Use the local clustering coefficient features
    standardize: bool = True  # Standardize the input attributes.
    use_cpu: bool = False  # Force use CPU.
    lr: float = 1e-3  # Learning rate.
    wd: float = 1e-4  # Weight decay.
    num_heads: int = 8  # Number of heads.
    mask_rate: float = 0.5  # Mask ratio.
    replace_rate: float = 0.05  # Replacement rate.
    alpha_l: int = 3  # Loss exponent parameter.
    lam: float = 1.0  # Loss weighting parameter.
    encoder: str = "gat_switch"  # Which encoder to use.
    dir_seqs: str = "OI::IO"  # Which edge directions to use.
    name: str = "graphmae2_switch"
    path: str = "methods/structural/ssgnn/run_graphmae2.py"
    env_name: str = "reachnes_main_env"
