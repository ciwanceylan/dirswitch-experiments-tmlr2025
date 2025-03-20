from typing import List, Collection, Dict, Literal
import os.path as osp
import inspect
import json
from nebtools.utils import NEB_ROOT
import nebtools.algs.algorithms as embalgs
import numpy as np


def get_algs(
    method_sets: List[str], emb_dims: List[int] = None, num_epochs: int = None
):
    methods = []
    if isinstance(method_sets, str):
        method_sets = [method_sets]
    if emb_dims is None or isinstance(emb_dims, int):
        emb_dims = [emb_dims]
    for method_set in method_sets:
        for emb_dim in emb_dims:
            methods += _get_algs(method_set, emb_dim=emb_dim, num_epochs=num_epochs)
    return sorted(list(set(methods)), key=lambda x: x.name)


def _get_algs(method_set: str, emb_dim: int = None, num_epochs: int = None):
    if not method_set:
        raise ValueError(f"No method set specified.")
    elif method_set == "reachnes_ga_comparison":
        methods = reachnes_ga_comparison()
    elif method_set == "reachnes_dirswitch_compare_message_passing":
        methods = reachnes_dirswitch_compare_message_passing()
    elif method_set == "reachnes_dirswitch_compare_rsvd":
        methods = reachnes_dirswitch_compare_rsvd()
    elif method_set == "reachnes_rsvd_nc_hp":
        methods = reachnes_rsvd_nc_hp()
    elif method_set == "nerd_hp_search":
        methods = nerd_hp_search()
    elif method_set == "app_hp_search":
        methods = app_hp_search()
    elif method_set == "blade_hp_search":
        methods = blade_hp_search()
    elif method_set == "hope_hp_search":
        methods = hope_hp_search()
    elif method_set == "dggan_hp_search":
        methods = dggan_hp_search()
    elif method_set == "graphmae2_dir_comparison_graphsage":
        methods = graphmae2_dir_comparison(
            dim=emb_dim, num_epochs=num_epochs, encoder="graphsage"
        )
    elif method_set == "ccassg_dir_comparison_graphsage":
        methods = ccassg_dir_comparison(
            dim=emb_dim, num_epochs=num_epochs, encoder="graphsage"
        )
    elif method_set == "reachnes_pokec_investigation":
        methods = reachnes_pokec_investigation()
    else:
        methods = get_alg_by_name({method_set}, emb_dim=emb_dim, num_epochs=num_epochs)
        if len(methods) == 0:
            raise NotImplementedError(f"Method set {method_set} not implemented.")

    return methods


def reachnes_ga_comparison():
    orientation_specs = {
        "undir": "U",
        "default": "O",
        "multidir1": "O::I",
        "dirswitch1": "OU::IU",
        "multidir2": "O::I::OI::IO",
        "dirswitch2": "OOU::IIU::OIU::IOU",
        "multidir3": "OOO::OOI::OIO::IOO::III::IIO::IOI::OII",
        "dirswitch3": "OOOU::OOIU::OIOU::IOOU::IIIU::IIOU::IOIU::OIIU",
    }
    methods = []
    tau_vals = np.concatenate(
        (np.arange(0.0, 4.0, 0.2), np.power(2, np.arange(2, 4.5, 0.25)))
    )
    order_ = 30
    dims = [128]
    for dim in dims:
        for _, adj_seq in orientation_specs.items():
            for tau in tau_vals:
                methods.append(
                    embalgs.ReachnesxAlg(
                        dimensions=dim,
                        adj_seq=adj_seq,
                        rw_distribution="geometric",
                        tau=f"{tau}",
                        loc="0",
                        standardize_input=False,
                        use_degree=True,
                        use_lcc=True,
                        order=order_,
                    )
                )

                methods.append(
                    embalgs.ReachnesxAlg(
                        dimensions=dim,
                        adj_seq=adj_seq,
                        rw_distribution="poisson",
                        tau=f"{tau}",
                        loc="0",
                        standardize_input=False,
                        use_degree=True,
                        use_lcc=True,
                        order=order_,
                    )
                )

                methods.append(
                    embalgs.ReachnesxAlg(
                        dimensions=dim,
                        adj_seq=adj_seq,
                        rw_distribution="binom",
                        tau=f"{tau}",
                        loc="0",
                        standardize_input=False,
                        use_degree=True,
                        use_lcc=True,
                        order=order_,
                    )
                )

                methods.append(
                    embalgs.ReachnesxAlg(
                        dimensions=dim,
                        adj_seq=adj_seq,
                        rw_distribution="uniform",
                        tau=f"{tau}",
                        loc="0",
                        standardize_input=False,
                        use_degree=True,
                        use_lcc=True,
                        order=order_,
                    )
                )

    return methods


def reachnes_dirswitch_compare_message_passing():
    orientation_specs = {
        "undir": "U",
        "default": "O",
        "multidir1": "O::I",
        "dirswitch1": "OU::IU",
        "multidir2": "O::I::OI::IO",
        "dirswitch2": "OOU::IIU::OIU::IOU",
        "multidir3": "OOO::OOI::OIO::IOO::III::IIO::IOI::OII",
        "dirswitch3": "OOOU::OOIU::OIOU::IOOU::IIIU::IIOU::IOIU::OIIU",
    }
    methods = []
    dims = [512, 1024]
    for dim in dims:
        for _, adj_seq in orientation_specs.items():
            methods.append(
                embalgs.ReachnesxAlg(
                    dimensions=dim,
                    adj_seq=adj_seq,
                    rw_distribution="geometric",
                    tau=f"1.0",
                    loc="0",
                    standardize_input=True,
                    order=12,
                )
            )

            methods.append(
                embalgs.ReachnesxAlg(
                    dimensions=dim,
                    adj_seq=adj_seq,
                    rw_distribution="poisson",
                    tau=f"2.0",
                    loc="0",
                    standardize_input=True,
                    order=12,
                )
            )

            methods.append(
                embalgs.ReachnesxAlg(
                    dimensions=dim,
                    adj_seq=adj_seq,
                    rw_distribution="geometric::uniform",
                    tau=f"1.0::2.0",
                    loc="0::1",
                    standardize_input=True,
                    order=12,
                )
            )

            methods.append(
                embalgs.ReachnesxAlg(
                    dimensions=dim,
                    adj_seq=adj_seq,
                    rw_distribution="binom::binom::binom",
                    tau=f"1.0::2.0::3.0",
                    loc="0::2::5",
                    standardize_input=True,
                    order=12,
                )
            )

            methods.append(
                embalgs.ReachnesxAlg(
                    dimensions=dim,
                    adj_seq=adj_seq,
                    rw_distribution="geometric::geometric::geometric::geometric",
                    tau=f"1.0::2.0::3.0::4.0",
                    loc="0::1::2::3",
                    standardize_input=True,
                    order=12,
                )
            )
    return methods


def reachnes_dirswitch_compare_rsvd():
    orientation_specs = {
        "undir": "U",
        "default": "O",
        "multidir1": "O::I",
        "dirswitch1": "OU::IU",
        "multidir2": "O::I::OI::IO",
        "dirswitch2": "OOU::IIU::OIU::IOU",
        "multidir3": "OOO::OOI::OIO::IOO::III::IIO::IOI::OII",
        "dirswitch3": "OOOU::OOIU::OIOU::IOOU::IIIU::IIOU::IOIU::OIIU",
    }
    dims = [512, 1024]
    methods = []
    for _, adj_seq in orientation_specs:
        for filter_ in ["log"]:
            for dim in dims:
                methods.append(
                    embalgs.Reachnes_rsvdAlg(
                        dimensions=dim,
                        adj_seq=adj_seq,
                        rw_distribution="geometric",
                        tau=f"1.0",
                        loc="1",
                        filter=filter_,
                        order=12,
                    )
                )
                methods.append(
                    embalgs.Reachnes_rsvdAlg(
                        dimensions=dim,
                        adj_seq=adj_seq,
                        rw_distribution="poisson",
                        tau=f"2.0",
                        loc="1",
                        filter=filter_,
                        order=12,
                    )
                )
                methods.append(
                    embalgs.Reachnes_rsvdAlg(
                        dimensions=dim,
                        adj_seq=adj_seq,
                        rw_distribution="geometric::uniform",
                        tau=f"1.0::2.0",
                        loc="1::2",
                        filter=filter_,
                        order=12,
                    )
                )
                methods.append(
                    embalgs.Reachnes_rsvdAlg(
                        dimensions=dim,
                        adj_seq=adj_seq,
                        rw_distribution="binom::binom::binom",
                        tau=f"1.0::2.0::3.0",
                        loc="1::3::6",
                        filter=filter_,
                        order=12,
                    )
                )
                methods.append(
                    embalgs.Reachnes_rsvdAlg(
                        dimensions=dim,
                        adj_seq=adj_seq,
                        rw_distribution="geometric::geometric::geometric::geometric",
                        tau=f"1.0::2.0::3.0::4.0",
                        loc="1::2::3::4",
                        filter=filter_,
                        order=12,
                    )
                )

    return methods


def reachnes_rsvd_nc_hp():
    orientation_specs = [
        "OU::IU",
        "OOU::IIU::OIU::IOU",
        "OOOU::OOIU::OIOU::IOOU::IIIU::IIOU::IOIU::OIIU",
    ]
    dims = [64, 128, 256, 512, 1024]
    methods = []
    for dim in dims:
        for adj_seq in orientation_specs:
            methods.append(
                embalgs.Reachnes_rsvdAlg(
                    dimensions=dim,
                    adj_seq=adj_seq,
                    rw_distribution="geometric",
                    tau=f"1.0",
                    loc="1",
                    order=10,
                )
            )
            methods.append(
                embalgs.Reachnes_rsvdAlg(
                    dimensions=dim,
                    adj_seq=adj_seq,
                    rw_distribution="poisson",
                    tau=f"2.0",
                    loc="1",
                    order=10,
                )
            )
            methods.append(
                embalgs.Reachnes_rsvdAlg(
                    dimensions=dim,
                    adj_seq=adj_seq,
                    rw_distribution="geometric::uniform",
                    tau=f"1.0::2.0",
                    loc="1::2",
                    order=10,
                )
            )
            methods.append(
                embalgs.Reachnes_rsvdAlg(
                    dimensions=dim,
                    adj_seq=adj_seq,
                    rw_distribution="binom::binom::binom",
                    tau=f"1.0::2.0::3.0",
                    loc="1::3::6",
                    order=10,
                )
            )
            methods.append(
                embalgs.Reachnes_rsvdAlg(
                    dimensions=dim,
                    adj_seq=adj_seq,
                    rw_distribution="geometric::geometric::geometric::geometric",
                    tau=f"1.0::2.0::3.0::4.0",
                    loc="1::2::3::4",
                    order=10,
                )
            )

    return methods


def nerd_hp_search():
    methods = []
    for dim in [64, 128, 256, 512, 1024]:
        for walk_size in [2, 3, 4, 5, 6]:
            for samples in [1, 2, 3]:
                methods.append(
                    embalgs.NerdAlg(
                        dimensions=dim, walk_size=walk_size, samples=samples
                    )
                )
    return methods


def app_hp_search():
    methods = []
    for dim in [64, 128, 256, 512, 1024]:
        for steps in [2, 3, 4, 5, 6, 7]:
            for jump_factor in [0.1, 0.25, 0.5, 0.75]:
                methods.append(
                    embalgs.AppAlg(dimensions=dim, steps=steps, jump_factor=jump_factor)
                )
    return methods


def blade_hp_search():
    methods = []
    for dim in [64, 128, 256, 512, 1024]:
        for layers in [2, 3, 4, 5, 6]:
            for lr in [1e-4, 1e-3, 1e-2]:
                for num_epochs in [10, 30, 50, 100]:
                    methods.append(
                        embalgs.BladeAlg(
                            dimensions=dim,
                            num_layers=layers,
                            lr=lr,
                            num_epochs=num_epochs,
                        )
                    )
    return methods


def hope_hp_search():
    methods = []
    for dim in [32, 64, 128, 256, 512]:
        for beta in [1e-3, 1e-2, 1e-1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0]:
            methods.append(embalgs.HopeAlg(dimensions=dim, beta_multiplier=beta))
    return methods


def dggan_hp_search():
    methods = []
    for dims in [128, 256, 512, 1024]:
        for lmbda in [5e-6, 1e-5, 5e-5]:
            for lr in [5e-5, 1e-4, 5e-4]:
                methods.append(embalgs.DgganAlg(dimensions=dims, lr=lr, lmbda=lmbda))
    return methods


def graphmae2_dir_comparison(dim: int, num_epochs: int, encoder: str):
    methods = []
    methods.append(
        embalgs.Graphmae2Alg(
            dimensions=dim,
            num_epochs=num_epochs,
            num_layers=4,
            lam=1.0,
            dir_seqs="none",
            encoder=f"{encoder}_rossi",
            add_degree=False,
            add_lcc=False,
            name="graphmae2gs_rossi",
        )
    )
    methods.append(
        embalgs.Graphmae2Alg(
            dimensions=dim,
            num_epochs=num_epochs,
            lam=1.0,
            num_layers=4,
            encoder=f"{encoder}_switch",
            dir_seqs="U",
            add_degree=False,
            add_lcc=False,
            name="graphmae2gs_undir",
        )
    )
    methods.append(
        embalgs.Graphmae2Alg(
            dimensions=dim,
            num_epochs=num_epochs,
            lam=1.0,
            num_layers=4,
            encoder=f"{encoder}_switch",
            dir_seqs="OOOU::OOIU::OIOU::IOOU::IIIU::IIOU::IOIU::OIIU",
            add_degree=False,
            add_lcc=False,
            name="graphmae2gs_switch3",
        )
    )
    methods.append(
        embalgs.Graphmae2Alg(
            dimensions=dim,
            num_epochs=num_epochs,
            lam=1.0,
            num_layers=4,
            encoder=f"{encoder}_switch",
            dir_seqs="OU::IU",
            add_degree=False,
            add_lcc=False,
            name="graphmae2gs_switch1",
        )
    )
    methods.append(
        embalgs.Graphmae2Alg(
            dimensions=dim,
            num_epochs=num_epochs,
            lam=1.0,
            num_layers=4,
            encoder=f"{encoder}_switch",
            dir_seqs="O",
            add_degree=False,
            add_lcc=False,
            name="graphmae2gs_default",
        )
    )
    return methods


def ccassg_dir_comparison(dim: int, num_epochs: int, encoder: str):
    methods = []
    num_layers = 4
    methods.append(
        embalgs.CcassgAlg(
            dimensions=dim,
            num_epochs=num_epochs,
            num_layers=num_layers,
            encoder=f"{encoder}_rossi",
            dir_seqs="none",
            add_degree=False,
            add_lcc=False,
            name="ccassg_gs_rossi",
        )
    )
    methods.append(
        embalgs.CcassgAlg(
            dimensions=dim,
            num_epochs=num_epochs,
            num_layers=num_layers,
            encoder=f"{encoder}_switch",
            dir_seqs="U",
            add_degree=False,
            add_lcc=False,
            name="ccassg_undir",
        )
    )
    methods.append(
        embalgs.CcassgAlg(
            dimensions=dim,
            num_epochs=num_epochs,
            num_layers=num_layers,
            encoder=f"{encoder}_switch",
            dir_seqs="OOOU::OOIU::OIOU::IOOU::IIIU::IIOU::IOIU::OIIU",
            add_degree=False,
            add_lcc=False,
            name="ccassg_switch3",
        )
    )
    methods.append(
        embalgs.CcassgAlg(
            dimensions=dim,
            num_epochs=num_epochs,
            num_layers=num_layers,
            encoder=f"{encoder}_switch",
            dir_seqs="OU::IU",
            add_degree=False,
            add_lcc=False,
            name="ccassg_switch1",
        )
    )
    methods.append(
        embalgs.CcassgAlg(
            dimensions=dim,
            num_epochs=num_epochs,
            num_layers=num_layers,
            encoder=f"{encoder}_switch",
            dir_seqs="O",
            add_degree=False,
            add_lcc=False,
            name="ccassg_default",
        )
    )
    return methods


def reachnes_pokec_investigation():
    orientation_specs = [
        "U",
        "O",
        "O::I",
        "OU::IU",
        "OOU::IIU",
        "O::I::OI::IO",
        "OOU::IIU::OIU::IOU",
        "OOOU::IIIU",
        "OOOU::OOIU::OIOU::IOOU::IIIU::IIOU::IOIU::OIIU",
    ]
    methods = []
    dims = [512, 1024]
    for dim in dims:
        for adj_seq in orientation_specs:
            methods.append(
                embalgs.ReachnesxAlg(
                    dimensions=dim,
                    adj_seq=adj_seq,
                    rw_distribution="geometric",
                    tau=f"0.0",
                    loc="0",
                    standardize_input=True,
                    order=4,
                )
            )
            methods.append(
                embalgs.ReachnesxAlg(
                    dimensions=dim,
                    adj_seq=adj_seq,
                    rw_distribution="geometric::geometric",
                    tau=f"0.0::0.0",
                    loc="0::2",
                    standardize_input=True,
                    order=4,
                    name="zero_and_two",
                )
            )
            methods.append(
                embalgs.ReachnesxAlg(
                    dimensions=dim,
                    adj_seq=adj_seq,
                    rw_distribution="geometric::geometric",
                    tau=f"0.0::0.0",
                    loc="0::1",
                    standardize_input=True,
                    order=4,
                    name="zero_and_one",
                )
            )
            methods.append(
                embalgs.ReachnesxAlg(
                    dimensions=dim,
                    adj_seq=adj_seq,
                    rw_distribution="geometric::geometric",
                    tau=f"0.0::0.0",
                    loc="0::3",
                    standardize_input=True,
                    order=4,
                    name="zero_and_three",
                )
            )
    return methods


def get_alg_by_name(
    names: Collection[str],
    emb_dim: int = None,
    num_epochs: int = None,
    alg_kwargs: Dict = None,
):
    if alg_kwargs is None:
        alg_kwargs = dict()
    all_algs = inspect.getmembers(embalgs, inspect.isclass)
    out_algs = []
    for cls_name, alg in all_algs:
        if is_alg_valid(alg) and alg.name in names:
            alg = instantiate_alg(
                alg, dimensions=emb_dim, num_epochs=num_epochs, **alg_kwargs
            )
            out_algs.append(alg)
    return out_algs


def is_alg_valid(alg):
    if alg.__name__ in ("EmbeddingAlg", "EmbeddingAlgSpec", "AlgGraphSupport"):
        return False
    if hasattr(alg, "disabled") and alg.disabled:
        return False
    return True


def instantiate_alg(alg, dimensions: int = None, num_epochs: int = None, **kwargs):
    if hasattr(alg, "dimensions"):
        dimensions = alg.dimensions if dimensions is None else dimensions
        kwargs["dimensions"] = dimensions

    if hasattr(alg, "num_epochs"):
        num_epochs = alg.num_epochs if num_epochs is None else num_epochs
        kwargs["num_epochs"] = num_epochs

    alg = alg(**kwargs)
    return alg


def get_best_hp_alg(dataset, alg_names, mode: Literal["nc", "lp"]):
    if mode == "nc":
        with open(
            osp.join(NEB_ROOT, "src", "nebtools", "algs", "algs_best_hps.json"), "r"
        ) as fp:
            all_hps = json.load(fp)
    else:
        raise ValueError
    all_algs = {
        alg.name: alg
        for cls_name, alg in inspect.getmembers(embalgs, inspect.isclass)
        if is_alg_valid(alg)
    }

    methods = []
    for alg_name in alg_names:
        try:
            best_hps = all_hps[dataset][alg_name]
            if best_hps["accuracy"] is None:
                raise KeyError
            del best_hps["accuracy"]
            if "undirected" in best_hps:
                del best_hps["undirected"]
        except KeyError:
            print(
                f"Alg {alg_name} does not have any best hyperparameter for dataset {dataset}"
            )
            continue
        methods.append(all_algs[alg_name](**best_hps))
    return methods
