from typing import Dict
import dataclasses as dc
import os
from glob import glob
import json
import yaml
import shutil
from tqdm import tqdm
import requests
import tarfile
import numpy as np
import pandas as pd
import nebtools.data.graph as dgraph
import torch_geometric as pyg
from nebtools.utils import NEB_DATAROOT

DEFAULT_COMMENT_CHAR = "%"


class MetadataError(ValueError):
    pass


@dc.dataclass(frozen=True)
class DatasetMetadata:
    name: str
    graphpath: str
    datapath: str
    file_info: Dict
    graph_info: Dict


def get_data_folder(data_root: str, metadata: Dict, metadata_path: str):
    # if 'abspath' in metadata and metadata['abspath']:
    #     path = metadata['abspath']
    # elif 'relpath' in metadata and metadata['relpath']:
    #     path = os.path.abspath(os.path.join(data_root, metadata['relpath']))
    # else:
    relpath = os.path.dirname(os.path.abspath(metadata_path))[len(NEB_DATAROOT) + 1 :]
    path = os.path.abspath(os.path.join(data_root, relpath))
    # print(f"Missing relpath '{metadata_path}'")
    return path


def graph_file_exists(data_root: str, metadata: Dict, metadata_path: str):
    datapath = get_data_folder(data_root, metadata, metadata_path)
    graphpath = os.path.join(datapath, metadata["graph_file"])
    return os.path.exists(graphpath)


def verify_graph(data_root: str, metadata: Dict, metadata_path: str):
    datapath = get_data_folder(data_root, metadata, metadata_path)
    graphpath = os.path.join(datapath, metadata["graph_file"])
    if not os.path.exists(graphpath):
        raise FileNotFoundError(
            f"Could not find graph data {graphpath} for {metadata_path}"
        )


def verify_filetype_info(metadata: Dict, metadata_path: str):
    if "filetype" not in metadata["file_info"]:
        raise MetadataError(f"Filetype not avaiable in {metadata_path}")
    # if metadata['filetype_info']['filetype'] in {"csv", "tsv", "edgelist"}:
    #     if 'comment_char' not in metadata['filetype_info']:
    #         metadata['filetype_info']["comment_char"] = DEFAULT_COMMENT_CHAR
    #         print(f"No comment char available for {metadata_path}, setting to default ({DEFAULT_COMMENT_CHAR})")
    # elif metadata['filetype_info']['filetype']  == 'npz':


def verify_graph_info(metadata: Dict, metadata_path: str):
    if "num_nodes" not in metadata["graph_info"]:
        raise MetadataError(f"'num_nodes' not in {metadata_path}")
    if "weights" not in metadata["graph_info"]:
        raise MetadataError(f"'weights' info not in {metadata_path}")
    if "directed" not in metadata["graph_info"]:
        raise MetadataError(f"'directed' info not in {metadata_path}")


def verify_metadata(data_root: str, metadata: Dict, methadata_path: str):
    if "name" not in metadata:
        raise MetadataError(f"'name' not in {methadata_path}")
    verify_graph(data_root, metadata, methadata_path)
    verify_filetype_info(metadata, methadata_path)
    verify_graph_info(metadata, methadata_path)


def get_ac_graph_name(directed: bool, num_connected_edges: int, seed: int):
    dirstr = "directed" if directed else "undirected"
    return f"ac::{dirstr}::{num_connected_edges}::{seed}"


def get_lrs_graph_name(degree: int, num_layers: int):
    return f"lrs::{degree}::{num_layers}"


def get_lrs_triplet_graph_name(triplet_name: str, degree: int, num_layers: int):
    return f"lrs::{triplet_name}::{degree}::{num_layers}"


def download_synthetic(data_root: str):
    with open(os.path.join(NEB_DATAROOT, "incloud/download_links.yml"), "r") as fp:
        download_links = yaml.safe_load(fp)
    datafolder = os.path.abspath(os.path.join(data_root, "synthetic"))
    if os.path.exists(datafolder):
        print("Synthetic datafolder already exists")
        return
    print("Downloading synthetic data")
    try:
        url = download_links["synthetic"]
    except KeyError:
        print("No download link for synthetic data. Skipping")
        return
    targz_path = os.path.join(data_root, "synthetic_data.tar.gz")

    r = requests.get(url, stream=True)
    total_length = r.headers.get("content-length")
    total_length = (
        int(total_length) // 4096 if total_length is not None else total_length
    )
    with open(targz_path, "wb") as f:
        if total_length is None:
            f.write(r.content)
        else:
            for chunk in tqdm(r.iter_content(chunk_size=4096), total=total_length):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush() commented by recommendation from J.F.Sebastian
    # wget.download(url, out=targz_path)
    # response = requests.get(url, stream=True)
    #
    #
    # with open(targz_path, "wb") as handle:
    #     for data in tqdm(response.iter_content()):
    #         handle.write(data)

    try:
        tar = tarfile.open(targz_path, "r:gz")
        tar.extractall(path=datafolder)
        tar.close()
    except tarfile.ReadError:
        print(f"Could not read '{targz_path}', probably bad download url. Skipping.")


def index_for_synthetic_data(data_root: str, download: bool = False):
    synthetic_index = dict()
    if download:
        download_synthetic(data_root)

    if os.path.exists(os.path.join(data_root, "synthetic")):
        print(f"Indexing synthetic data")
        synthetic_index.update(index_lrs_graphs())
        synthetic_index.update(index_lrs_triplett_graphs())
        synthetic_index.update(index_automorphic_classification())
    return synthetic_index


def index_lrs_graphs():
    data_index = {}
    for degree in [5]:
        for num_layers in range(1, 7):
            dirstr = f"lrs_graph_{degree}_{num_layers}"
            name = get_lrs_graph_name(degree, num_layers)
            datapath = os.path.join("data", "synthetic", "lrs_graphs", dirstr)
            graphpath = os.path.join(datapath, "composed_lrs_graph.edgelist")
            with open(graphpath, "r") as fp:
                num_nodes = int(fp.readline()[1:].strip())
            file_info = {"filetype": "edgelist", "comment_char": "%"}
            graph_info = {"num_nodes": num_nodes, "weights": False, "directed": False}
            data_index[name] = dict(
                name=name,
                graphpath=graphpath,
                datapath=datapath,
                node_train_labels_file="train_labels.json",
                node_test_labels_file="test_labels.json",
                file_info=file_info,
                graph_info=graph_info,
            )

    return data_index


def index_lrs_triplett_graphs():
    data_index = {}
    for degree in [5]:
        for num_layers in range(3, 7):
            for triplet_names in ["StCiCl", "DiTrSt"]:
                dirstr = f"lrs_{triplet_names}_graph_{degree}_{num_layers}"
                name = get_lrs_triplet_graph_name(triplet_names, degree, num_layers)
                datapath = os.path.join("data", "synthetic", "lrs_graphs", dirstr)
                graphpath = os.path.join(datapath, "composed_lrs_graph.edgelist")
                with open(graphpath, "r") as fp:
                    num_nodes = int(fp.readline()[1:].strip())
                file_info = {"filetype": "edgelist", "comment_char": "%"}
                graph_info = {
                    "num_nodes": num_nodes,
                    "weights": False,
                    "directed": False,
                }
                data_index[name] = dict(
                    name=name,
                    graphpath=graphpath,
                    datapath=datapath,
                    node_train_labels_file="train_labels.json",
                    node_test_labels_file="test_labels.json",
                    file_info=file_info,
                    graph_info=graph_info,
                )

    return data_index


def index_automorphic_classification():
    data_index = {}
    for dirstr, is_directed in zip(["directed", "undirected"], [True, False]):
        for nce in range(6):
            for seed in range(5):
                name = get_ac_graph_name(is_directed, nce, seed)
                datapath = os.path.join(
                    "data",
                    "synthetic",
                    "automorphic_classification",
                    dirstr,
                    f"cycle_main_10_{nce}-{seed}",
                )
                graphpath = os.path.join(datapath, "composed_graph.edgelist")
                with open(graphpath, "r") as fp:
                    num_nodes = int(fp.readline()[1:].strip())
                file_info = {"filetype": "edgelist", "comment_char": "%"}
                graph_info = {
                    "num_nodes": num_nodes,
                    "weights": False,
                    "directed": is_directed,
                }
                data_index[name] = dict(
                    name=name,
                    graphpath=graphpath,
                    datapath=datapath,
                    node_train_labels_file="train_labels.json",
                    node_test_labels_file="test_labels.json",
                    file_info=file_info,
                    graph_info=graph_info,
                )
                # data_index[name] = dc.asdict(DatasetMetadata(name=name, graphpath=graphpath, datapath=datapath,
                #                                              file_info=file_info, graph_info=graph_info))

    return data_index


def download_data_if_required(data_root, download, metadata, metadata_path):
    graph_exists_dst = graph_file_exists(data_root, metadata, metadata_path)
    graph_exists_neb_root = graph_file_exists(NEB_DATAROOT, metadata, metadata_path)
    if graph_exists_dst:
        return
    elif graph_exists_neb_root:
        copy_inrepo_data(data_root, metadata, metadata_path)
    elif metadata["name"] in {
        "pyg_cora",
        "pyg_cora_ml",
        "pyg_citeseer",
        "pyg_dblp",
        "pyg_pubmed",
    }:
        download_pyg_citation_full_datasets(data_root, metadata, metadata_path)
    elif metadata["name"] in {"pyg_email_eu_core"}:
        download_pyg_email_eu_core(data_root, metadata, metadata_path)
    elif download and "download" in metadata and metadata["download"]:
        download_data(data_root, metadata, metadata_path)
    else:
        print(f"No existing graph or download possibility for '{metadata['name']}'")

    # try:
    #     verify_graph(data_root, metadata, metadata_path)
    #     return
    # except FileNotFoundError as e:
    #     copy_inrepo_data(data_root, metadata, metadata_path)
    #
    # if download and "download" in metadata and metadata["download"]:
    #     try:
    #         verify_graph(data_root, metadata, metadata_path)
    #     except FileNotFoundError as e:
    #         download_data(data_root, metadata, metadata_path)


def copy_inrepo_data(data_root, metadata, metadata_path):
    datafolder = get_data_folder(data_root, metadata, metadata_path)
    try:
        shutil.copytree(os.path.dirname(metadata_path), datafolder, dirs_exist_ok=False)
    except FileExistsError as e:
        shutil.rmtree(datafolder)
        shutil.copytree(os.path.dirname(metadata_path), datafolder, dirs_exist_ok=False)


def download_data(data_root, metadata, metadata_path):
    with open(os.path.join(NEB_DATAROOT, "incloud/download_links.yml"), "r") as fp:
        download_links = yaml.safe_load(fp)

    datafolder = get_data_folder(data_root, metadata, metadata_path)
    try:
        url = download_links[metadata["name"]]
    except KeyError:
        print(f"No available downloadlink for '{metadata['name']}'. Skipping.")
        return
    os.makedirs(datafolder, exist_ok=True)
    targz_path = os.path.join(datafolder, "data.tar.gz")
    # datafolder = get_data_folder(data_root, metadata, metadata_path)

    r = requests.get(url, stream=True)
    total_length = r.headers.get("content-length")
    total_length = (
        int(total_length) // 4096 if total_length is not None else total_length
    )
    with open(targz_path, "wb") as f:
        if total_length is None:
            f.write(r.content)
        else:
            for chunk in tqdm(r.iter_content(chunk_size=4096), total=total_length):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush() commented by recommendation from J.F.Sebastian
    # wget.download(url, out=targz_path)
    # response = requests.get(url, stream=True)
    #
    #
    # with open(targz_path, "wb") as handle:
    #     for data in tqdm(response.iter_content()):
    #         handle.write(data)

    try:
        tar = tarfile.open(targz_path, "r:gz")
        tar.extractall(path=datafolder)
        tar.close()
    except tarfile.ReadError:
        print(f"Could not read '{targz_path}', probably bad download url. Skipping.")

    os.remove(targz_path)


def pyg_dataset_to_graph(dataset):
    has_node_attr = hasattr(dataset, "x")
    num_nodes = dataset.x.shape[0] if has_node_attr else dataset.num_nodes
    graph = dgraph.SimpleGraph(
        num_nodes=num_nodes,
        edges=dataset.edge_index.T.numpy().astype(np.int64),
        node_attributes=dataset.x.numpy() if has_node_attr else None,
        directed=dataset[0].is_directed(),
    )
    return graph


def download_pyg_citation_full_datasets(data_root, metadata, metadata_path):
    name_map = {
        "pyg_cora": "Cora",
        "pyg_cora_ml": "Cora_ML",
        "pyg_citeseer": "CiteSeer",
        "pyg_dblp": "DBLP",
        "pyg_pubmed": "PubMed",
    }
    dataset = pyg.datasets.CitationFull(
        root="/tmp/tmp_pyg_data", name=name_map[metadata["name"]], to_undirected=False
    )
    graph = pyg_dataset_to_graph(dataset)
    datapath = get_data_folder(
        data_root=data_root, metadata=metadata, metadata_path=metadata_path
    )
    os.makedirs(datapath, exist_ok=True)
    graph.save_npz(os.path.join(datapath, metadata["graph_file"]))
    pd.Series(dataset.y).to_json(os.path.join(datapath, metadata["node_labels_file"]))


def download_pyg_email_eu_core(data_root, metadata, metadata_path):
    dataset = pyg.datasets.EmailEUCore(root="/tmp/tmp_pyg_data")
    graph = pyg_dataset_to_graph(dataset)
    datapath = get_data_folder(
        data_root=data_root, metadata=metadata, metadata_path=metadata_path
    )
    os.makedirs(datapath, exist_ok=True)
    graph.save_npz(os.path.join(datapath, metadata["graph_file"]))
    pd.Series(dataset.y).to_json(os.path.join(datapath, metadata["node_labels_file"]))


def index_datasets(
    data_root: str, download: bool = False, index_synthetic: bool = False
):
    if not data_root:
        data_root = NEB_DATAROOT
        print(f"No data root provided. Using '{data_root}'.")
    os.makedirs(data_root, exist_ok=True)
    available_metadata_paths = glob(
        os.path.join(NEB_DATAROOT, "**/metadata.yml"), recursive=True
    )
    print(f"{len(available_metadata_paths)} datasets with metadata located")
    data_index = {}

    for metadata_path in available_metadata_paths:
        try:
            with open(metadata_path, "r") as fp:
                metadata = yaml.safe_load(fp)
            download_data_if_required(data_root, download, metadata, metadata_path)
            # print(metadata)
            verify_metadata(data_root, metadata, metadata_path)
            if metadata["name"] in data_index:
                raise MetadataError(
                    f"Name in {metadata_path} already used by "
                    f"{data_index[metadata['name']]['datapath']} dataset."
                )
        except FileNotFoundError as e:
            # print(e)
            continue
        except MetadataError as e:
            print(e)
            continue

        datapath = get_data_folder(data_root, metadata, metadata_path)
        graphpath = os.path.join(datapath, metadata["graph_file"])
        metadata["datapath"] = datapath
        metadata["graphpath"] = graphpath
        data_index[metadata["name"]] = metadata
    if index_synthetic:
        data_index.update(
            index_for_synthetic_data(data_root=data_root, download=download)
        )

    with open(os.path.join(data_root, "data_index.json"), "w") as fp:
        json.dump(data_index, fp, indent=2)
