import os
import dataclasses as dc
import pickle
import numpy as np
import tqdm
import nebtools.experiments.switch_analysis as switch_analysis
import reachnes.coeffs as rcoeffs
from nebtools.utils import NEB_ROOT


def compute_stat_distances(dataset, batch_size: int, num_samples: int):
    adj_seqs_dict = {
        "basic": "U::OU::IU::O::I",
        "level2": "OO::OI::IO::II::OOU::OIU::IOU::IIU",
        "switch3": "U::OOOU::OOIU::OIOU::OIIU::IIIU::IIOU::IOIU::IOOU",
    }
    distr_dict = {
        "geom1": rcoeffs.CoeffsSpec(name="geometric", kwargs={"tau": 1.0}, loc=0),
        "poisson2": rcoeffs.CoeffsSpec(name="poisson", kwargs={"tau": 2.0}, loc=0),
        "uniform3": rcoeffs.CoeffsSpec(name="uniform", kwargs={"tau": 3.0}, loc=0),
        "poisson6": rcoeffs.CoeffsSpec(name="poisson", kwargs={"tau": 6.0}, loc=0),
    }
    data_graph = switch_analysis.load_graph(dataset)
    for distr_name, coeffs_spec in tqdm.tqdm(distr_dict.items()):
        for adj_seq_name, adj_seqs in tqdm.tqdm(adj_seqs_dict.items()):
            save_folder = os.path.join(
                NEB_ROOT,
                "neb_gcp_results",
                "stat_distance_analysis",
                dataset,
                distr_name,
                adj_seq_name,
            )

            os.makedirs(save_folder, exist_ok=True)
            metadata = {
                "coeffs": (distr_name, dc.asdict(coeffs_spec)),
                "adj_seq": (adj_seq_name, adj_seqs),
            }
            with open(os.path.join(save_folder, "metadata.pkl"), "wb") as fp:
                pickle.dump(obj=metadata, file=fp)

            hellinger_distances, tv_distances = (
                switch_analysis.compute_stat_distances_for_all_nodes(
                    graph=data_graph,
                    adj_seqs=adj_seqs,
                    coeff_specs=[coeffs_spec],
                    batch_size=batch_size,
                    num_samples=num_samples,
                )
            )
            # np.save(os.path.join(save_folder, f"hellinger_distances.npy"), hellinger_distances.cpu().numpy())
            np.save(
                os.path.join(save_folder, f"tv_distances.npy"),
                tv_distances.cpu().numpy(),
            )


def compute_stat_distances_multiscale(dataset, batch_size: int, num_samples: int):
    distr_dict = {
        "geom1": rcoeffs.CoeffsSpec(name="geometric", kwargs={"tau": 1.0}, loc=0),
        "geom3": rcoeffs.CoeffsSpec(name="geometric", kwargs={"tau": 3.0}, loc=0),
        "poisson2": rcoeffs.CoeffsSpec(name="poisson", kwargs={"tau": 2.0}, loc=0),
        "uniform3": rcoeffs.CoeffsSpec(name="uniform", kwargs={"tau": 3.0}, loc=0),
        "poisson6": rcoeffs.CoeffsSpec(name="poisson", kwargs={"tau": 6.0}, loc=0),
    }
    data_graph = switch_analysis.load_graph(dataset)
    coeff_specs = [spec for key, spec in distr_dict.items()]
    for adj_seq in ["U", "OU"]:
        save_folder = os.path.join(
            NEB_ROOT,
            "neb_gcp_results",
            "multiscale_stat_distance_analysis",
            dataset,
            "5different",
            adj_seq,
        )

        os.makedirs(save_folder, exist_ok=True)
        metadata = {
            "coeff_names": list(distr_dict.keys()),
            "coeffs": {
                distr_name: dc.asdict(coeffs_spec)
                for distr_name, coeffs_spec in distr_dict.items()
            },
            "adj_seq": adj_seq,
        }
        with open(os.path.join(save_folder, "metadata.pkl"), "wb") as fp:
            pickle.dump(obj=metadata, file=fp)

        hellinger_distances, tv_distances = (
            switch_analysis.compute_stat_distances_for_all_nodes(
                graph=data_graph,
                adj_seqs=adj_seq,
                coeff_specs=coeff_specs,
                batch_size=batch_size,
                num_samples=num_samples,
            )
        )
        # np.save(os.path.join(save_folder, f"hellinger_distances.npy"), hellinger_distances.cpu().numpy())
        np.save(
            os.path.join(save_folder, f"tv_distances.npy"), tv_distances.cpu().numpy()
        )


def main_stats_distances():
    datasets = [
        # "ogb_arxiv",
        # "pokec_gender",
        # "snap_patents",
        # "roman_empire",
        # "pyg_cora",
        "pyg_cora_ml",
        # "subelj_cora",
        # "pyg_citeseer",
        # "cocite",
        # "pubmed",
        # "pyg_email_eu_core",
        # "enron_na",
        # "polblogs",
        # "flylarva",
    ]
    for dataset in (pbar := tqdm.tqdm(datasets)):
        pbar.set_description(f"Analysing {dataset}")
        if dataset in {"snap_patents", "pokec_gender", "ogb_arxiv"}:
            batch_size = 128
            num_samples = 200000
        else:
            batch_size = 2048
            num_samples = -1
        compute_stat_distances(
            dataset=dataset, batch_size=batch_size, num_samples=num_samples
        )
        compute_stat_distances_multiscale(
            dataset=dataset, batch_size=batch_size, num_samples=num_samples
        )


if __name__ == "__main__":
    main_stats_distances()
