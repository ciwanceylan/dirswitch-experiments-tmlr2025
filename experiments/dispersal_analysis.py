import argparse
import os
import pandas as pd
import tqdm
import nebtools.experiments.switch_analysis as switch_analysis
import reachnes.coeffs as rcoeffs
from nebtools.utils import NEB_ROOT


def compute_dispersal(dataset, batch_size: int, num_samples: int, num_steps: int):
    data_graph = switch_analysis.load_graph(dataset)
    coeff_specs = [
        rcoeffs.CoeffsSpec(name="uniform", kwargs={"tau": 5}, loc=0),
        rcoeffs.CoeffsSpec(name="poisson", kwargs={"tau": 2.0}, loc=0),
    ]
    adj_orientations = ["U", "O", "OI", "OIO", "OU", "OIU", "OIOU"]
    results = []
    for coeff_spec in tqdm.tqdm(coeff_specs):
        for adj_seqs in adj_orientations:
            _, entropy_results = switch_analysis.compute_dispersal(
                data_graph,
                adj_seqs=adj_seqs,
                coeff_spec=coeff_spec,
                batch_size=batch_size,
                num_samples=num_samples,
                num_steps=num_steps,
            )
            df = pd.DataFrame(
                {
                    "entropy": entropy_results,
                    "adj_seq": adj_seqs,
                    "rwl_distrs": coeff_spec.name,
                    "tau": coeff_spec.kwargs["tau"],
                    "order": num_steps,
                    "loc": coeff_spec.loc,
                }
            )
            results.append(df)
    results = pd.concat(results, axis=0, ignore_index=True)
    return results


def compute_dispersal_over_tau(
    dataset, batch_size: int, num_samples: int, num_steps: int
):
    data_graph = switch_analysis.load_graph(dataset)
    coeff_specs = []
    tau_values = [0.0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12]
    for tau in tau_values:
        coeff_specs.append(
            rcoeffs.CoeffsSpec(name="poisson", kwargs={"tau": tau}, loc=0)
        )
        coeff_specs.append(
            rcoeffs.CoeffsSpec(name="geometric", kwargs={"tau": tau}, loc=0)
        )
    results = []
    for coeff_spec in tqdm.tqdm(coeff_specs):
        for adj_seqs in ["U", "O", "OU"]:
            _, entropy_results = switch_analysis.compute_dispersal(
                data_graph,
                adj_seqs=adj_seqs,
                coeff_spec=coeff_spec,
                batch_size=batch_size,
                num_samples=num_samples,
                num_steps=num_steps,
            )
            df = pd.DataFrame(
                {
                    "entropy": entropy_results,
                    "adj_seq": adj_seqs,
                    "rwl_distrs": coeff_spec.name,
                    "tau": coeff_spec.kwargs["tau"],
                    "order": num_steps,
                    "loc": coeff_spec.loc,
                }
            )
            results.append(df)

    # coverage_results = pd.DataFrame(data=coverage_results)
    results = pd.concat(results, axis=0, ignore_index=True)
    return results


def main_dispersal(num_steps: int):
    datasets = [
        # "roman_empire",
        # "pyg_cora",
        "pyg_cora_ml",
        # "subelj_cora",
        # "pyg_citeseer",
        # "cocite",
        # "pubmed",
        # "pyg_email_eu_core",
        # "email_eu_core_na",
        # "enron_na",
        # "polblogs",
        # "flylarva",
        # "wikivotes"
        # "ogb_arxiv",
        # "pokec_gender",
        # "snap_patents",
    ]
    for dataset in (pbar := tqdm.tqdm(datasets)):
        pbar.set_description(f"Analysing {dataset}")
        if dataset in {"snap_patents", "pokec_gender", "ogb_arxiv"}:
            batch_size = 224
            num_samples = 200000
        else:
            batch_size = 2048
            num_samples = -1
        save_folder = os.path.join(
            NEB_ROOT, "neb_gcp_results", f"dispersal_analysis_{num_steps}", dataset
        )
        os.makedirs(save_folder, exist_ok=True)
        entropy_results = compute_dispersal(
            dataset, batch_size=batch_size, num_samples=num_samples, num_steps=num_steps
        )
        entropy_results.to_pickle(
            os.path.join(save_folder, f"entropy_{dataset}_{num_steps}.pkl")
        )


def main_dispersal_over_tau(num_steps: int):
    datasets = [
        # "roman_empire",
        # "pyg_cora",
        "pyg_cora_ml",
        # "subelj_cora",
        # "pyg_citeseer",
        # "cocite",
        # "pubmed",
        # "pyg_email_eu_core",
        # "email_eu_core_na",
        # "enron_na",
        # "polblogs",
        # "flylarva",
        # "wikivotes"
        # "ogb_arxiv",
        # "pokec_gender",
        # "snap_patents",
    ]
    for dataset in (pbar := tqdm.tqdm(datasets)):
        pbar.set_description(f"Analysing {dataset}")
        if dataset in {"snap_patents", "pokec_gender", "ogb_arxiv"}:
            batch_size = 224
            num_samples = 200000
        else:
            batch_size = 2048
            num_samples = -1
        save_folder = os.path.join(
            NEB_ROOT,
            "neb_gcp_results",
            f"dispersal_analysis_over_tau_{num_steps}",
            dataset,
        )
        os.makedirs(save_folder, exist_ok=True)
        entropy_results = compute_dispersal_over_tau(
            dataset, batch_size=batch_size, num_samples=num_samples, num_steps=num_steps
        )
        entropy_results.to_pickle(
            os.path.join(save_folder, f"entropy_{dataset}_{num_steps}.pkl")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directions",
        action="store_true",
        default=False,
        help="Compute dispersal for multiple different edge direction specifiers (Figure 6 in the paper).",
    )
    parser.add_argument(
        "--tau",
        action="store_true",
        default=False,
        help="Compute dispersal for multiple different tau values (Figure 5 in the paper).",
    )
    args = parser.parse_args()

    if args.directions:
        main_dispersal(num_steps=20)
    if args.tau:
        main_dispersal_over_tau(num_steps=20)
