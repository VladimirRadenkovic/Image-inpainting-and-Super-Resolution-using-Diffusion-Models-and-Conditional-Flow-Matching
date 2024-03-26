import argparse
import os
import pathlib
from typing import Iterable, List, Tuple, Union
import subprocess
import json
import glob

import biotite.structure as bs
import dill
import csv
import joblib
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch_geometric.data import Data
import biotite.structure.io.pdb as pdb

from src.utils.biotite_utils import atom_array_from_numpy, ca_backbone_from_atom_array
from src.evaluation.plot_pipeline import run_plot_pipeline
from src.evaluation.novelty.novelty_calculation import find_closest_structure

from proteinmpnn.run import nll_score
from proteinmpnn.data import BackboneSample, untokenise_sequence
from proteinmpnn.run import load_protein_mpnn_model, set_seed


__all__ = [
    "EvaluationPipeline",
    "EvaluationStage",
    "BackboneCAEvaluator",
    "BackboneCAAngleEvaluator",
    "BackboneCASSEEvaluator",
    "RadiusOfGyrationEvaluator",
    "VolumeEvaluator",
    "BackboneCANoveltyEvaluator",
    "BackboneSanityCheck",
    "ProteinMPNNEvaluator",
    "find_files",
    "calculate_ca_distance_statistics",
    "calculate_ca_angle_statistics",
    "calculate_secondary_structure_statistics",
    "calculate_radius_of_gyration",
    "calculate_volume_and_sphericality",
    "calculate_proteinmpnn_scores"
]


def find_files(path: str, ext: str, depth: int = 3) -> List[str]:
    """Find files up to `depth` levels down that match the given file extension.

    Args:
        path (str): The starting directory path.
        ext (str): The file extension to match.
        depth (int, optional): Maximum number of subdirectory levels to search. Defaults to 3.

    Returns:
        list: A list of file paths that match the file extension.
    """
    path = str(path)
    if depth < 0:
        logger.error("Depth cannot be negative.")
        return []
    elif not os.path.isdir(path):
        logger.error(f"Path {path} does not exist.")
        return []

    files = []
    root_depth = path.rstrip(os.path.sep).count(os.path.sep)
    for dirpath, dirs, filenames in os.walk(path):
        current_depth = dirpath.count(os.path.sep)
        if current_depth - root_depth <= depth:
            for filename in filenames:
                if filename.endswith(ext):
                    files.append(os.path.join(dirpath, filename))
        if current_depth >= root_depth + depth:
            # Modify dirs in-place to limit os.walk's recursion
            dirs.clear()

    logger.info(f"Found {len(files)} files with extension {ext} in {path}.")
    return files


# General evaluation pipeline
class EvaluationPipeline:
    def __init__(
        self,
        stages: List["EvaluationStage"],
        keep_objects: bool = False,
        transform: callable = None,
    ):
        self.stages = stages
        self.keep_objects = keep_objects
        self.transform = transform

    def __repr__(self):
            stages = "\n\t".join([f"{stage}" for stage in self.stages])
            return f"Pipeline(keep_objects={self.keep_objects}, has_transform={self.transform is not None}, stages=(\n\t{stages}\n)"
    
    def __call__(self, sample: "Data"):
        if isinstance(sample, (pathlib.Path, str)):
            sample = self._load_sample_from_path(sample)
        eval_dict, obj_dict = self.eval_single(sample)
        return eval_dict, obj_dict

    def _load_sample_from_path(self, path: Union[pathlib.Path, str]):
        if isinstance(path, str):
            path = pathlib.Path(path)
        assert path.exists(), f"File does not exist: {path}"

        if path.suffix == ".npy":
            sample = np.load(path)
        elif path.suffix == ".pt":
            sample = torch.load(path, map_location=torch.device("cpu"))

        try:
            sample.id
        except AttributeError:
            sample.id = str(path)
        return sample

    def eval_single(self, sample: "Data") -> Tuple[dict, dict]:
        if self.transform is not None:
            sample = self.transform(sample)
        try:
            sample_id = sample.id
        except AttributeError:
            sample_id = "unknown"

        eval_dict = dict(id=sample_id)
        object_dict = dict()

        for stage in self.stages:
            eval_dict2, object_dict2 = stage(sample)
            eval_dict.update(eval_dict2)
            if self.keep_objects:
                object_dict.update(object_dict2)
        return eval_dict, object_dict

    def eval_many(
        self, samples: Iterable["Data"], n_jobs: int = None
    ) -> Tuple[List[dict], List[dict]]:
        eval_dicts = []
        object_dicts = []

        if n_jobs is None or n_jobs == 1 or len(samples) == 1:
            for sample in samples:
                eval_dict, object_dict = self(sample)
                eval_dicts.append(eval_dict)
                if self.keep_objects:
                    object_dicts.append(object_dict)
        else:
            with joblib.parallel_backend("multiprocessing"):
                results = joblib.Parallel(n_jobs=n_jobs)(
                    joblib.delayed(self)(sample) for sample in samples
                )
            eval_dicts, object_dicts = zip(*results)
            if not self.keep_objects:
                object_dicts = []
            eval_dicts = list(eval_dicts)
            object_dicts = list(object_dicts)

        return eval_dicts, object_dicts

    def eval_dir(
        self, dir_path: str, ext: str, n_jobs: int = None, depth: int = 4
    ) -> Tuple[List[dict], List[dict]]:
        logger.info(
            f"Searching for files with extension {ext} in {dir_path} up to depth {depth}..."
        )
        assert os.path.exists(dir_path), f"Directory does not exist: {dir_path}"
        assert os.path.isdir(dir_path), f"Path is not a directory: {dir_path}"
        files = find_files(dir_path, ext)
        return self.eval_many(files, n_jobs=n_jobs)


class EvaluationStage:
    STATISTICS_SELECTOR = dict(mean=np.mean, std=np.std, min=np.min, max=np.max, median=np.median)

    def __init__(
        self, statistics: List[Union[str, callable]] = ["mean", "std", "min", "max", "median"]
    ):
        self.statistic_funcs = dict()
        for s in statistics:
            if isinstance(s, str):
                assert (
                    s in self.STATISTICS_SELECTOR
                ), f"Statistic {s} not supported. Supported statistics are: {self.STATISTICS_SELECTOR.keys()}"
                self.statistic_funcs[s] = self.STATISTICS_SELECTOR[s]
            elif isinstance(s, callable):
                self.statistic_funcs[s.__name__] = s
            else:
                raise ValueError(f"Statistic {s} is neither a string nor a callable.")

    def __call__(self, sample: "Data") -> Tuple[dict, dict]:
        raise NotImplementedError

    def __repr__(self):
        statistic_names = ""
        if hasattr(self, "statistic_funcs"):
            statistic_names = list(self.statistic_funcs.keys())
        return f"{self.__class__.__name__}({statistic_names})"


# ====================
# Backbone evaluation
# ====================


class BackboneCAEvaluator(EvaluationStage):
    def __call__(self, sample: "Data"):
        eval_dict = dict()
        object_dict = dict()

        pos = sample.pos.numpy()
        ca_distances = calculate_ca_distance_statistics(
            pos, statistic_funcs=self.statistic_funcs, eval_dict=eval_dict
        )
        object_dict["ca_distances"] = ca_distances

        return eval_dict, object_dict


def calculate_ca_distance_statistics(
    pos: np.ndarray,
    statistic_funcs: dict = EvaluationStage.STATISTICS_SELECTOR,
    eval_dict: dict = {},
) -> np.ndarray:
    """Calculate the distances between the CA atoms of a protein chain.

    Args:
        pos (np.ndarray): The positions of the CAatoms in the protein chain.
        eval_dict (dict): The dictionary to store the evaluation results in.
        statistic_funcs (dict): The dictionary of statistics to compute.

    Returns:
        np.ndarray: The distances between the CA atoms.
    """
    ca_distances = np.linalg.norm(pos[1:] - pos[:-1], axis=1)

    eval_dict["n_ca_atoms"] = len(ca_distances) + 1
    for name, func in statistic_funcs.items():
        eval_dict[f"ca_distance_{name}"] = func(ca_distances)

    return ca_distances


class BackboneCAAngleEvaluator(EvaluationStage):
    def __call__(self, sample: "Data"):
        eval_dict = dict()
        object_dict = dict()

        pos = sample.pos.numpy()
        ca_angles = calculate_ca_angle_statistics(
            pos, statistic_funcs=self.statistic_funcs, eval_dict=eval_dict
        )
        object_dict["ca_angles"] = ca_angles

        return eval_dict, object_dict


def calculate_ca_angle_statistics(
    pos: np.ndarray,
    statistic_funcs: dict = EvaluationStage.STATISTICS_SELECTOR,
    eval_dict: dict = {},
) -> np.ndarray:
    """Calculate the angles between the CA atoms of a protein chain.

    Args:
        pos (np.ndarray): The positions of the CAatoms in the protein chain.
        eval_dict (dict): The dictionary to store the evaluation results in.
        statistic_funcs (dict): The dictionary of statistics to compute.

    Returns:
        np.ndarray: The angles between the CA atoms.
    """
    ca_vectors = pos[1:] - pos[:-1]
    # Calculate the angles between consecutive CA vectors
    vec_norms = np.linalg.norm(ca_vectors, axis=1)
    vec_normalised = ca_vectors / vec_norms[:, None]
    vec_cos = np.sum(vec_normalised[:-1] * vec_normalised[1:], axis=1)
    ca_angles = np.degrees(np.arccos(vec_cos))

    for name, func in statistic_funcs.items():
        eval_dict[f"ca_angle_{name}"] = func(ca_angles)
    return ca_angles


class BackboneCASSEEvaluator(EvaluationStage):
    def __init__(self):
        pass

    def __call__(self, sample: "Data"):
        eval_dict = dict()
        object_dict = dict()

        pos = sample.pos.numpy()
        sse = calculate_secondary_structure_statistics(pos, eval_dict=eval_dict)
        object_dict["sse"] = sse

        return eval_dict, object_dict


def calculate_secondary_structure_statistics(pos: np.ndarray, eval_dict: dict = {}) -> np.ndarray:
    """Calculate the secondary structure of a protein chain.

    Args:
        pos (np.ndarray): The positions of the CAatoms in the protein chain.
        eval_dict (dict): The dictionary to store the evaluation results in.

    Returns:
        np.ndarray: The secondary structure of the protein chain.
    """
    sse = bs.annotate_sse(atom_array_from_numpy(pos))
    eval_dict["helix_proportion"] = np.sum(sse == "a") / len(sse)
    eval_dict["sheet_proportion"] = np.sum(sse == "b") / len(sse)
    eval_dict["coil_proportion"] = np.sum(sse == "c") / len(sse)
    return sse


class BackboneSanityCheck(EvaluationStage):
    def __init__(self, canvas_size: float = 21.0):
        self.canvas_size = canvas_size

    def __call__(self, sample: "Data"):
        eval_dict = dict(has_nan=False, exceeds_canvas=False)

        pos = sample.pos.numpy()

        # Check if any pos is nan
        if np.isnan(pos).any():
            eval_dict["has_nan"] = True
            logger.warning("NaN positions in sample {}", id)
        if (np.abs(pos) > self.canvas_size).any():
            eval_dict["exceeds_canvas"] = True

        return eval_dict, {}


class RadiusOfGyrationEvaluator(EvaluationStage):
    def __init__(self):
        pass

    def __call__(self, sample: "Data"):
        eval_dict = dict()
        object_dict = dict()

        pos = sample.pos.numpy()
        radius_of_gyration = calculate_radius_of_gyration(pos)
        eval_dict["radius_of_gyration"] = radius_of_gyration

        return eval_dict, object_dict


def calculate_radius_of_gyration(pos: np.ndarray) -> float:
    num_points, num_coords = pos.shape
    # Calculate the centroid of the point cloud
    centroid = np.mean(pos, axis=0)

    # Calculate the squared distances from the centroid
    squared_distances = np.sum((pos - centroid) ** 2, axis=1)

    # Calculate the radius of gyration
    radius_of_gyration = np.sqrt(np.mean(squared_distances))

    return radius_of_gyration


class VolumeEvaluator(EvaluationStage):
    def __init__(self):
        pass

    def __call__(self, sample: "Data"):
        eval_dict = dict()
        object_dict = dict()

        pos = sample.pos.numpy()
        calculate_volume_and_shpericality(pos, eval_dict=eval_dict)

        return eval_dict, object_dict


def calculate_volume_and_shpericality(pos: np.ndarray, eval_dict: dict = {}):
    from scipy.spatial import ConvexHull

    # Calculate the centroid of the point cloud
    centroid = np.mean(pos, axis=0)

    # Calculate distance from the centroid
    distances = np.sqrt(np.sum((pos - centroid) ** 2, axis=1))
    reference_distance = np.mean(distances)
    max_distance = np.max(distances)

    # Compute the volume of the sphere
    sphere_volume = (4 / 3) * np.pi * reference_distance**3

    # Compute the convex hull of the point cloud
    hull = ConvexHull(pos)
    convex_hull_volume = hull.volume

    # Compute shpericality
    shpericality = convex_hull_volume / sphere_volume

    eval_dict["sphere_volume"] = sphere_volume
    eval_dict["convex_hull_volume"] = convex_hull_volume
    eval_dict["shpericality"] = shpericality
    eval_dict["frac_of_bounding_sphere"] = convex_hull_volume / (4 / 3 * np.pi * max_distance**3)

    return convex_hull_volume, shpericality


class BackboneCANoveltyEvaluator(EvaluationStage):
    def __init__(self, reference_structures: dict):
        from src.evaluation.novelty.novelty_calculation import find_closest_structure

        self.find_closest_structure = find_closest_structure
        self.reference_structures = reference_structures

    def __repr__(self):
        return f"{self.__class__.__name__}(n_reference_structures={len(self.reference_structures)})"

    def __call__(self, sample: "Data"):
        eval_dict = dict()
        object_dict = dict()

        pos = sample.pos.numpy()
        result = self.find_closest_structure(pos, self.reference_structures)
        eval_dict.update(result)

        return eval_dict, object_dict

#utils for ProteinMPNN sequence design
def create_backbone(motif_inds, motif_res, protein_length):
    res_name = ['X']*protein_length
    res_mask = np.ones(protein_length)

    for i, index in enumerate(motif_inds):
        res_name[index] = motif_res[i]
        res_mask[index] = 0

    res_name = "".join(res_name)
    res_mask = np.array(res_mask, dtype=int)
    
    backbone = BackboneSample(bb_coords=np.random.rand(protein_length, 3), 
                              ca_only=True, 
                              res_name=res_name, 
                              res_mask=res_mask)
    
    return backbone
    
def calculate_proteinmpnn_scores(pos, model, n_seq=8, motif_inds=None, motif_res=None, device="cuda"):
    if motif_inds:
        backbones = [create_backbone(motif_inds, motif_res, pos.shape[0])]
    else:
        backbones = [BackboneSample(bb_coords=pos, 
                             ca_only=True)]
    # with torch.inference_mode():
    samples = [model.sample(
    randn=torch.randn(1, backbone.n_residues, device=device), 
    **backbone.to_protein_mpnn_input("sampling")) 
    for backbone in backbones
    ]
    for sample, backbone in zip(samples, backbones):
        inpt = backbone.to_protein_mpnn_input("scoring")
        inpt["decoding_order"] = sample["decoding_order"]
        inpt["S"] = sample["S"]
        sample_scores = np.zeros(n_seq)
        sample_seqs = []
        for i in range(n_seq):
            #set random seed to get diversity in the sequences produced
            set_seed(i)
            log_probs = model(randn=torch.randn(1, backbone.n_residues), 
                        use_input_decoding_order=True, 
                        **inpt)
            sample["nll_score"] = nll_score(sample["S"], log_probs, mask=inpt["mask"])
            sample["prob"] = torch.exp(-sample["nll_score"])
            protein_mpnn_score = sample["prob"].item()
            protein_mpnn_seq = untokenise_sequence(sample["S"])
            sample_scores[i] = protein_mpnn_score
            sample_seqs.append(protein_mpnn_seq)
    return sample_scores, sample_seqs

def run_proteinmpnn_eval(dir_path, n_seq=8, device="cuda", motif_inds=None, motif_res=None):
    model = load_protein_mpnn_model(model_type="ca")
    scores = [] #holds ProteinMPNN scores
    seqs = [] #holds ProteinMPNN designed sequences
    #load samples similar to eval_many
    assert os.path.exists(dir_path), f"Directory does not exist: {dir_path}"
    assert os.path.isdir(dir_path), f"Path is not a directory: {dir_path}"
    files = find_files(dir_path, ".pt")

    for sample_file in files:
        sample = torch.load(sample_file)
        sample = _scale_pos(sample)
        protein_mpnn_scores, protein_mpnn_seqs = calculate_proteinmpnn_scores(sample.pos, model, n_seq, motif_inds, motif_res)
        #append to eval_dict
        scores.append(protein_mpnn_scores)
        seqs.append(protein_mpnn_seqs)
    return scores, seqs

def write_protein_mpnn_seqs_file(protein_mpnn_objects, output_file):
    """
    Write a csv file containing the ProteinMPNN sequences for each sample for consumption by ColabFold.
    """
    with open(output_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "sequence"])
        for i, sublist in enumerate(protein_mpnn_objects):
            for j, string in enumerate(sublist):
                seq_id = f"sample{i}_{j}"
                writer.writerow([seq_id, string])

def process_af2_output(af2_output_dir, sample_dir, n_seq=8, motif_inds=None, motif_res=None):
    """
    Process the output of AlphaFold2 for evaluation.
    Args:

    Returns:
        plddt: mean plddt for samples
        max_pae: max pae for samples
        pTM: 
        scTM: max TM score between sampled backbone and AlphaFold2 prediction
        scRMSD: min RMSD between sampled backbone and AlphaFold2 prediction
        scmotifRMSD: min RMSD between sampled backbone and AlphaFold2 prediction for motif residues only
    """
    files = find_files(sample_dir, ".pt")
    plddt = []
    max_pae = []
    pTM = []
    for i, file in enumerate(files):
        plddt_sample = []
        max_pae_sample = []
        pTM_sample = []
        sample = torch.load(file)
        af2_structures = dict()
        for j in range(n_seq+1):
            json_file_pattern = f"{af2_output_dir}/sample{i}_{j}_scores_rank_001*.json"
            json_file_path = glob.glob(json_file_pattern)
            if json_file_path:
                with open(json_file_path[0]) as json_file:
                    data = json.load(json_file)
                    mean_plddt = np.mean(data["plddt"])
                    plddt_sample.append(mean_plddt)
                    max_pae_sample.append(data["max_pae"])
                    pTM_sample.append(data["ptm"])

            pdb_file_pattern = f"{af2_output_dir}/sample{i}_{j}_relaxed_rank_001*.pdb"
            pdb_file_path = glob.glob(pdb_file_pattern)
            # if pdb_file_path:
            #     pdb_file_path = pdb_file_path[0]
            #     pdb_file = pdb.PDBFile.read(pdb_file_path)
            #     structure = pdb_file.get_structure()
            #     coords = ca_backbone_from_atom_array(structure)
            #     af2_structures[f"sample{i}_{j}"] = coords
            
            #calculate scTM, scRMSD, scmotifRMSD
            # alignment_output = find_closest_structure(sample, af2_structures)
            # scTM = max(alignment_output["TM"])
            # scRMSD = min(alignment_output["RMSD"])

            # #!TODO: continue here
        #gather stats from seqs per sample
        plddt.append(plddt_sample)
        max_pae.append(max_pae_sample)
        pTM.append(pTM_sample)
    af2_pdb_files = []
    af2_data = {"plddt": plddt, "max_pae": max_pae, "pTM": pTM}
    return af2_data, af2_pdb_files





    



if __name__ == "__main__":
    from src.constants import PROJECT_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_dir",
        type=str,
        default="/home/ked48/rds/hpc-work/protein-diffusion/data/exp_data/cond_new_denoise_cont_3steps_nonoise/samples",
        help="The directory containing the samples to evaluate. Samples can be several folders deep.",
    )
    parser.add_argument(
        "--training_dataset",
        type=str,
        default="/home/ked48/rds/hpc-work/protein-diffusion/data/cath_calpha/raw/pdb_domain_ca_coords_v2023-04-25.npz",
        help="The training dataset to evaluate against. Must be a .npz file (at the moment).",
    )
    parser.add_argument(
        "--keep_objects",
        type=bool,
        default=False,
        help="Whether to keep the objects returned by the evaluation stages.",
    )
    parser.add_argument(
        "--motif_inds",
        type=List[int],
        default=None,
        help="The indices of the motif residues.",
    )
    parser.add_argument(
        "--motif_res",
        type=List[str],
        default=None,
        help="The residues of the motif as one letter codes.",
    )

    args = parser.parse_args()
    sample_dir = pathlib.Path(args.sample_dir)
    training_dataset = pathlib.Path(args.training_dataset)
    keep_objects = args.keep_objects
    motif_inds = args.motif_inds
    motif_res = args.motif_res

    # ==================================
    # Evaluate the training dataset
    logger.info("Evaluating training dataset")
    logger.info("Loading training dataset ... ")
    pdb_data = np.load(training_dataset)
    pdb_data = {
        sample_id: pdb_data[sample_id]
        for sample_id in pdb_data.files
        if len(pdb_data[sample_id]) > 0
    }
    logger.info("... done loading training dataset")

    training_dataset = pathlib.Path(training_dataset)
    training_dataset_statistics = training_dataset.with_name(training_dataset.stem + "_stats.csv")

    if training_dataset_statistics.exists():
        logger.info("Loading training dataset statistics from {}", training_dataset_statistics)
        pdb_data_stats = pd.read_csv(training_dataset_statistics)
    else:

        def _load_npz_to_data(key_value):
            key, pos = key_value
            com = np.mean(pos, axis=0)

            return Data(id=key, pos=torch.from_numpy(pos - com))

        npz_evaluation_pipeline = EvaluationPipeline(
            [
                BackboneSanityCheck(canvas_size=21.0),
                BackboneCAEvaluator(),
                BackboneCAAngleEvaluator(),
                BackboneCASSEEvaluator(),
                RadiusOfGyrationEvaluator(),
                VolumeEvaluator(),
            ],
            keep_objects=keep_objects,
            transform=_load_npz_to_data,
        )

        logger.info("Starting evaluation ...")
        pdb_data_stats, pdb_data_objects = npz_evaluation_pipeline.eval_many(
            pdb_data.items(), n_jobs=-1
        )
        logger.info("... done evaluation")

        # Save statistics
        pdb_data_stats = pd.DataFrame.from_records(pdb_data_stats)
        pdb_data_stats.to_csv(training_dataset_statistics, index=False)
        # Save objects with dill
        if keep_objects:
            with open(
                training_dataset.with_name(training_dataset.stem + "_objects.dill"), "wb"
            ) as f:
                dill.dump(pdb_data_objects, f)

    # ==================================
    # Evaluate the samples
    sample_dir = pathlib.Path(sample_dir)
    save_path = sample_dir / "sample_stats.csv"

    def _scale_pos(sample: "Data", factor: float = 15.0):
        sample.pos = sample.pos * factor
        return sample

    evaluation_pipeline = EvaluationPipeline(
        [
            BackboneSanityCheck(canvas_size=21.0),
            BackboneCAEvaluator(),
            BackboneCAAngleEvaluator(),
            BackboneCASSEEvaluator(),
            RadiusOfGyrationEvaluator(),
            VolumeEvaluator(),
            # BackboneCANoveltyEvaluator(
            #      reference_structures=pdb_data
            #  ),  # WARNING: This is slow! (takes ~2min per sample)
        ],
        keep_objects=True,
        transform=_scale_pos,
    )

    logger.info("1/3 Starting backbone evaluation ...")
    eval_stats, eval_objects = evaluation_pipeline.eval_dir(sample_dir, ext=".pt", n_jobs=-1)
    logger.info("2/3 Starting sequence prediction ...")
    protein_mpnn_stats, protein_mpnn_objects = run_proteinmpnn_eval(sample_dir, n_seq=3, motif_inds=motif_inds, motif_res=motif_res)
    logger.info("3/3 Starting structure prediction ...")
    protein_mpnn_seqs_file = sample_dir / "protein_mpnn_seqs.csv"
    write_protein_mpnn_seqs_file(protein_mpnn_objects, protein_mpnn_seqs_file)
    af2_output_dir = sample_dir / "af2_output"
    af2_output_dir.mkdir(exist_ok=True, parents=True)
    cmd = ['colabfold_batch', '--templates', '--amber', protein_mpnn_seqs_file, af2_output_dir]
    subprocess.run(cmd, check=True)
    af2_data, af2_pdb_files = process_af2_output(af2_output_dir, sample_dir, n_seq=3, motif_inds=motif_inds, motif_res=motif_res)
    logger.info("... done evaluation")

    # Save statistics
    eval_stats = pd.DataFrame.from_records(eval_stats)
    eval_stats["protein_mpnn_scores"] = protein_mpnn_stats
    eval_stats["protein_mpnn_seqs"] = protein_mpnn_objects
    eval_stats["plddt"] = af2_data["plddt"]
    eval_stats["max_pae"] = af2_data["max_pae"]
    eval_stats["pTM"] = af2_data["pTM"]

    eval_stats.to_csv(sample_dir / "sample_stats.csv", index=False)
    # Save objects with dill
    if keep_objects:
        with open(sample_dir / "sample_objects.dill", "wb") as f:
            dill.dump(eval_objects, f)

    logger.info(f"Statistics saved to {sample_dir}. Proceeding to plotting")
    plot_dir = sample_dir / "eval_plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    run_plot_pipeline(eval_stats, pdb_data_stats, plot_dir=plot_dir, data_dir=sample_dir)