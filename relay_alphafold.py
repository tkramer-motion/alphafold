import enum
import json
import os
import pathlib
import pickle
import random
import sys
import time
import zipfile
from typing import Any, Dict, Union

import jax.numpy as jnp
import numpy as np
import smart_open
from absl import logging

from alphafold.common import confidence, protein, residue_constants
from alphafold.data import pipeline, pipeline_multimer, templates
from alphafold.data.tools import hmmsearch
from alphafold.model import config, data, model
from alphafold.relax import relax

logging.set_verbosity(logging.INFO)


@enum.unique
class ModelsToRelax(enum.Enum):
    ALL = 0
    BEST = 1
    NONE = 2


MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _jnp_to_np(output: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively changes jax arrays to numpy arrays."""
    for k, v in output.items():
        if isinstance(v, dict):
            output[k] = _jnp_to_np(v)
        elif isinstance(v, jnp.ndarray):
            output[k] = np.array(v)
    return output


def _save_confidence_json_file(
        plddt: np.ndarray, output_dir: str, model_name: str
) -> None:
    confidence_json = confidence.confidence_json(plddt)

    # Save the confidence json.
    confidence_json_output_path = os.path.join(
        output_dir, f'confidence_{model_name}.json'
    )
    with open(confidence_json_output_path, 'w') as f:
        f.write(confidence_json)


def _save_mmcif_file(
        prot: protein.Protein,
        output_dir: str,
        model_name: str,
        file_id: str,
        model_type: str,
) -> None:
    """Crate mmCIF string and save to a file.

    Args:
      prot: Protein object.
      output_dir: Directory to which files are saved.
      model_name: Name of a model.
      file_id: The file ID (usually the PDB ID) to be used in the mmCIF.
      model_type: Monomer or multimer.
    """

    mmcif_string = protein.to_mmcif(prot, file_id, model_type)

    # Save the MMCIF.
    mmcif_output_path = os.path.join(output_dir, f'{model_name}.cif')
    with open(mmcif_output_path, 'w') as f:
        f.write(mmcif_string)


def _save_pae_json_file(
        pae: np.ndarray, max_pae: float, output_dir: str, model_name: str
) -> None:
    """Check prediction result for PAE data and save to a JSON file if present.

    Args:
      pae: The n_res x n_res PAE array.
      max_pae: The maximum possible PAE value.
      output_dir: Directory to which files are saved.
      model_name: Name of a model.
    """
    pae_json = confidence.pae_json(pae, max_pae)

    # Save the PAE json.
    pae_json_output_path = os.path.join(output_dir, f'pae_{model_name}.json')
    with open(pae_json_output_path, 'w') as f:
        f.write(pae_json)


def predict_structure(
        fasta_path: str,
        fasta_name: str,
        output_dir_base: str,
        data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
        model_runners: Dict[str, model.RunModel],
        amber_relaxer: relax.AmberRelaxation,
        benchmark: bool,
        random_seed: int,
        models_to_relax: ModelsToRelax,
        model_type: str,
        suffix: str
) -> dict[str, Any]:
    """Predicts structure using AlphaFold for the given sequence."""
    logging.info('Predicting %s', fasta_name)
    timings = {}
    output_dir = os.path.join(output_dir_base, fasta_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    msa_output_dir = os.path.join(output_dir, 'msas')
    if not os.path.exists(msa_output_dir):
        os.makedirs(msa_output_dir)

    # Get features.
    t_0 = time.time()
    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        msa_output_dir=msa_output_dir)
    timings['features'] = time.time() - t_0

    # Write out features as a pickled dictionary.
    features_output_path = os.path.join(output_dir, 'features.pkl')
    with open(features_output_path, 'wb') as f:
        pickle.dump(feature_dict, f, protocol=4)

    unrelaxed_pdbs = {}
    unrelaxed_proteins = {}
    relaxed_pdbs = {}
    relax_metrics = {}
    ranking_confidences = {}

    # Run the models.
    num_models = len(model_runners)
    for model_index, (model_name, model_runner) in enumerate(
            model_runners.items()):
        logging.info('Running model %s on %s', model_name, fasta_name)
        t_0 = time.time()
        model_random_seed = model_index + random_seed * num_models
        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=model_random_seed)
        timings[f'process_features_{model_name}'] = time.time() - t_0

        t_0 = time.time()
        prediction_result = model_runner.predict(processed_feature_dict,
                                                 random_seed=model_random_seed)
        t_diff = time.time() - t_0
        timings[f'predict_and_compile_{model_name}'] = t_diff
        logging.info(
            'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
            model_name, fasta_name, t_diff)

        if benchmark:
            t_0 = time.time()
            model_runner.predict(processed_feature_dict,
                                 random_seed=model_random_seed)
            t_diff = time.time() - t_0
            timings[f'predict_benchmark_{model_name}'] = t_diff
            logging.info(
                'Total JAX model %s on %s predict time (excludes compilation time): %.1fs',
                model_name, fasta_name, t_diff)

        plddt = prediction_result['plddt']
        _save_confidence_json_file(plddt, output_dir, model_name)
        ranking_confidences[model_name] = prediction_result['ranking_confidence']

        if (
                'predicted_aligned_error' in prediction_result
                and 'max_predicted_aligned_error' in prediction_result
        ):
            pae = prediction_result['predicted_aligned_error']
            max_pae = prediction_result['max_predicted_aligned_error']
            _save_pae_json_file(pae, float(max_pae), output_dir, model_name)

        # Remove jax dependency from results.
        np_prediction_result = _jnp_to_np(dict(prediction_result))

        # Save the model outputs.
        result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
        with open(result_output_path, 'wb') as f:
            pickle.dump(np_prediction_result, f, protocol=4)

        # Add the predicted LDDT in the b-factor column.
        # Note that higher predicted LDDT value means higher model confidence.
        plddt_b_factors = np.repeat(
            plddt[:, None], residue_constants.atom_type_num, axis=-1)
        unrelaxed_protein = protein.from_prediction(
            features=processed_feature_dict,
            result=prediction_result,
            b_factors=plddt_b_factors,
            remove_leading_feature_dimension=not model_runner.multimer_mode)

        unrelaxed_proteins[model_name] = unrelaxed_protein
        unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
        unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
        with open(unrelaxed_pdb_path, 'w') as f:
            f.write(unrelaxed_pdbs[model_name])

        _save_mmcif_file(
            prot=unrelaxed_protein,
            output_dir=output_dir,
            model_name=f'unrelaxed_{model_name}',
            file_id=str(model_index),
            model_type=model_type,
        )

    # Rank by model confidence.
    ranked_order = [
        model_name for model_name, confidence in
        sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)]

    # Relax predictions.
    if models_to_relax == ModelsToRelax.BEST:
        to_relax = [ranked_order[0]]
    elif models_to_relax == ModelsToRelax.ALL:
        to_relax = ranked_order
    elif models_to_relax == ModelsToRelax.NONE:
        to_relax = []

    for model_name in to_relax:
        t_0 = time.time()
        relaxed_pdb_str, _, violations = amber_relaxer.process(
            prot=unrelaxed_proteins[model_name])
        relax_metrics[model_name] = {
            'remaining_violations': violations,
            'remaining_violations_count': sum(violations)
        }
        timings[f'relax_{model_name}'] = time.time() - t_0

        relaxed_pdbs[model_name] = relaxed_pdb_str

        # Save the relaxed PDB.
        relaxed_output_path = os.path.join(
            output_dir, f'relaxed_{model_name}.pdb')
        with open(relaxed_output_path, 'w') as f:
            f.write(relaxed_pdb_str)

        relaxed_protein = protein.from_pdb_string(relaxed_pdb_str)
        _save_mmcif_file(
            prot=relaxed_protein,
            output_dir=output_dir,
            model_name=f'relaxed_{model_name}',
            file_id='0',
            model_type=model_type,
        )

    # Write out relaxed PDBs in rank order.
    for idx, model_name in enumerate(ranked_order):
        ranked_output_path = os.path.join(output_dir, f'ranked_{suffix}_{idx}.pdb')
        with open(ranked_output_path, 'w') as f:
            if model_name in relaxed_pdbs:
                f.write(relaxed_pdbs[model_name])
            else:
                f.write(unrelaxed_pdbs[model_name])

    label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
    return {label: ranking_confidences, 'order': ranked_order}


def main(fasta_path: str,
         output_dir: str,
         suffix: str,
         num_multimer_predictions_per_model: int = 5,
         dropout: bool = False,
         max_recycles: int = 3,
         no_templates: bool = False,
         model_preset: str = "multimer") -> dict[str, Any]:
    num_ensemble = 1

    template_searcher = hmmsearch.Hmmsearch(
        binary_path="/usr/bin/hmmsearch",
        hmmbuild_binary_path="/usr/bin/hmmbuild",
        database_path="/data/pdb_seqres/pdb_seqres.txt")
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir="/data/pdb_mmcif/mmcif_files",
        max_template_date="2024-05-15",
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path="/usr/bin/kalign",
        release_dates_path=None,
        obsolete_pdbs_path="/data/pdb_mmcif/obsolete.dat")

    monomer_data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path="/usr/bin/jackhmmer",
        hhblits_binary_path="/usr/bin/hhblits",
        uniref90_database_path="/data/uniref90/uniref90.fasta",
        mgnify_database_path="/data/mgnify/mgy_clusters_2022_05.fa",
        bfd_database_path="/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
        uniref30_database_path="/data/uniref30/UniRef30_2021_03",
        small_bfd_database_path=None,
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_small_bfd=False,
        use_precomputed_msas=True,
        no_templates=no_templates)

    data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path="/usr/bin/jackhmmer",
        uniprot_database_path="/data/uniprot/uniprot.fasta",
        use_precomputed_msas=True)

    model_runners = {}
    model_names = config.MODEL_PRESETS[model_preset]
    for model_name in model_names:
        model_config = config.model_config(model_name)
        model_config.model.num_ensemble_eval = num_ensemble

        if dropout:
            model_config.model.num_ensemble_train = num_ensemble
            model_config.model.heads.structure_module.dropout = 0.0

        if max_recycles != 3:
            logging.info(f'Setting max_recycles to {max_recycles}')
            model_config.model.num_recycle = max_recycles

        model_params = data.get_model_haiku_params(
            model_name=model_name, data_dir="/data/")
        model_runner = model.RunModel(model_config, model_params, is_training=dropout)
        for i in range(num_multimer_predictions_per_model):
            model_runners[f'{model_name}_{suffix}_{i}'] = model_runner

    logging.info('Have %d models: %s', len(model_runners),
                 list(model_runners.keys()))

    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
        use_gpu=False)

    random_seed = random.randrange(sys.maxsize // len(model_runners))
    logging.info('Using random seed %d for the data pipeline', random_seed)

    fasta_name = pathlib.Path(fasta_path).stem
    return predict_structure(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=False,
        random_seed=random_seed,
        models_to_relax=ModelsToRelax.BEST,
        model_type="Multimer",
        suffix=suffix
    )


def get_high_confidence_prediction(results: dict[str, Any]) -> float:
    rows = []
    for model_name, confidence in results["iptm+ptm"].items():
        rows.append((model_name, confidence))
    rows.sort(key=lambda row: row[1], reverse=True)

    return rows[0][1]


if __name__ == '__main__':
    output_dir = sys.argv[1]
    cutoff = .9
    fasta = sys.argv[3]
    zip_file = sys.argv[4]
    json_file = sys.argv[5]
    total_results = {"iptm+ptm": {}}
    results = main(fasta, output_dir, "plain", 100)
    total_results["iptm+ptm"].update(results["iptm+ptm"])
    top_score = get_high_confidence_prediction(results)
    logging.info(f"Got a high confidence score of {top_score} from plain")
    if top_score < cutoff:
        results = main(fasta, output_dir, "dropout_9", 100, True, 9, True)
        total_results["iptm+ptm"].update(results["iptm+ptm"])
        top_score = get_high_confidence_prediction(results)
        logging.info(f"Got a high confidence score of {top_score} from dropout_9")
        if top_score < cutoff:
            results = main(fasta, output_dir, "dropoutv2_21", 100, True, 21, True, "multimer_v2")
            total_results["iptm+ptm"].update(results["iptm+ptm"])
            top_score = get_high_confidence_prediction(results)
            logging.info(f"Got a high confidence score of {top_score} from dropout_v2_21")
            if top_score < cutoff:
                results = main(fasta, output_dir, "dropoutv1_9", 100, True, 9, True, "multimer_v1")
                total_results["iptm+ptm"].update(results["iptm+ptm"])
                top_score = get_high_confidence_prediction(results)
                logging.info(f"Got a high confidence score of {top_score} from dropout_v1_9")
                if top_score < cutoff:
                    results = main(fasta, output_dir, "dropoutv1_21", 100, True, 21, True, "multimer_v1")
                    total_results["iptm+ptm"].update(results["iptm+ptm"])
                    top_score = get_high_confidence_prediction(results)
                    logging.info(f"Got a high confidence score of {top_score} from dropout_v1_21")
                    if top_score < cutoff:
                        results = main(fasta, output_dir, "dropoutv2_9", 100, True, 9, True, "multimer_v2")
                        total_results["iptm+ptm"].update(results["iptm+ptm"])
                        top_score = get_high_confidence_prediction(results)
                        logging.info(f"Got a high confidence score of {top_score} from dropout_v2_9")
    top_score = get_high_confidence_prediction(total_results)
    logging.info(f"Got a final high confidence score of {top_score}")

    with smart_open.open(zip_file, "wb") as zipf:
        with zipfile.ZipFile(zipf, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for model_name, confidence in total_results["iptm+ptm"].items():
                if confidence >= .7:
                    if os.path.exists(os.path.join(output_dir, "example", f"relaxed_{model_name}.pdb")):
                        archive.write(os.path.join(output_dir, "example", f"relaxed_{model_name}.pdb"), f"relaxed_{model_name}.pdb")
                    if os.path.exists(os.path.join(output_dir, "example", f"unrelaxed{model_name}.pdb")):
                        archive.write(os.path.join(output_dir, "example", f"unrelaxed{model_name}.pdb"), f"unrelaxed{model_name}.pdb")

    with smart_open.open(json_file, "w") as f:
        json.dump(total_results, f)
