from pathlib import Path
from run_batch_cleaned import run_batch_colabfold
import numpy as np

# Alphafold's 5 models are configured in: https://github.com/google-deepmind/alphafold/blob/main/alphafold/model/config.py#L39
# Check the screenshot in this directory for a summary of those models from the paper's supplemental material

SAVE_TO_NPY = True

seq = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK"
out_path = "./output"
queries = [('test_seq', seq, None)]
result_dir = f"{out_path}/batch"
save_single_representations = False # Whether to save each model's output embeddings to a .npy file


config = dict(
    queries=queries,
    result_dir=result_dir,
    use_templates=False,
    custom_template_path=None,
    num_relax=0,
    relax_max_iterations=2000,
    relax_tolerance=2.39,
    relax_stiffness=10.0,
    relax_max_outer_iterations=3,
    msa_mode='single_sequence',
    model_type='auto',
    num_models=5,
    # model_order=['1,2,3,4,5'], # Uncommenting this for some reasons breaks the mean_score calculation
    num_recycles=None,
    recycle_early_stop_tolerance=None,
    num_ensemble=1,
    is_complex=False,
    keep_existing_results=True,
    rank_by='auto',
    pair_mode='unpaired_paired',
    pairing_strategy='greedy',
    data_dir=Path('/data/gpfs/projects/punim2199/rob/localcolabfold/localcolabfold/colabfold'),
    host_url='https://api.colabfold.com',
    user_agent='colabfold/1.5.5 (1ccca5a53d20c909f3ccf8a4b81df804e6717cb1)',
    random_seed=0,
    num_seeds=1,
    stop_at_score=100,
    recompile_padding=10,
    zip_results=False,
    save_single_representations=save_single_representations,
    save_pair_representations=True,
    use_dropout=False,
    max_seq=None,
    max_extra_seq=None,
    max_msa=None,
    pdb_hit_file=None,
    local_pdb_path=None,
    use_cluster_profile=True,
    use_gpu_relax = False,
    jobname_prefix=None,
    save_all=True,
    save_recycles=False,
)

# print(f"Running single colabfold on sequence: {seq}")
# single_embeddings = run_single_colabfold(seq=seq, result_dir=f"{out_path}/single")


models_embeddings = run_batch_colabfold(config)
comparison_embeddings = [models_embeddings[k] for k in models_embeddings]

for k1 in models_embeddings:
    embeddings = models_embeddings[k1]['embeddings']
    print(f"\nModel '{k1}'. Embeddings shape: {embeddings.shape}. Exec time: {models_embeddings[k1]['time']:.3f}s")
    for k2 in models_embeddings:
        if k2 == k1: continue
        distance = np.linalg.norm(embeddings - models_embeddings[k2]['embeddings'], axis=1)
        similarity = np.mean(distance)
        print(f"\t - {k1} ~ {k2} embedding similarity: {similarity}")
    
# TODO concatenate these instead of writing
breakpoint()