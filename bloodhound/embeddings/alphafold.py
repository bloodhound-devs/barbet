from enum import Enum
import typer
from pathlib import Path
import torch
from torchapp.cli import method
from bloodhound.embedding import Embedding
from bloodhound.embeddings.af.run_batch_cleaned import run_batch_colabfold
from bloodhound.embeddings.af.AlphafoldBatchRunner import AlphafoldBatchRunner
import numpy as np


# Alphafold's 5 models are configured in: https://github.com/google-deepmind/alphafold/blob/main/alphafold/model/config.py#L39
# Check the screenshot in this directory for a summary of those models from the paper's supplemental material

SAVE_TO_NPY = True
seq = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK"
out_path = "./output"
queries = [('test_seq', seq, None)]
result_dir = f"{out_path}/batch"
save_single_representations = False # Whether to save each model's output embeddings to a .npy file


af_config = dict(
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
    data_dir=Path('/data/gpfs/projects/punim2199/gambit_data/colabfold'),
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



class AFModels(Enum):
    ALL = "0"
    PTM_1 = "1"
    PTM_2 = "2"
    PTM_3 = "3"
    PTM_4 = "4"
    PTM_5 = "5"

class AFEmbedding(Embedding):
    @method
    def setup(
        self, 
        layers:AFModels=typer.Option(..., help="The number of ESM layers to use."),
        hub_dir:Path=typer.Option(None, help="The torch hub directory where the ESM models will be cached."),
    ):
        self.layers = layers
        assert layers in AFModels, f"Please ensure the number of ESM layers is one of " + ", ".join(AFModels.keys())

        self.hub_dir = hub_dir
        if hub_dir:
            torch.hub.set_dir(str(hub_dir))
        self.model = None
        self.device = None
        self.batch_converter = None
        self.alphabet = None

    def __getstate__(self):
        # Return a dictionary of attributes to be pickled
        state = self.__dict__.copy()
        # Remove the attribute that should not be pickled
        if 'model' in state:
            del state['model']
        if 'batch_converter' in state:
            del state['batch_converter']
        if 'alphabet' in state:
            del state['alphabet']
        if 'device' in state:
            del state['device']
        return state

    def __setstate__(self, state):
        # Restore the object state from the unpickled state
        self.__dict__.update(state)
        self.model = None
        self.device = None
        self.batch_converter = None
        self.alphabet = None

    def load(self):
        self.model, self.alphabet = self.layers.get_model_alphabet()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def embed(self, seq:str) -> torch.Tensor:
        """ Takes a protein sequence as a string and returns an embedding vector. """
        layers = int(self.layers.value)

        if not self.model:
            self.load()

        _, _, batch_tokens = self.batch_converter([("marker_id", seq)])
        batch_tokens = batch_tokens.to(self.device)                
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[layers], return_contacts=True)
        token_representations = results["representations"][layers]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

        vector = sequence_representations[0]
        # if torch.isnan(vector).any():
        #     return None

        return vector       
    

    def test_class():
        af_runner = AlphafoldBatchRunner(**af_config)
        models_embeddings = af_runner.run()
        AFEmbedding.print_test_result(models_embeddings)    

    def test():
        models_embeddings = run_batch_colabfold(af_config)
        AFEmbedding.print_test_result(models_embeddings)

    def print_test_result(models_embeddings):
        comparison_embeddings = [models_embeddings[k] for k in models_embeddings]
        for k1 in models_embeddings:
            embeddings = models_embeddings[k1]['embeddings']
            print(f"\nModel '{k1}'. Embeddings shape: {embeddings.shape}. Exec time: {models_embeddings[k1]['time']:.3f}s")
            for k2 in models_embeddings:
                if k2 == k1: continue
                distance = np.linalg.norm(embeddings - models_embeddings[k2]['embeddings'], axis=1)
                similarity = np.mean(distance)
                print(f"\t - {k1} ~ {k2} embedding similarity: {similarity}")
            