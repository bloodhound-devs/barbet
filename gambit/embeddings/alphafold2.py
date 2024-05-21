# Gabry: This is from https://www.kaggle.com/code/alexandervc/alphafold2predictstructure and from the AlphaFold notebook in alphafold's github
# Gabry: Check https://github.com/xinformatics/alphafold/blob/main/Representations_AlphaFold2_v3.ipynb

import typer
from pathlib import Path
import torch

import enum, os, sys, random, shutil
import jax, tqdm
import numpy as np
import haiku as hk
import time
import pickle
import re
import hashlib

from alphafold.model import config as a2config
from alphafold.model import data as a2data
from alphafold.model import model as a2model
from alphafold.model import modules as a2modules
from alphafold.model import modules_multimer as a2modules_multimer
from alphafold.common import protein as a2protein
from alphafold.data import pipeline as a2pipeline
from alphafold.notebooks import notebook_utils
import alphafold as a2

from absl import logging
logging.set_verbosity(logging.INFO)

from gambit.embedding import Embedding
app = typer.Typer()

params_dir = '/data/scratch/datasets/alphafold/v2.3.2' # This directory should contain a 'params' directory
out_dir = './gambit/embeddings/af2_out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
available_models = ["model_1","model_2","model_3","model_4","model_5"]


protein_test_1 = 'MLILISPAKTLDYQSPLTTTRYTLPELLDNSQQLIHEARKLTPPQISTLMRISDKLAGINAARFHDWQPDFTPANARQAILAFKGDVYTGLQAETFSEDDFDFAQQHLRMLSGLYGVLRPLDLMQPYRLEMGIRLENARGKDLYQFWGDIITNKLNEALAAQGDNVVINLASDEYFKSVKPKKLNAEIIKPVFLDEKNGKFKIISFYAKKARGLMSRFIIENRLTKPEQLTGFNSEGYFFDEDSSSNGELVFKRYEQR' #@param {type:"string"}

protein_test_2 = 'MTVKTEAAKGTLTYSRMRGMVAILIAFMKQRRMGLNDFIQKIANNSYACKHPEVQSILKISQPQEPELMNANPSPPPSPSQQINLGPSSNPHAKPSDFHFLKVIGKGSFGKVLLARHKAEEVFYAVKVLQKKAILKKKEEKHIMSERNVLLKNVKHPFLVGLHFSFQTADKLYFVLDYINGGELFYHLQRERCFLEPRARFYAAEIASALGYLHSLNIVYRDLKPENILLDSQGHIVLTDFGLCKENIEHNSTTSTFCGTPEYLAPEVLHKQPYDRTVDWWCLGAVLYEMLYGLPPFYSRNTAEMYDNILNKPLQLKPNITNSARHLLEGLLQKDRTKRLGAKDDFMEIKSHVFFSLINWDDLINKKITPPFNPNVSGPNDLRHFDPEFTEEPVPNSIGKSPDSVLVTASVKEAAEAFLGFSYAPPTDSFL'



def add_hash(x,y):
  return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]

@enum.unique
class ModelType(enum.Enum):
  MONOMER = 0
  MULTIMER = 1

# Override of AlphaFold's RunModel class, to overwrite the return_representations parameter
class RunRepresentationsModel(a2model.RunModel):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.multimer_mode = False # Default to Monomer mode
    if self.multimer_mode:
      def _forward_fn(batch):
        model = a2modules_multimer.AlphaFold(self.config.model)
        return model(
            batch,
            is_training=False,
            return_representations=True)
    else:
      def _forward_fn(batch):
        model = a2modules.AlphaFold(self.config.model)
        # Here we call the AlphaFold model with the extra `return_representations` param set to True
        return model(
            batch,
            is_training=False,
            compute_loss=False,
            ensemble_representations=True,
            return_representations=True)

    self.apply = jax.jit(hk.transform(_forward_fn).apply)
    self.init = jax.jit(hk.transform(_forward_fn).init)



class AlphaFold2Embedding(Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # AlphaFold2 offer 5 models, each model can be monomer or multimer. Monomer is the default one.
        # The model parameter files are in the params folder under `/data/scratch/datasets/alphafold/v2.3.2/params` on Spartan.
        # Each model consists of 4 parameter files where <N> is 1,2,3,4 or 5:
        #  
        # - params_model_<N>.npz              # Monomer 
        # - params_model_<N>_ptm.npz          # Monomer
        # - params_model_<N>_multimer_v2.npz  # Multimer v2 (I guess from alphafold v2.2.0)
        # - params_model_<N>_multimer_v3.npz  # Multimer v2 (I guess from alphafold v2.3.2)

        self.num_models = 1 #@param [1,2,3,4,5] {type:"raw"} number of models to use
        self.model_runners = {} # collect model weights
        self.model_params = {}
        for model_name in available_models[:self.num_models]:
            if model_name not in self.model_params:
                self.model_params[model_name] = a2.model.data.get_model_haiku_params(model_name=model_name+"_ptm", data_dir=params_dir)
                if model_name == "model_1":
                    model_config = a2.model.config.model_config(model_name+"_ptm") # is '_ptm' even needed?
                    model_config.data.eval.num_ensemble = 1
                if model_name == "model_3":
                    model_config = a2.model.config.model_config(model_name+"_ptm") # is '_ptm' even needed?
                    model_config.data.eval.num_ensemble = 1
                self.model_runners[model_name] = RunRepresentationsModel(model_config, self.model_params[model_name])
        self.query_sequence = None
        self.jobname = "NONE_JOB"
        self.a3m_file = ""


    def prepare_a3mfile(self, seq: str):
        protein_seq = seq
        if protein_seq is None:
            protein_seq = protein_test_2
            # remove whitespaces and force uppercase
            protein_seq = "".join(protein_seq.split())
            protein_seq = re.sub(r'[^a-zA-Z]','', protein_seq).upper()

        self.query_sequence = protein_seq

        self.jobname = 'gambit_AF2' #@param {type:"string"}
        self.jobname = "".join(self.jobname.split())
        self.jobname = re.sub(r'\W+', '', self.jobname)
        self.jobname = add_hash(self.jobname, self.query_sequence)

        # Create fasta file from the sequence
        with open(f"{out_dir}/{self.jobname}.fasta", "w") as text_file:
            text_file.write(">1\n%s" % self.query_sequence)

        # jackhmmer is the default approach from Deepmind
        msa_mode = "single_sequence" #@param ["jackhmmer", "MMseqs2 (UniRef+Environmental)", "MMseqs2 (UniRef only)","single_sequence","custom"]
        use_msa = True if msa_mode.startswith("MMseqs2") else False
        use_env = True if msa_mode == "MMseqs2 (UniRef+Environmental)" else False
        use_custom_msa = True if msa_mode == "custom" else False
        use_amber = False #@param {type:"boolean"}
        use_templates = True #@param {type:"boolean"}
        homooligomer = 1 #@param [1,2,3,4,5,6,7,8] {type:"raw"}

        # decide which a3m to use
        self.a3m_file = 'none.a3m'
        if use_msa:
            self.a3m_file = f"{self.jobname}.a3m"
        elif use_custom_msa:
            self.a3m_file = f"{self.jobname}.custom.a3m"
            if not os.path.isfile(self.a3m_file):
                # custom_msa_dict = files.upload() # this is from google.colab: `from google.colab import files`
                # No idea I'm just making this up as a placeholder, in case we'll need it in the future
                with open(f"{out_dir}/custom_msa.json", 'r') as f:
                    custom_msa_dict = f.read()
                custom_msa = list(custom_msa_dict.keys())[0]
                header = 0
                import fileinput
                for line in fileinput.FileInput(custom_msa,inplace=1):
                    if line.startswith(">"):
                        header = header + 1 
                    if line.startswith("#"):
                        continue
                    if line.rstrip() == False:
                        continue
                    if line.startswith(">") == False and header == 1:
                        self.query_sequence = line.rstrip() 
                    print(line, end='')
                os.rename(custom_msa, self.a3m_file)
                print(f"moving {custom_msa} to {self.a3m_file}")
        else:
            self.a3m_file = f"{self.jobname}.single_sequence.a3m"
            with open(f"{out_dir}/{self.a3m_file}", "w") as text_file:
                text_file.write(">1\n%s" % self.query_sequence)

    def embed(self, seq:str) -> torch.Tensor:
        """ Takes a protein sequence as a string and returns an embedding vector. """
        
        self.prepare_a3mfile(seq)
        a3m_lines = ""
        with open(f"{out_dir}/{self.a3m_file}","r") as f:
            a3m_lines = "".join(f.readlines())
        # This used to be a tuple (msa_sequences, deletion_matrix, descr) but it is not a Msa class object with the fields: sequences, deletion_matrix and description
        parsed_msa = a2pipeline.parsers.parse_a3m(a3m_lines)
        msa = parsed_msa.sequences
        deletion_matrix = parsed_msa.deletion_matrix

        sequence = msa[0] # Since we only have one sequence in this test example
        single_chain_msas = [parsed_msa]

        # This is from ColabFold
        # Turn the raw data into model features.
        feature_dict = {}    
        feature_dict.update(a2pipeline.make_sequence_features(sequence=sequence, description='query', num_res=len(sequence)))
        feature_dict.update(a2pipeline.make_msa_features(msas=single_chain_msas))
        # We don't use templates in AlphaFold Colab notebook, add only empty placeholder features.
        feature_dict.update(notebook_utils.empty_placeholder_template_features(num_templates=0, num_res=len(sequence)))

        # This is another way of preparing the feature dictionary, however I'm not sure how the template was created and why it's been created that way
        # feature_dict = {
        #     **a2pipeline.make_sequence_features(sequence=self.query_sequence, description="none", num_res=len(self.query_sequence)),
        #     # **a2pipeline.make_msa_features(msas=single_chain_msas, deletion_matrices=[deletion_matrix]), # As said above, the deletion matrices are now included in the parsed_msa object of type parsers.Msa
        #     **a2pipeline.make_msa_features(msas=[parsed_msa]),
        #     **mk_mock_template(msa)
        # }

        plddts, embeddings = self.predict_structure(self.jobname, feature_dict, self.model_runners, random_seed=random.randrange(sys.maxsize))
        with open(f"{self.jobname}_embeddings.pkl", 'wb') as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{self.jobname}_plddts.pkl", 'wb') as f:
            pickle.dump(plddts, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return None

        # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens, incl. prefix ([0,1:8]) 
        # emb_0 = embedding_repr.last_hidden_state[0] # shape (n, 1024)

        # if you want to derive a single representation (per-protein embedding) for the whole protein
        # vector = emb_0.mean(dim=0) # shape (1024)

        # if torch.isnan(vector).any():
        #     return None

        # return vector      
    

    def predict_structure(self, prefix, feature_dict, model_runners, random_seed=0, do_relax=True):  
        """ This is the core function to run the model and get the predictions  """
        """ Predicts structure using AlphaFold for the given sequence. """

        plddts = {} # Predicted lDDT
        embeddings = {} # Predicted embeddings, returned thanks to return_embeddings=True
        
        for model_name, model_runner in model_runners.items():
            processed_feature_dict = model_runner.process_features(feature_dict, random_seed=0)
            prediction_result = model_runner.predict(processed_feature_dict, random_seed=random_seed)
            if do_relax:
                unrelaxed_protein = a2protein.from_prediction(processed_feature_dict,prediction_result)
                unrelaxed_pdb_path = f'{prefix}_unrelaxed_{model_name}.pdb'
                with open(f"{out_dir}/{unrelaxed_pdb_path}", 'w') as f:
                  f.write(a2.common.protein.to_pdb(unrelaxed_protein))
            plddts[model_name] = prediction_result['plddt']
            embeddings[model_name] = prediction_result['representations']

            print(f"{model_name} {plddts[model_name].mean()}")

            if do_relax:
              # Relax the prediction.
              amber_relaxer = a2.relax.relax.AmberRelaxation(max_iterations=0, tolerance=2.39, stiffness=10.0,exclude_residues=[], max_outer_iterations=20)      
              relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
              relaxed_pdb_path = f'{prefix}_relaxed_{model_name}.pdb'
              with open(f"{out_dir}/{relaxed_pdb_path}", 'w') as f:
                f.write(relaxed_pdb_str)

        return plddts, embeddings  

    
    # This is a mock template I found in one of the colab notebooks. Colabfold does not use it and I'm not sure how we could create a template of whether it is even needed
    # def mk_mock_template(self):
    #   # since alphafold's model requires a template input
    #   # we create a blank example w/ zero input, confidence -1
    #   ln = len(self.query_sequence)
    #   output_templates_sequence = "-"*ln
    #   output_confidence_scores = np.full(ln,-1)
    #   templates_all_atom_positions = np.zeros((ln, a2.data.templates.residue_constants.atom_type_num, 3))
    #   templates_all_atom_masks = np.zeros((ln, a2.data.templates.residue_constants.atom_type_num))
    #   templates_aatype = a2.data.templates.residue_constants.sequence_to_onehot(output_templates_sequence, a2.data.templates.residue_constants.HHBLITS_AA_TO_ID)
    #   template_features = {'template_all_atom_positions': templates_all_atom_positions[None],
    #                        'template_all_atom_masks': templates_all_atom_masks[None],
    #                        'template_sequence': [f'none'.encode()],
    #                        'template_aatype': np.array(templates_aatype)[None],
    #                        'template_confidence_scores': output_confidence_scores[None],
    #                        'template_domain_names': [f'none'.encode()],
    #                        'template_release_date': [f'none'.encode()]}
    #   return template_features
      


@app.command()
def main(
    taxonomy:Path,
    marker_genes:Path,
    output_seqtree:Path,
    output_seqbank:Path,
    partitions:int=5,
    seed:int=42,
):
    model = AlphaFold2Embedding()
    model.preprocess(
        taxonomy=taxonomy,
        marker_genes=marker_genes,
        output_seqtree=output_seqtree,
        output_seqbank=output_seqbank,
        partitions=partitions,
        seed=seed,
    )

def test():
   a2_embed_model = AlphaFold2Embedding()
   a2_embed_model.embed(protein_test_2)

if __name__ == "__main__":
    app()
