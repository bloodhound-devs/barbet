# Gabry: Check https://github.com/xinformatics/alphafold/blob/main/Representations_AlphaFold2_v3.ipynb
import typer
from pathlib import Path
import torch

import enum, os, sys, random, shutil
import jax, tqdm
import numpy as np
import haiku as hk

from a2_model import mk_mock_template, predict_structure
from a2_config import jobname, a3m_file, model_runners
import alphafold as a2
import pickle
# from alphafold.common import protein
# from alphafold.data import pipeline
# from alphafold.data import templates
# from alphafold.model import data
# from alphafold.model import config
# from alphafold.model import model
# from alphafold.relax import relax


from gambit.embedding import Embedding
app = typer.Typer()

class AlphaFold2Embedding(Embedding):
    def __init__(self):
        super().__init__()

    def embed(self, seq:str) -> torch.Tensor:
        """ Takes a protein sequence as a string and returns an embedding vector. """
        a3m_lines = ""
        with open(a3m_file,"r") as f:
            a3m_lines = "".join(f.readlines())
        msa, deletion_matrix = a2.data.pipeline.parsers.parse_a3m(a3m_lines)
        query_sequence = msa[0]

        feature_dict = {
            **a2.data.pipeline.make_sequence_features(sequence=query_sequence, description="none", num_res=len(query_sequence)),
            **a2.data.pipeline.make_msa_features(msas=[msa], deletion_matrices=[deletion_matrix]),
            **mk_mock_template(query_sequence)
        }
        plddts, embeddings = predict_structure(jobname, feature_dict, model_runners)
        with open(f"{jobname}_embeddings.pkl", 'wb') as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{jobname}_plddts.pkl", 'wb') as f:
            pickle.dump(plddts, f, protocol=pickle.HIGHEST_PROTOCOL)
        

        # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens, incl. prefix ([0,1:8]) 
        # emb_0 = embedding_repr.last_hidden_state[0] # shape (n, 1024)

        # if you want to derive a single representation (per-protein embedding) for the whole protein
        # vector = emb_0.mean(dim=0) # shape (1024)

        # if torch.isnan(vector).any():
        #     return None

        # return vector        


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


if __name__ == "__main__":
    app()
