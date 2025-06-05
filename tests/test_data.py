import torch
import numpy as np
from barbet.data import BarbetPredictionDataset
import pytest

def test_prediction_dataloader():
    embedding_size = 4
    species_count = 2
    gene_count = 20
    repeats = 1
    seq_count = 10
    
    x = np.arange(gene_count)
    base_embeddings = np.repeat(x[:, np.newaxis], embedding_size, axis=1)
    embeddings = []
    accessions = []
    for i in range(species_count):
        for j in range(gene_count):
            # Create a unique accession for each gene in each species
            accessions.append(f"species_{i}/gene_{j}")
        embeddings.append(base_embeddings + i * 10)

    embeddings = np.vstack(embeddings)
    assert embeddings.shape == (species_count * gene_count, embedding_size), "Embeddings shape mismatch"
    assert len(accessions) == species_count * gene_count, "Accessions count mismatch"

    dataset = BarbetPredictionDataset(array=embeddings, accessions=accessions, seq_count=10, repeats=1)
    assert len(dataset) == len(dataset.stacks) == species_count * gene_count//(repeats * seq_count), "Dataset length mismatch"
    all_indices = np.array([stack.array_indices for stack in dataset.stacks])
    assert all_indices.shape == (species_count * gene_count//(repeats * seq_count), seq_count), "Indices shape mismatch"
    # Get counts of each index
    unique, counts = np.unique(all_indices, return_counts=True)
    if gene_count % seq_count == 0:
        assert np.all(counts == repeats), "Each index should appear exactly 'repeats' times"
    else:
        assert np.all((counts == repeats + 1) | (counts == repeats)), "Each index should appear 'repeats' or 'repeats + 1' times due to uneven division"
    
    batch = dataset[0]
    for batch in dataset:
        assert isinstance(batch, torch.Tensor), "Batch should be a torch tensor"
        assert batch.shape == (seq_count, embedding_size), "Batch shape mismatch"

