import torch
import numpy as np
from barbet.data import BarbetPredictionDataset
import pytest

@pytest.mark.parametrize("embedding_size,species_count,gene_count,repeats,stack_size", [
    (4, 2, 20, 1, 10),
    (8, 3, 30, 2, 5),
    (2, 1, 10, 1, 5),
    (4, 4, 40, 4, 20),
    (6, 2, 25, 1, 7),  # edge case where gene_count % stack_size ≠ 0
])
def test_prediction_dataloader(embedding_size, species_count, gene_count, repeats, stack_size):
    embedding_size = 4
    species_count = 2
    gene_count = 20
    repeats = 1
    stack_size = 10
    
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

    dataset = BarbetPredictionDataset(array=embeddings, accessions=accessions, stack_size=10, repeats=1)
    assert len(dataset) == len(dataset.stacks) == species_count * gene_count//(repeats * stack_size), "Dataset length mismatch"
    all_indices = np.array([stack.array_indices for stack in dataset.stacks])
    assert all_indices.shape == (species_count * gene_count//(repeats * stack_size), stack_size), "Indices shape mismatch"
    # Get counts of each index
    unique, counts = np.unique(all_indices, return_counts=True)
    if gene_count % stack_size == 0:
        assert np.all(counts == repeats), "Each index should appear exactly 'repeats' times"
    else:
        assert np.all((counts == repeats + 1) | (counts == repeats)), "Each index should appear 'repeats' or 'repeats + 1' times due to uneven division"
    
    batch = dataset[0]
    for batch in dataset:
        assert isinstance(batch, torch.Tensor), "Batch should be a torch tensor"
        assert batch.shape == (stack_size, embedding_size), "Batch shape mismatch"

