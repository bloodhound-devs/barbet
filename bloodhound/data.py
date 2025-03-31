import random
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import lightning as L
from dataclasses import dataclass
from corgi.seqtree import SeqTree


RANKS = ["phylum", "class", "order", "family", "genus", "species"]

def read_memmap(path, count, dtype:str="float16") -> np.memmap:
    file_size = os.path.getsize(path)
    dtype_size = np.dtype(dtype).itemsize
    num_elements = file_size // dtype_size
    embedding_size = num_elements // count
    shape = (count, embedding_size)
    return np.memmap(path, dtype=dtype, mode='r', shape=shape)


def gene_id_from_accession(accession:str):
    return accession.split("/")[-1]


def choose_k_from_n(lst, k) -> list:
    n = len(lst)
    if n == 0:
        return []
    repetitions = k // n
    remainder = k % n
    result = lst * repetitions + random.sample(lst, remainder)
    return result


@dataclass(kw_only=True)
class BloodhoundDataset(Dataset):
    accessions: list[str]
    array:np.memmap|np.ndarray
    gene_id_dict: dict[str, int]
    accession_to_array_index:dict[str,int]|None=None

    def __len__(self):
        return len(self.accessions)

    def __getitem__(self, idx):
        accession = self.accessions[idx]
        array_index = self.accession_to_array_index[accession] if self.accession_to_array_index else idx
        embedding = torch.as_tensor(np.array(self.array[array_index,:], copy=False), dtype=torch.float16)
        gene_id = gene_id_from_accession(accession)
        return embedding, self.gene_id_dict[gene_id]


@dataclass(kw_only=True)
class BloodhoundTrainingDataset(Dataset):
    accessions: list[str]
    seqtree: SeqTree
    array:np.memmap|np.ndarray
    gene_id_dict: dict[str, int]
    accession_to_array_index:dict[str,int]|None=None
    seq_count:int = 0

    def __len__(self):
        return len(self.accessions)

    def __getitem__(self, idx):
        accession = self.accessions[idx]
        array_indices = self.accession_to_array_index[accession] if self.accession_to_array_index else idx
        if self.seq_count:
            array_indices = choose_k_from_n(array_indices, self.seq_count)

        assert len(array_indices) > 0, f"Accession {accession} has no array indices"
        with torch.no_grad():
            data = np.array(self.array[array_indices, :], copy=False)
            embedding = torch.tensor(data, dtype=torch.float16)
            del data
    
        # gene_id = gene_id_from_accession(accession)
        seq_detail = self.seqtree[accession]
        node_id = int(seq_detail.node_id)
        del seq_detail
        
        # return embedding, self.gene_id_dict[gene_id], self.seqtree[self.accessions[0]].node_id # hack
        return embedding, node_id


@dataclass(kw_only=True)
class BloodhoundPredictionDataset(Dataset):
    embeddings: list[torch.Tensor]
    gene_family_ids: list[int]

    def __post_init__(self):
        assert len(self.embeddings) == len(self.gene_family_ids)

    def __len__(self):
        return len(self.gene_family_ids)

    def __getitem__(self, idx):
        return self.embeddings[idx] #, self.gene_family_ids[idx]
    

@dataclass
class BloodhoundDataModule(L.LightningDataModule):
    seqtree: SeqTree
    # seqbank: SeqBank
    array:np.memmap|np.ndarray
    accession_to_array_index:dict[str,int]
    gene_id_dict: dict[str,int]
    max_items: int = 0
    batch_size: int = 16
    num_workers: int = 0
    validation_partition:int = 0
    test_partition:int = -1
    train_all:bool = False

    def __init__(
        self,
        seqtree: SeqTree,
        array:np.memmap|np.ndarray,
        accession_to_array_index:dict[str,list[int]],
        gene_id_dict: dict[str,int],
        max_items: int = 0,
        batch_size: int = 16,
        num_workers: int = None,
        validation_partition:int = 0,
        test_partition:int=-1,
        seq_count:int=0,
        train_all:bool=False,
    ):
        super().__init__()
        self.array = array
        self.accession_to_array_index = accession_to_array_index
        self.seqtree = seqtree
        self.gene_id_dict = gene_id_dict
        self.max_items = max_items
        self.batch_size = batch_size
        self.validation_partition = validation_partition
        self.test_partition = test_partition
        self.num_workers = min(os.cpu_count(), 8) if num_workers is None else num_workers
        self.seq_count = seq_count
        self.train_all = train_all

    def setup(self, stage=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.training = []
        self.validation = []

        for accession, details in self.seqtree.items():
            partition = details.partition
            if partition == self.test_partition:
                continue

            dataset = self.validation if partition == self.validation_partition else self.training
            dataset.append( accession )

            if self.max_items and len(self.training) >= self.max_items and len(self.validation) > 0:
                break

        if self.train_all:
            self.training += self.validation

        self.train_dataset = self.create_dataset(self.training)
        self.val_dataset = self.create_dataset(self.validation)

    def create_dataset(self, accessions:list[str]) -> BloodhoundTrainingDataset:
        return BloodhoundTrainingDataset(
            accessions=accessions, 
            seqtree=self.seqtree, 
            array=self.array,
            accession_to_array_index=self.accession_to_array_index,
            gene_id_dict=self.gene_id_dict,
            seq_count=self.seq_count,
        )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

