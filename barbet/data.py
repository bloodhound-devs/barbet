import random
import os
import numpy as np
from collections import defaultdict
from typing import Iterable
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import lightning as L
from dataclasses import dataclass, field
from hierarchicalsoftmax import TreeDict


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


def choose_k_from_n(lst, k) -> list[int]:
    n = len(lst)
    if n == 0:
        return []
    repetitions = k // n
    remainder = k % n
    result = lst * repetitions + random.sample(lst, remainder)
    return result


@dataclass(kw_only=True)
class BarbetStack():
    genome:str
    array_indices:np.array

    def __post_init__(self):
        assert self.array_indices.ndim == 1, "Stack indices must be a 1D array"


@dataclass(kw_only=True)
class BarbetPredictionDataset(Dataset):
    array:np.memmap|np.ndarray
    accessions: list[str]
    stack_size:int
    repeats:int = 2
    seed:int = 42
    stacks: list[BarbetStack] = field(init=False)
    genome_filter:set[str]|None = None

    def __post_init__(self):
        genome_to_array_indices = defaultdict(set)
        for index, accession in enumerate(self.accessions):
            slash_position = accession.rfind("/")
            assert slash_position != -1
            genome = accession[:slash_position]
            if self.genome_filter and genome not in self.genome_filter:
                continue
            genome_to_array_indices[genome].add(index)

        # Build stacks
        random.seed(self.seed)
        self.stacks = []
        for genome, genome_array_indices in genome_to_array_indices.items():
            stack_indices = []
            remainder = []
            for repeat_index in range(self.repeats + 1):
                if len(remainder) == 0 and repeat_index >= self.repeats:
                    break

                # Finish Remainder
                genome_array_indices_set = set(genome_array_indices)
                available = genome_array_indices_set - set(remainder)
                needed = self.stack_size - len(remainder)
                available_list = list(available)

                if len(available_list) >= needed:
                    to_add = random.sample(available_list, needed)  # without replacement
                else:
                    to_add = random.choices(available_list, k=needed)  # with replacement
                to_add_set = set(to_add)
                assert not set(remainder) & to_add_set, "remainder and to_add should be disjoint"

                self.add_stack(genome, remainder + to_add)
                remainder = list(genome_array_indices_set - to_add_set)
                random.shuffle(remainder)

                # If we have already added each item the required number of times, then stop
                if repeat_index >= self.repeats:
                    break

                while len(remainder) >= self.stack_size:
                    self.add_stack(genome, remainder[:self.stack_size])
                    remainder = remainder[self.stack_size:]
                            
    def add_stack(self, genome:str, indices:Iterable[int]) -> BarbetStack:
        """
        Add a new stack to the dataset.
        """
        indices = np.array(sorted(indices))
        stack = BarbetStack(genome=genome, array_indices=indices)
        self.stacks.append(stack)
        return stack

    def __len__(self):
        return len(self.stacks)

    def __getitem__(self, idx):
        stack = self.stacks[idx]
        array_indices = stack.array_indices

        assert len(array_indices) > 0, f"Stack has no array indices"
        with torch.no_grad():
            # data = np.array(self.array[array_indices, :], copy=False)
            # embeddings = torch.tensor(data, dtype=torch.float16)
            data = np.asarray(self.array[array_indices, :]).copy()
            embeddings = torch.from_numpy(data).to(torch.float16)

            del data
        
        return embeddings


@dataclass(kw_only=True)
class BarbetTrainingDataset(Dataset):
    accessions: list[str]
    treedict: TreeDict
    array:np.memmap|np.ndarray
    accession_to_array_index:dict[str,int]|None=None
    stack_size:int = 0

    def __len__(self):
        return len(self.accessions)

    def __getitem__(self, idx):
        accession = self.accessions[idx]
        array_indices = self.accession_to_array_index[accession] if self.accession_to_array_index else idx
        if self.stack_size:
            array_indices = choose_k_from_n(array_indices, self.stack_size)

        assert len(array_indices) > 0, f"Accession {accession} has no array indices"
        with torch.no_grad():
            data = np.array(self.array[array_indices, :], copy=False)
            embedding = torch.tensor(data, dtype=torch.float16)
            del data
    
        # gene_id = gene_id_from_accession(accession)
        seq_detail = self.treedict[accession]
        node_id = int(seq_detail.node_id)
        del seq_detail
        
        return embedding, node_id
    

@dataclass
class BarbetDataModule(L.LightningDataModule):
    treedict: TreeDict
    # seqbank: SeqBank
    array:np.memmap|np.ndarray
    accession_to_array_index:dict[str,int]
    max_items: int = 0
    batch_size: int = 16
    num_workers: int = 0
    validation_partition:int = 0
    test_partition:int = -1
    train_all:bool = False

    def __init__(
        self,
        treedict: TreeDict,
        array:np.memmap|np.ndarray,
        accession_to_array_index:dict[str,list[int]],
        max_items: int = 0,
        batch_size: int = 16,
        num_workers: int = None,
        validation_partition:int = 0,
        test_partition:int=-1,
        stack_size:int=0,
        train_all:bool=False,
    ):
        super().__init__()
        self.array = array
        self.accession_to_array_index = accession_to_array_index
        self.treedict = treedict
        self.max_items = max_items
        self.batch_size = batch_size
        self.validation_partition = validation_partition
        self.test_partition = test_partition
        self.num_workers = min(os.cpu_count(), 8) if num_workers is None else num_workers
        self.stack_size = stack_size
        self.train_all = train_all

    def setup(self, stage=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.training = []
        self.validation = []

        for accession, details in self.treedict.items():
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

    def create_dataset(self, accessions:list[str]) -> BarbetTrainingDataset:
        return BarbetTrainingDataset(
            accessions=accessions, 
            treedict=self.treedict, 
            array=self.array,
            accession_to_array_index=self.accession_to_array_index,
            stack_size=self.stack_size,
        )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

