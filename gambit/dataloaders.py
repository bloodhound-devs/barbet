import torch
from pathlib import Path
import random
from seqbank import SeqBank
from corgi.seqtree import SeqTree
from corgi.dataloaders import CorgiDataloaders
from fastcore.transform import Transform
from fastai.data.core import TfmdDL
from dataclasses import dataclass
from rich.progress import track

from .embedding import get_key


def get_preprocessed_path(base_dir:Path, name:str) -> Path:
    assert len(name) == len("RS_GCF_000006945.2")
    path = base_dir/name[3:6]/name[7:10]/name[10:13]/name[13:16]/f"{name}.pt"    
    return path


@dataclass
class Item():
    accession:str
    index:int


@dataclass
class AccessionToInput(Transform):
    seqbank:SeqBank

    def encodes(self, accession:str):
        data = self.seqbank[accession]
        array = torch.frombuffer(data, dtype=torch.float32)
        return array,


@dataclass
class AccessionToInputOutput(Transform):
    seqtree:SeqTree
    seqbank:SeqBank

    def encodes(self, accession:str):
        # TODO Refactor to use AccessionToInput
        data = self.seqbank[accession]
        array = torch.frombuffer(data, dtype=torch.float32)
        return array, self.seqtree[accession].node_id


@dataclass
class FileToInput(Transform):
    path:Path
    cache:bool=False
    _data=None

    def data(self):
        self.path = Path(self.path)
        if self._data is not None:
            return self._data
        
        assert self.path.exists()
        data = torch.load(str(self.path))
        
        if self.cache:
            self._data = data

        return data

    def encodes(self, i:int):
        return self.data()[i],


def create_dataloaders(
    seqtree:SeqTree, 
    seqbank:SeqBank, 
    batch_size:int, 
    validation_partition:int,
    max_items:int=0,
) -> CorgiDataloaders:   
    training = []
    validation = []
    for accession, details in seqtree.items():
        partition = details.partition
        dataset = validation if partition == validation_partition else training
        dataset.append( accession )

        if max_items and len(training) >= max_items and len(validation) > 0:
            break

    training_dl = TfmdDL(
        dataset=training,
        batch_size=batch_size, 
        shuffle=True,
        after_item=AccessionToInputOutput(seqbank=seqbank, seqtree=seqtree),
    )   

    validation_dl = TfmdDL(
        dataset=validation,
        batch_size=batch_size, 
        shuffle=False,
        after_item=AccessionToInputOutput(seqbank=seqbank, seqtree=seqtree),
    )

    dls = CorgiDataloaders(training_dl, validation_dl, classification_tree=seqtree.classification_tree)
    return dls


def species_test_dataloader(accession:str, seqbank:SeqBank, batch_size:int) -> TfmdDL:
    prefix = get_key(accession=accession, gene="")
    items = [key for key in seqbank.get_accessions() if key.startswith(prefix)]

    return TfmdDL(
        dataset=items,
        batch_size=batch_size, 
        shuffle=False,
        after_item=AccessionToInput(seqbank=seqbank),
    )


def species_dataloader(embeddings:Path, batch_size:int, cache:bool=True) -> TfmdDL:
    getter = FileToInput(path=embeddings, cache=cache)
    dataset = []
    data = getter.data()
    for i, embedding in enumerate(data):
        if not torch.isnan(embedding).any():
            dataset.append( i )

    return TfmdDL(
        dataset=dataset,
        batch_size=batch_size, 
        shuffle=False,
        after_item=getter,
    )   

