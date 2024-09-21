import torch
from pathlib import Path
from seqbank import SeqBank
from corgi.seqtree import SeqTree
from fastai.data.core import TfmdDL, DataLoaders
from fastcore.transform import Transform
from dataclasses import dataclass
from rich.progress import track
from hierarchicalsoftmax import SoftmaxNode

from .embedding import Embedding


def get_preprocessed_path(base_dir:Path, name:str) -> Path:
    assert len(name) == len("RS_GCF_000006945.2")
    path = base_dir/name[3:6]/name[7:10]/name[10:13]/name[13:16]/f"{name}.pt"    
    return path


@dataclass
class Item():
    accession:str
    index:int


def gene_id_from_accession(accession:str):
    return accession.split("/")[-1]

@dataclass
class AccessionToInput(Transform):
    seqbank:SeqBank
    gene_id_dict:dict

    def encodes(self, accession:str):
        data = self.seqbank[accession]
        array = torch.frombuffer(data, dtype=torch.float32)
        del data
        gene_id = gene_id_from_accession(accession)
        return array, self.gene_id_dict[gene_id]


@dataclass
class AccessionToInputOutput(AccessionToInput):
    seqtree:SeqTree

    def encodes(self, accession:str):
        inputs = super().encodes(accession)
        return *inputs, self.seqtree[accession].node_id


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


class BloodhoundDataloaders(DataLoaders):
    def __init__(
        self, 
        *loaders, # `DataLoader` objects to wrap
        path:str='.', # Path to store export objects
        device=None, # Device to put `DataLoaders`
        classification_tree:SoftmaxNode=None,
        embedding:Embedding=None,
        gene_id_dict:dict[str:int]=None
    ):
        super().__init__(*loaders, path=path, device=device)
        self.classification_tree = classification_tree
        self.embedding = embedding
        self.gene_id_dict = gene_id_dict

    def new_empty(self):
        loaders = [dl.new([]) for dl in self.loaders]
        return type(self)(*loaders, path=self.path, device=self.device, classification_tree=self.classification_tree, embedding=self.embedding, gene_id_dict=self.gene_id_dict)


def create_dataloaders(
    seqtree:SeqTree, 
    seqbank:SeqBank, 
    batch_size:int, 
    validation_partition:int,
    embedding:Embedding,
    max_items:int=0,
) -> BloodhoundDataloaders:   
    training = []
    validation = []
    family_ids = set()

    for accession, details in seqtree.items():
        partition = details.partition
        dataset = validation if partition == validation_partition else training
        dataset.append( accession )

        gene_id = accession.split("/")[-1]
        family_ids.add(gene_id)

        if max_items and len(training) >= max_items and len(validation) > 0:
            break

    gene_id_dict = {family_id:index for index, family_id in enumerate(sorted(family_ids))}

    training_dl = TfmdDL(
        dataset=training,
        batch_size=batch_size, 
        shuffle=True,
        after_item=AccessionToInputOutput(seqbank=seqbank, seqtree=seqtree, gene_id_dict=gene_id_dict),
        n_imp=2,
    )   

    validation_dl = TfmdDL(
        dataset=validation,
        batch_size=batch_size, 
        shuffle=False,
        after_item=AccessionToInputOutput(seqbank=seqbank, seqtree=seqtree, gene_id_dict=gene_id_dict),
        n_imp=2,
    )

    dls = BloodhoundDataloaders(
        training_dl, 
        validation_dl, 
        classification_tree=seqtree.classification_tree, 
        gene_id_dict=gene_id_dict, 
        embedding=embedding,
    )
    return dls


def seqbank_dataloader(seqbank:SeqBank, items:list[str], batch_size:int) -> TfmdDL:
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

