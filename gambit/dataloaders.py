import torch
from pathlib import Path
import random
from corgi.seqtree import SeqTree
from corgi.dataloaders import CorgiDataloaders
from fastcore.transform import Transform
from fastai.data.core import TfmdDL
from dataclasses import dataclass
from rich.progress import track


def get_preprocessed_path(base_dir:Path, name:str) -> Path:
    assert len(name) == len("RS_GCF_000006945.2")
    path = base_dir/name[3:6]/name[7:10]/name[10:13]/name[13:16]/f"{name}.pt"    
    return path


@dataclass
class Item():
    accession:str
    index:int


@dataclass
class AccessionToInputOutput(Transform):
    seqtree:SeqTree
    base_dir:Path

    def encodes(self, item:Item):
        accession = item.accession
        i = item.index
        path = get_preprocessed_path(self.base_dir, accession)
        assert path.exists()
        data = torch.load(str(path))

        return data[i], self.seqtree[accession].node_id


def create_dataloaders(
    seqtree:SeqTree, 
    base_dir:Path, 
    batch_size:int, 
    validation_partition:int,
    max_items:int=0,
) -> CorgiDataloaders:   
    
    def check_file(accession, details):
        path = get_preprocessed_path(base_dir, accession)
        assert path.exists()
        data = torch.load(str(path))
        dataset = []
        for i, embedding in enumerate(data):
            if not torch.isnan(embedding).any():
                dataset.append( (accession,i, details.partition))

        return dataset

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=-1)(delayed(check_file)(accession, details) for accession, details in track(seqtree.items()))
    training = []
    validation = []
    for result in results:
        for accession,index,partition in result:
            dataset = validation if partition == validation_partition else training
            item = Item(accession=accession, index=index)
            dataset.append( item )

        if max_items and len(training) >= max_items and len(validation) > 0:
            break

    training_dl = TfmdDL(
        dataset=training,
        batch_size=batch_size, 
        shuffle=True,
        after_item=AccessionToInputOutput(base_dir=base_dir, seqtree=seqtree),
    )   

    validation_dl = TfmdDL(
        dataset=validation,
        batch_size=batch_size, 
        shuffle=False,
        after_item=AccessionToInputOutput(base_dir=base_dir, seqtree=seqtree),
    )   

    dls = CorgiDataloaders(training_dl, validation_dl, classification_tree=seqtree.classification_tree)
    return dls