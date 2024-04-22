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

        # i = random.randint(0,len(data)-1)
        return data[i], self.seqtree[accession].node_id


# def create_dataloader(
#     seqtree:SeqTree,
#     base_dir:Path, 
#     batch_size:int, 
#     validation_partition:int, 
#     max_seqs: int = 0,
# ) -> TfmdDL:
#     accessions = []
#     for accession, details in seqtree.items():
#         if details.partition == validation_partition:
#             accessions.append(accession)

#     if max_seqs:
#         accessions = accessions[:max_seqs]

#     return TfmdDL(
#         dataset=accessions,
#         batch_size=batch_size, 
#         shuffle=False,
#         after_item=AccessionToInputOutput(base_dir=base_dir, seqtree=seqtree),
#     )   


def create_dataloaders(
    seqtree:SeqTree, 
    base_dir:Path, 
    batch_size:int, 
    validation_partition:int,
    max_items:int=0,
) -> CorgiDataloaders:   
    
    training = []
    validation = []
    for accession, details in track(seqtree.items()):
        dataset = validation if details.partition == validation_partition else training
        path = get_preprocessed_path(base_dir, accession)
        assert path.exists()
        data = torch.load(str(path))
        for i, embedding in enumerate(data):
            if not torch.isnan(embedding).any():
                item = Item(accession=accession, index=i)
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