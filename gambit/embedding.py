from pathlib import Path
from abc import ABC, abstractmethod
import typer
from Bio import SeqIO
import random
from rich.progress import track
from seqbank import SeqBank
from hierarchicalsoftmax import SoftmaxNode
from corgi.seqtree import SeqTree
import tarfile
import torch
from io import StringIO

def get_key(accession:str, gene:str) -> str:
    """ Returns the standard format of a key """
    assert len(accession) == len("RS_GCF_000006945.2")
    key = f"{accession[3:6]}/{accession[7:10]}/{accession[10:13]}/{accession}/{gene}"
    return key


def get_node(lineage:str, lineage_to_node:dict[str,SoftmaxNode]) -> SoftmaxNode:
    if lineage in lineage_to_node:
        return lineage_to_node[lineage]

    split_point = lineage.rfind(";")
    parent_lineage = lineage[:split_point]
    name = lineage[split_point+1:]
    parent = get_node(parent_lineage, lineage_to_node)
    node = SoftmaxNode(name=name, parent=parent)
    lineage_to_node[lineage] = node
    return node


class Embedding(ABC):

    @abstractmethod
    def embed(self, seq:str) -> torch.Tensor:
        """ Takes a protein sequence as a string and returns an embedding vector. """
        pass

    def __call__(self, seq:str) -> torch.Tensor:
        """ Takes a protein sequence as a string and returns an embedding vector. """
        return self.embed(seq)

    def build_seqtree(self, taxonomy:Path) -> tuple[SeqTree,dict[str,SoftmaxNode]]:
        # Create root of tree
        lineage_to_node = {}
        root = None

        # Fill out tree with taxonomy
        accession_to_node = {}
        with open(taxonomy) as f:
            for line in f:
                accesssion, lineage = line.split("\t")

                if not root:
                    root_name = lineage.split(";")[0]
                    root = SoftmaxNode(root_name)
                    lineage_to_node[root_name] = root

                node = get_node(lineage, lineage_to_node)
                accession_to_node[accesssion] = node
        
        seqtree = SeqTree(classification_tree=root)
        return seqtree, accession_to_node

    def preprocess(
        self,
        taxonomy:Path,
        marker_genes:Path,
        output_seqtree:Path,
        output_seqbank:Path,
        partitions:int=5,
        seed:int=42,
    ):
        seqtree, accession_to_node = self.build_seqtree(taxonomy)

        seqbank = SeqBank(output_seqbank, write=True)        
        random.seed(seed)

        partitions_dict = {}

        with tarfile.open(marker_genes, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(".faa"):
                    continue

                f = tar.extractfile(member)
                marker_id = Path(member.name.split("_")[-1]).with_suffix("").name
                print(marker_id)

                fasta_io = StringIO(f.read().decode('ascii'))

                total = sum(1 for _ in SeqIO.parse(fasta_io, "fasta"))
                fasta_io.seek(0)
        
                for record in track(SeqIO.parse(fasta_io, "fasta"), total=total):
                    species_accession = record.id
                    
                    node = accession_to_node[species_accession]
                    partition_key = f"{node}|{marker_id}"
                    if partition_key not in partitions_dict:
                        partitions_dict[partition_key] = random.randint(0,partitions-1)
                    
                    partition = partitions_dict[partition_key]
                    key = get_key(species_accession, marker_id)

                    try:
                        if key not in seqbank:
                            seq = str(record.seq).replace("-","").replace("*","")
                            vector = self(seq)
                            if vector is not None and not torch.isnan(vector).any():
                                seqbank.add(
                                    seq=vector.cpu().detach().clone().numpy().tobytes(),
                                    accession=key,
                                )
                                seqtree.add(key, node, partition)
                                
                            del vector
                    except Exception as err:
                        print(f"ERROR for {key} ({len(seq)}): {err}")
        
        seqtree.save(output_seqtree)        
