import typer
from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode
from Bio import SeqIO
from gambit.models import get_esm2_model_alphabet
from corgi.seqtree import SeqTree
import random
import csv
from seqbank import SeqBank
from rich.progress import track
import torch

app = typer.Typer()


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

@app.command()
# def preprocess(taxonomy:Path):
def preprocess(
    partitions:int=5,
):
    taxonomy = Path("/home/rturnbull/wytamma/gambit_data/release214/taxonomy/bac120_taxonomy_r214_reps.tsv")
    msa = Path("/home/rturnbull/wytamma/gambit_data/release214/msa/gtdb_r214_bac120.faa")
    marker_info = Path("/home/rturnbull/wytamma/gambit_data/genomic_files_all/bac120_msa_marker_info.tsv")

    # Create root of tree
    lineage_to_node = {}
    root = SoftmaxNode("d__Bacteria")
    lineage_to_node["d__Bacteria"] = root

    # Fill out tree with taxonomy
    accesssion_to_node = {}
    with open(taxonomy) as f:
        for line in f:
            accesssion, lineage = line.split("\t")
            node = get_node(lineage, lineage_to_node)
            accesssion_to_node[accesssion] = node

    # Read marker locations
    current_location = 0
    marker_info_dict = {}
    with open(marker_info) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            id = row["Marker Id"]
            length = int(row['Length (bp)'])
            end = current_location+length

            marker_info_dict[id] = (current_location, end)
            current_location = end

    layers = 6
    model, alphabet = get_esm2_model_alphabet(layers)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    seqtree = SeqTree(classification_tree=root)
    random.seed(42)
    total = sum(1 for _ in SeqIO.parse(msa, "fasta"))
    results_dir = Path(f"esm{layers}")
    results_dir.mkdir(exist_ok=True, parents=True)
    # for record in SeqIO.parse(msa, "fasta"):
    for record in track(SeqIO.parse(msa, "fasta"), total=total):
        name = record.id
        assert len(name) == len("RS_GCF_000006945.2")
        
        path = results_dir/name[3:6]/name[7:10]/name[10:13]/name[13:16]/f"{name}.pt"
        if path.exists():
            continue
        path.parent.mkdir(exist_ok=True, parents=True)
        
        node = accesssion_to_node[record.id]
        partition = random.randint(0,partitions-1)
        seqtree.add(name, node, partition)

        assert len(record.seq) == current_location
        index = 0
        array = None
        for marker_id, (start,end) in marker_info_dict.items():
            seq = str(record.seq[start:end]).replace("-","")

            _, _, batch_tokens = batch_converter([(marker_id, seq)])
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            # Extract per-residue representations (on CPU)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[layers], return_contacts=True)
            token_representations = results["representations"][layers]

            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

            vector = sequence_representations[0]
            if array is None:
                array = torch.zeros((len(marker_info_dict), len(vector)), dtype=vector.dtype)
            array[index] = vector
            index += 1

        torch.save(array.half(), str(path))
        
    seqtree.save("seqtree2D.st")



if __name__ == "__main__":
    app()
