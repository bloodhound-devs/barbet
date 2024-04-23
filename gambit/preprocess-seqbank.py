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

    _, alphabet = get_esm2_model_alphabet(6)
    seqtree = SeqTree(classification_tree=root)
    random.seed(42)
    seqbank = SeqBank("seqbank.sb", write=True)
    total = sum(1 for _ in SeqIO.parse(msa, "fasta"))
    for record in track(SeqIO.parse(msa, "fasta"), total=total):
        print(record.id)
        node = accesssion_to_node[record.id]
        partition = random.randint(0,partitions-1)
        assert len(record.seq) == current_location
        for marker_id, (start,end) in marker_info_dict.items():
            name = f"{record.id}__{marker_id}"
            seq = str(record.seq[start:end]).replace("-","")
            
            if len(seq):
                seqtree.add(name, node, partition)
                seqbank.add(
                    seq=bytes(alphabet.encode(seq)),
                    accession=name,
                )
    seqtree.save("seqtree.st")


if __name__ == "__main__":
    app()
