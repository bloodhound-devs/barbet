import typer
from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode
from Bio import SeqIO
from corgi.seqtree import SeqTree
import random
import csv
from rich.progress import track
import torch
from gambit.dataloaders import get_preprocessed_path

app = typer.Typer()


ESM2_LAYERS_TO_MODEL_NAME = {
    48 : "esm2_t48_15B_UR50D",
    36 : "esm2_t36_3B_UR50D",
    33 : "esm2_t33_650M_UR50D",
    30 : "esm2_t30_150M_UR50D",
    12 : "esm2_t12_35M_UR50D",
    6  : "esm2_t6_8M_UR50D",
}


def get_esm2_model_alphabet(layers:int) -> tuple["ESM2", "Alphabet"]:
    assert layers in ESM2_LAYERS_TO_MODEL_NAME
    model_name = ESM2_LAYERS_TO_MODEL_NAME[layers]
    return torch.hub.load("facebookresearch/esm:main", model_name)


def get_esm_representations(model, alphabet, esm_layers, batch_tokens):
    batch_lengths = (batch_tokens != alphabet.padding_idx).sum(1)
    esm_results = model(batch_tokens, repr_layers=[esm_layers], return_contacts=False)
    token_representations = esm_results['representations'][esm_layers]
    sample_representations = torch.zeros(batch_tokens.shape[0], token_representations[0].shape[-1])
    for i, tokens_count in enumerate(batch_lengths):
        sample_representations[i] = token_representations[i, 1 : tokens_count - 1].mean(0)

    return sample_representations


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
def preprocess_esm(
    taxonomy:Path,
    msa:Path,
    marker_info:Path,
    output_seqtree:Path,
    output_dir:Path,
    partitions:int=5,
    layers:int = 6,
    seed:int=42,
):
    assert layers in ESM2_LAYERS_TO_MODEL_NAME.keys()

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

    
    model, alphabet = get_esm2_model_alphabet(layers)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    seqtree = SeqTree(classification_tree=root)
    
    random.seed(seed)
    total = sum(1 for _ in SeqIO.parse(msa, "fasta"))
    
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for record in track(SeqIO.parse(msa, "fasta"), total=total):
        name = record.id
        node = accesssion_to_node[record.id]
        partition = random.randint(0,partitions-1)
        seqtree.add(name, node, partition)

        path = get_preprocessed_path(output_dir, name)
        if path.exists():
            continue
        path.parent.mkdir(exist_ok=True, parents=True)
        
        assert len(record.seq) == current_location
        index = 0
        array = None
        data = []
        for marker_id, (start,end) in marker_info_dict.items():
            seq = str(record.seq[start:end]).replace("-","")
            data.append((marker_id, seq))

        index = 0
        batch_size = 1
        while index < len(data):
            batch = data[index:index+batch_size]
            _, _, batch_tokens = batch_converter(batch)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            # Extract per-residue representations (on CPU)
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[layers], return_contacts=True)
            token_representations = results["representations"][layers]

            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

            if array is None:
                array = torch.zeros((len(marker_info_dict), len(sequence_representations[0])), dtype=sequence_representations[0].dtype)
            
            for sub_index in range(batch_size):
                array[index+sub_index] = sequence_representations[sub_index]
            index += batch_size

        torch.save(array.half(), str(path))
        
    seqtree.save(output_seqtree)


if __name__ == "__main__":
    app()
