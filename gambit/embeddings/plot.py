from pathlib import Path
from seqbank import SeqBank
from corgi.seqtree import SeqTree
from sklearn.decomposition import PCA
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import typer
import torch
import numpy as np


app = typer.Typer()

def plot_embeddings(seqtree:SeqTree, seqbank:SeqBank, gene_family_id:str="", depth:int=1) -> go.Figure:
    categories = []
    data = []
    for accession in seqtree:
        if not gene_family_id:
            gene_family_id = accession.split("/")[-1]
            print(f"No gene family given. Using {gene_family_id}")

        if accession.endswith(f"/{gene_family_id}"):
            node = seqtree.node(accession)
            category = node.ancestors[depth].name
            categories.append(category)
            embedding = seqbank[accession]
            array = torch.frombuffer(embedding, dtype=torch.float32)
            data.append(array.numpy())

    data = np.array(data)
    print(data.shape)
    
    # Do PCA with sklearn
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)

    # Plot with plotly
    df = pd.DataFrame(pca_data, columns=["x", "y"])
    df["category"] = categories
    fig = px.scatter(df, x="x", y="y", color="category", title=gene_family_id)
    filename = f"{gene_family_id}.png"
    print(f"Writing to {filename}")
    fig.write_image(f"{gene_family_id}.png")
    return fig


@app.command()
def plot_embeddings_cli(seqtree:Path, seqbank:Path, gene_family_id:str="", depth:int=1, show:bool=False):
    seqtree = SeqTree.load(seqtree)
    seqbank = SeqBank(seqbank)
    fig = plot_embeddings(seqtree, seqbank, gene_family_id, depth)
    if show:
        fig.show()


if __name__ == "__main__":
    app()