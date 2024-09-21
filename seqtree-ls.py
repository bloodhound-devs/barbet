from pathlib import Path
import typer
from corgi.seqtree import SeqTree
from rich.progress import track

app = typer.Typer()

@app.command()
def seqtree_ls(seqtree:Path):
    seqtree = SeqTree.load(seqtree)
    print(len(seqtree))
    xx = 0
    for accession in track(seqtree):
        # print(accession)
        detail = seqtree[accession]
        x = detail.node_id
        xx = max(x, xx)
        # print(detail.node_id)
    print(xx)



if __name__ == "__main__":
    app()    