from io import StringIO
import tarfile
from pathlib import Path
import typer
from hierarchicalsoftmax import TreeDict
from rich.progress import track
from Bio import SeqIO

from barbet.embedding import get_key

app = typer.Typer()

@app.command()
def prune_to_representatives(seqtree:Path, representatives:Path, output:Path):

    print("Getting list of representatives from", representatives)
    keys_to_keep = []
    with tarfile.open(representatives, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.isfile() and member.name.endswith(".faa")]
        
        print(f"Processing {len(members)} files in {representatives}")

        for member in track(members):
            f = tar.extractfile(member)
            marker_id = Path(member.name.split("_")[-1]).with_suffix("").name

            fasta_io = StringIO(f.read().decode('ascii'))

            for record in SeqIO.parse(fasta_io, "fasta"):
                species_accession = record.id
                key = get_key(species_accession, marker_id)
                keys_to_keep.append(key)

    # keys_to_keep = set(keys_to_keep)
    print(f"Keeping {len(keys_to_keep)} representatives")

    print(f"Loading seqtree {seqtree}")
    
    seqtree = TreeDict.load(seqtree)
    print("Total", len(seqtree))
    missing = []
    for key in track(keys_to_keep):
        if key not in seqtree:
            missing.append(key)

    print(f"{len(missing)} representatives missing output {len(keys_to_keep)} (total: {len(seqtree)})")
    if len(missing):
        keys_to_keep = [k for k in keys_to_keep if k not in missing]

    new_seqtree = TreeDict(seqtree.classification_tree)
    new_seqtree.update({k:seqtree[k] for k in keys_to_keep})
    print("Total after pruning", len(new_seqtree))

    print("Saving seqtree to", output)
    new_seqtree.save(output)        



if __name__ == "__main__":
    app()    