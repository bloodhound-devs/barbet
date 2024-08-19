import typer
from pathlib import Path
from seqbank import SeqBank
import torch
from rich.progress import track
import numpy as np

app = typer.Typer()

@app.command()
def convert(seqbank:Path, memmap:Path, index:Path):
    seqbank = SeqBank(seqbank)
    accessions = seqbank.get_accessions()
    count = len(accessions)
    dtype = 'float16'

    memmap_array = None
    with open(index, "w") as f:
        for index, accession in track(enumerate(accessions), total=len(accessions)):
            data = seqbank[accession]
            array = torch.frombuffer(data, dtype=torch.float32)
            del data

            if memmap_array is None:
                size = len(array)
                shape = (count,size)
                memmap_array = np.memmap(memmap, dtype=dtype, mode='w+', shape=shape)

            memmap_array[index,:] = array.half().numpy()
            if index % 1000 == 0:
                memmap_array.flush()

            print(accession, file=f)

    memmap_array.flush()


if __name__ == "__main__":
    app()    