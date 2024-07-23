from pathlib import Path
from seqbank import SeqBank
import torch
import typer
from rich.progress import track

app = typer.Typer()

def zscore_normalize_seqbank(seqbank:SeqBank, output:SeqBank) -> float:
    sum_embeddings = 0.0
    sum_squared_embeddings = 0.0
    num_samples = 0

    accessions = seqbank.get_accessions()
    
    for accession in track(accessions, description="Calculating statistics"):
        data = seqbank[accession]
        array = torch.frombuffer(data, dtype=torch.float32)
        del data

        sum_embeddings += torch.sum(array)
        sum_squared_embeddings += torch.sum(array ** 2)
        num_samples += 1
        
    mean = sum_embeddings / num_samples
    variance = (sum_squared_embeddings / num_samples) - (mean ** 2)
    std_dev = torch.sqrt(variance)
    print(mean, std_dev, num_samples)
    
    for accession in track(accessions, description="Writing output"):
        data = seqbank[accession]
        array = torch.frombuffer(data, dtype=torch.float32)
        del data

        normalized = (array - mean) / std_dev

        output.add(
            seq=normalized.cpu().detach().clone().numpy().tobytes(),
            accession=accession,
        )


@app.command()
def zscore_normalize(seqbank:Path, output:Path):
    seqbank = SeqBank(seqbank)
    output = SeqBank(output, write=True)

    zscore_normalize_seqbank(seqbank, output)


if __name__ == "__main__":
    app()