import typer
from pathlib import Path
import torch

from transformers import T5Tokenizer, T5EncoderModel

from gambit.embedding import Embedding
app = typer.Typer()


class ProstT5Embedding(Embedding):
    def __init__(self):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(self.device)
        # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
        self.model.full() if self.device=='cpu' else self.model.half()


    def embed(self, seq:str) -> torch.Tensor:
        """ Takes a protein sequence as a string and returns an embedding vector. """
        
        # add pre-fixes accordingly (this already expects 3Di-sequences to be lower-case)
        # if you go from AAs to 3Di (or if you want to embed AAs), you need to prepend "<AA2fold>"
        # if you go from 3Di to AAs (or if you want to embed 3Di), you need to prepend "<fold2AA>"
        batch = [f"<AA2fold> {seq}"]

        # tokenize sequences and pad up to the longest sequence in the batch
        ids = self.tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding="longest",return_tensors='pt').to(self.device)

        # generate embeddings
        with torch.no_grad():
            embedding_repr = self.model(ids.input_ids, attention_mask=ids.attention_mask)

        # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens, incl. prefix ([0,1:8]) 
        emb_0 = embedding_repr.last_hidden_state[0] # shape (n, 1024)

        # if you want to derive a single representation (per-protein embedding) for the whole protein
        vector = emb_0.mean(dim=0) # shape (1024)

        if torch.isnan(vector).any():
            return None

        return vector        


@app.command()
def main(
    taxonomy:Path,
    marker_genes:Path,
    output_seqtree:Path,
    output_seqbank:Path,
    partitions:int=5,
    seed:int=42,
):
    model = ProstT5Embedding()
    model.preprocess(
        taxonomy=taxonomy,
        marker_genes=marker_genes,
        output_seqtree=output_seqtree,
        output_seqbank=output_seqbank,
        partitions=partitions,
        seed=seed,
    )


if __name__ == "__main__":
    app()
