from enum import Enum
import typer
from pathlib import Path
import torch
from torchapp.cli import method
from bloodhound.embedding import Embedding



class ESMLayers(Enum):
    T6 = "6"
    T12 = "12"
    T30 = "30"
    T33 = "33"
    T36 = "36"
    T48 = "48"

    def model_name(self) -> str:
        match self:
            case ESMLayers.T48:
                return "esm2_t48_15B_UR50D"
            case ESMLayers.T36:
                return "esm2_t36_3B_UR50D"
            case ESMLayers.T33:
                return "esm2_t33_650M_UR50D"
            case ESMLayers.T30:
                return "esm2_t30_150M_UR50D"
            case ESMLayers.T12:
                return "esm2_t12_35M_UR50D"
            case ESMLayers.T6:
                return "esm2_t6_8M_UR50D"

    def get_model_alphabet(self) -> tuple["ESM2", "Alphabet"]:
        return torch.hub.load("facebookresearch/esm:main", self.model_name())


class ESMEmbedding(Embedding):
    @method
    def setup(
        self, 
        layers:ESMLayers=typer.Option(..., help="The number of ESM layers to use."),
        hub_dir:Path=typer.Option(None, help="The torch hub directory where the ESM models will be cached."),
    ):
        self.layers = layers
        assert layers in ESMLayers, f"Please ensure the number of ESM layers is one of " + ", ".join(ESMLayers.keys())

        self.hub_dir = hub_dir
        if hub_dir:
            torch.hub.set_dir(str(hub_dir))
        self.model = None
        self.device = None
        self.batch_converter = None
        self.alphabet = None

    def __getstate__(self):
        # Return a dictionary of attributes to be pickled
        state = self.__dict__.copy()
        # Remove the attribute that should not be pickled
        if 'model' in state:
            del state['model']
        if 'batch_converter' in state:
            del state['batch_converter']
        if 'alphabet' in state:
            del state['alphabet']
        if 'device' in state:
            del state['device']
        return state

    def __setstate__(self, state):
        # Restore the object state from the unpickled state
        self.__dict__.update(state)
        self.model = None
        self.device = None
        self.batch_converter = None
        self.alphabet = None

    def load(self):
        self.model, self.alphabet = self.layers.get_model_alphabet()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def embed(self, seq:str) -> torch.Tensor:
        """ Takes a protein sequence as a string and returns an embedding vector. """
        layers = int(self.layers.value)

        if not self.model:
            self.load()

        _, _, batch_tokens = self.batch_converter([("marker_id", seq)])
        batch_tokens = batch_tokens.to(self.device)                
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[layers], return_contacts=True)
        token_representations = results["representations"][layers]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

        vector = sequence_representations[0]
        # if torch.isnan(vector).any():
        #     return None

        return vector        


# @app.command()
# def main(
#     taxonomy:Path,
#     marker_genes:Path,
#     output_seqtree:Path,
#     output_seqbank:Path,
#     layers:int,
#     partitions:int=5,
#     seed:int=42,
#     file_stride: Annotated[
#         int, 
#         typer.Option(help="A stride value for subsetting the marker gene files. Set this to the number of jobs to run in parallel.")
#     ]=0,
#     file_offset:Annotated[
#         int, 
#         typer.Option(help="An offset value for subsetting the marker gene files. Set this to the job index (0 <= file_offset < file_stride).")
#     ]=0,
#     hub_dir:Annotated[
#         Path, 
#         typer.Option(help="The torch hub directory where the ESM models will be saved.")
#     ]=None,
#     filter:list[str]=None,
#     generate:bool=True, # whether or not to generate the embeddings if they don't exist. only use if all embeddings already exists and you only want to create a seqtree
# ):
#     model = ESMEmbedding(layers=layers, hub_dir=hub_dir)
#     model.preprocess(
#         taxonomy=taxonomy,
#         marker_genes=marker_genes,
#         output_seqtree=output_seqtree,
#         output_seqbank=output_seqbank,
#         partitions=partitions,
#         seed=seed,
#         file_stride=file_stride,
#         file_offset=file_offset,
#         filter=filter,
#         generate=generate,
#     )


# if __name__ == "__main__":
#     app()

