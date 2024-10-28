import torch
from transformers import T5Tokenizer, T5EncoderModel

from bloodhound.embedding import Embedding


class ProstT5Embedding(Embedding):
    def setup(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(self.device)
        # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
        self.model.full() if self.device=='cpu' else self.model.half()

    def embed(self, seq:str) -> torch.Tensor:
        """ Takes a protein sequence as a string and returns an embedding tensor per residue. """
        # add pre-fixes accordingly (this already expects 3Di-sequences to be lower-case)
        # if you go from AAs to 3Di (or if you want to embed AAs), you need to prepend "<AA2fold>"
        # if you go from 3Di to AAs (or if you want to embed 3Di), you need to prepend "<fold2AA>"
        spaced_seq = " ".join(seq.upper())
        batch = [f"<AA2fold> {spaced_seq}"]

        # tokenize sequences and pad up to the longest sequence in the batch
        ids = self.tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding="longest",return_tensors='pt').to(self.device)

        # generate embeddings
        with torch.no_grad():
            embedding_repr = self.model(ids.input_ids, attention_mask=ids.attention_mask)

        # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens, incl. prefix
        emb_0 = embedding_repr.last_hidden_state[0,1:-1]

        assert emb_0.shape[0] == len(seq)

        return emb_0

