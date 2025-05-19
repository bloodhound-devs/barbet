import torch
from barbet.embedding import Embedding


class DummyEmbedding(Embedding):
    def setup(self):
        self.size = 26
        self.max_length = 12
        self.overlap:int=4

    def embed(self, seq: str) -> str:
        tensor = torch.zeros( (len(seq), self.size) )
        for i, chararacter in enumerate(seq):
            tensor[i, ord(chararacter) - ord('A')] = 1
        return tensor


def test_embedding_short():
    embedding = DummyEmbedding()
    embedding.setup()
    assert embedding.embed('ABC').shape == (3, 26)
    assert embedding('ABC').shape == (26,)
    assert embedding('ABBC').argmax() == 1


def test_embedding_long():
    embedding = DummyEmbedding()
    embedding.setup()
    assert embedding.embed('ABC'*10).shape == (30, 26)
    assert embedding('ABC'*10).shape == (26,)
    assert embedding('ABBC'*10).argmax() == 1

    abc = embedding('ABC'*10)
    abbc = embedding('ABBC'*10)

    embedding.max_length = None
    assert embedding.embed('ABC'*10).shape == (30, 26)
    assert embedding('ABC'*10).shape == (26,)
    assert embedding('ABBC'*10).argmax() == 1
    assert (abc == embedding('ABC'*10)).all()
    assert (abbc == embedding('ABBC'*10)).all()