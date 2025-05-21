import pytest
from pathlib import Path
from barbet import Barbet
import torch
from barbet.embedding import Embedding
from hierarchicalsoftmax import SoftmaxNode

TEST_DATA_DIR = Path(__file__).parent / "data"


class MockPredictionTrainer():
    def predict(self, module, dataloaders):
        result = [
            torch.zeros( (32, 5) ),
            torch.zeros( (32, 5) ),
        ]
        result[0][0, -1] = 1
        return result


class MockEmbedding(Embedding):
    def embed(self, seq: str) -> str:
        embedding = torch.zeros((len(seq), 26))
        embedding[0, -1] = 1
        return embedding


class MockHParams:
    def __init__(self):
        self.embedding_model = MockEmbedding()
        root = SoftmaxNode('root')
        SoftmaxNode('A', parent=root)
        SoftmaxNode('B', parent=root)
        SoftmaxNode('C', parent=root)
        SoftmaxNode('D', parent=root)
        SoftmaxNode('E', parent=root)
        root.set_indexes()
        self.classification_tree = root

    def get(self, key, default=None):
        return getattr(self, key, default)


class MockCheckpoint:
    def __init__(self):
        self.hparams = MockHParams()
    

@pytest.mark.parametrize("k", [1,2])
def test_predict(k):
    barbet = Barbet()
    # mock checkpoint
    barbet.load_checkpoint = lambda *args, **kwargs: MockCheckpoint()
    barbet.prediction_trainer = lambda *args, **kwargs: MockPredictionTrainer()
    results = barbet(input=[TEST_DATA_DIR/"MAG-GUT41.fa.gz"] * k)
    assert len(results) == k
    assert 'name' in results.columns
    assert 'greedy_prediction' in results.columns
    assert 'probability' in results.columns
    for _, row in results.iterrows():
        assert "MAG-GUT41.fa.gz" in row['name']
        assert row['greedy_prediction'] == 'E'
        assert 0.2 < row['probability'] < 0.25
        