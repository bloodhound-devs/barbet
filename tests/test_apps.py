import pytest
from pathlib import Path
import torch
import polars as pl
from hierarchicalsoftmax import SoftmaxNode

from barbet import Barbet
from barbet.embedding import Embedding
from barbet.modules import BarbetLightningModule


TEST_DATA_DIR = Path(__file__).parent / "data"


class MockPredictionTrainer():
    def predict(self, module, dataloaders):
        result = [
            torch.zeros( (32, 5) ),
            torch.zeros( (32, 5) ),
        ]
        result[0][:, -1] = 1
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


class MockCheckpoint():
    def __init__(self, *args, **kwargs):
        pass
        
    @property
    def hparams(self):
        return MockHParams()

    def setup_prediction(self, barbet, names:list[str]|str, threshold:float=0.0, save_probabilities:bool=False):
        if isinstance(names, str):
            names = [names]
        unique_names = sorted(set(names))
        self.results_df = pl.DataFrame(
            {
                "name": unique_names,
                "species_probability": [0.295] * len(unique_names),
                "species_prediction": ["E"] * len(unique_names),
            }
        )


@pytest.mark.parametrize("k", [1,2])
def test_predict(k, tmp_path):
    barbet = Barbet()
    # mock checkpoint
    barbet.load_checkpoint = lambda *args, **kwargs: MockCheckpoint()
    barbet.prediction_trainer = lambda *args, **kwargs: MockPredictionTrainer()

    output_dir = tmp_path / "output"
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # copy test data to temp directory
    input = []
    for i, file in enumerate([TEST_DATA_DIR / "MAG-GUT41.fa.gz"] * k):
        output_file = tmp_path / f"{i}.fa.gz"
        output_file.write_bytes(file.read_bytes())
        input.append(output_file)

    results = barbet.predict(input=input, output_dir=output_dir, image_format="dot")
    
    # Check output directory
    assert output_dir.exists()    
    assert (output_dir / "0.fa").exists()
    assert (output_dir / "0.fa" / "pfam.tblout").exists()

    # Check image files
    # image_file = (output_dir / "0.fa.gz.dot")
    # assert image_file.exists()
    # assert 'root" -> "A" [label=0.18' in image_file.read_text()
    
    # Check result df
    assert len(results) == k
    assert 'name' in results.columns
    assert 'species_prediction' in results.columns
    assert 'species_probability' in results.columns
    for i, row in enumerate(results.iter_rows(named=True)):
        assert f"{i}.fa.gz" in row['name']
        assert row['species_prediction'] == 'E'
        assert 0.29 < row['species_probability'] < 0.30
        