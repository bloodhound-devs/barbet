import gc
from torchapp.modules import GeneralLightningModule
# import pandas as pd
import polars as pl
import torch
from collections import defaultdict
from hierarchicalsoftmax.inference import (
    node_probabilities,
    greedy_predictions,
    render_probabilities,
)

class BarbetLightningModule(GeneralLightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

#     def predict_step(self, x, batch_idx, dataloader_idx=0):
#         breakpoint()
#         return super().predict_step(x, batch_idx, dataloader_idx)

    def setup_prediction(self, barbet, names:list[str]|str):
        self.names = names
        self.classification_tree = self.hparams.classification_tree
        self.logits = defaultdict(lambda: 0.0)
        self.counts = defaultdict(lambda: 0)
        self.counter = 0
        self.category_names = [
            barbet.node_to_str(node) for node in self.classification_tree.node_list_softmax if not node.is_root
        ]

    def on_predict_batch_end(self, results, batch, batch_idx, dataloader_idx=0):
        batch_size = len(results)
        if isinstance(self.names, str):
            self.counts[self.names] += batch_size
            self.logis[self.names] += results.sum(dim=0).half().cpu()
        else:
            prev_name = self.names[self.counter]
            start_i = 0
            for end_i in range(batch_size):
                current_name = self.names[self.counter + end_i]
                if current_name != prev_name:
                    self.counts[prev_name] += (end_i - start_i)
                    self.logits[prev_name] += results[start_i:end_i].sum(dim=0).half().cpu()
                    start_i = end_i
                    prev_name = current_name
            
            # Handle the last chunk
            assert start_i < batch_size, "Start index should be less than batch size"
            self.logits[prev_name] += results[start_i:].sum(dim=0).cpu()
            self.counts[prev_name] += (batch_size - start_i)
            self.counter += batch_size

    def on_predict_epoch_end(self):
        print("AAAA")
        names = list(self.logits.keys())
        logits = torch.stack([
            self.logits[name] / self.counts[name] for name in names
        ], dim=0)
        print("BBB")
        # Convert to probabilities
        # Memory Spike here
        probabilities = node_probabilities(
            logits, 
            root=self.classification_tree,
        )
        print("CCCCC")
        self.results_df = pl.DataFrame(
            data=probabilities,
            schema=self.category_names
        ).with_columns([
            pl.Series("name", names, dtype=pl.Utf8)
        ]).with_columns([
            pl.col("name").cast(pl.Utf8)
        ]).select(["name", *self.category_names])
        print("DDDDDD")
        gc.collect()



        




