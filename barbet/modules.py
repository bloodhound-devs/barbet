import gc
from torchapp.modules import GeneralLightningModule
# import pandas as pd
import polars as pl
import torch
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
        self.batch_dfs = []
        self.counter = 0
        self.category_names = [
            barbet.node_to_str(node) for node in self.classification_tree.node_list_softmax if not node.is_root
        ]

    def on_predict_batch_end(self, results, batch, batch_idx, dataloader_idx=0):
        data = results.cpu().numpy()
        results_df = pl.DataFrame(data, schema=self.category_names)

        batch_size = len(results)
        names = (
            self.names if isinstance(self.names, str)
            else self.names[self.counter : self.counter + batch_size]
        )
        self.counter += batch_size
        
        results_df = results_df.with_columns([
            pl.Series("name", names),
            pl.lit(1).alias("counts"),
        ])

        # Group by name and sum predictions and counts
        results_df = results_df.group_by("name", maintain_order=True).sum()
        self.batch_dfs.append(results_df)

        # Concatenate and group every n batches to save memory
        # if batch_idx and batch_idx % 1000 == 0:
        #     grouped = pl.concat(self.batch_dfs).group_by("name").sum()
        #     self.batch_dfs = [grouped]
        #     gc.collect()

    def on_predict_epoch_end(self):
        results_df = pl.concat(self.batch_dfs).group_by("name").sum()
        del self.batch_dfs
        gc.collect()

        # Divide by counts to get average
        count_series = results_df["counts"]
        results_df = results_df.drop("counts")
        for col in self.category_names:
            results_df = results_df.with_columns([
                (pl.col(col) / count_series).alias(col)
            ])
        gc.collect()

        # Convert to probabilities
        probabilities = node_probabilities(
            torch.as_tensor(results_df[self.category_names].to_numpy()), 
            root=self.classification_tree,
        )
        self.results_df = pl.DataFrame(
            data=probabilities,
            schema=self.category_names
        ).with_columns([
            pl.Series("name", results_df["name"])
        ]).with_columns([
            pl.col("name").cast(pl.Utf8)
        ]).select(["name", *self.category_names])
        gc.collect()



        




