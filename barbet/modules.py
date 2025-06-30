import gc
from torchapp.modules import GeneralLightningModule
import pandas as pd
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
        results_df = pd.DataFrame(
            results.cpu().numpy(), 
            columns=self.category_names,
        )
        batch_size = len(results)
        results_df["name"] = (
            self.names if isinstance(self.names, str) 
            else self.names[self.counter:self.counter+batch_size]
        )
        self.counter += batch_size
        results_df["counts"] = 1
        results_df = results_df.groupby(["name"]).sum()
        self.batch_dfs.append(results_df)

        # Concatenate and group every 200 batches to save memory
        if batch_idx and batch_idx % 200 == 0:
            self.batch_dfs = [pd.concat(self.batch_dfs).groupby(level=0).sum()]
            
    def on_predict_epoch_end(self):
        self.results_df = pd.concat(self.batch_dfs).groupby(level=0).sum()
        del self.batch_dfs
        gc.collect()

        # Divide by counts to get average
        counts = self.results_df["counts"]
        self.results_df = self.results_df.drop(columns=["counts"]).div(counts, axis=0)
        gc.collect()

        # Convert to probabilities
        probabilities = node_probabilities(
            torch.as_tensor(self.results_df[self.category_names].to_numpy()), 
            root=self.classification_tree,
        )
        self.results_df = pd.DataFrame(
            probabilities, 
            columns=self.category_names,
            index=self.results_df.index,
        )
        gc.collect()



        




