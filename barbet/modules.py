from torchapp.modules import GeneralLightningModule
import pandas as pd
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
        self.results_df = None
        self.counter = 0
        self.category_names = [
            barbet.node_to_str(node) for node in self.classification_tree.node_list_softmax if not node.is_root
        ]

    def on_predict_batch_end(self, results, batch, batch_idx, dataloader_idx=0):
        classification_probabilities = node_probabilities(
            results.cpu(), 
            root=self.classification_tree,
        )

        results_df = pd.DataFrame(
            classification_probabilities.numpy(), 
            columns=self.category_names,
        )
        results_df["name"] = (
            self.names if isinstance(self.names, str) 
            else self.names[self.counter:self.counter+len(classification_probabilities)]
        )
        results_df["counts"] = 1
        results_df = results_df.groupby(["name"]).sum()

        if self.results_df is None:
            self.results_df = results_df
        else:
            self.results_df = self.results_df.add(results_df, fill_value=0)

    def on_predict_epoch_end(self):
        # Divide by counts to get average
        counts = self.results_df["counts"]
        self.results_df = self.results_df.drop(columns=["counts"]).div(counts, axis=0)


        




