import gc
from torchapp.modules import GeneralLightningModule
# import pandas as pd
import polars as pl
import torch
from collections import defaultdict
from hierarchicalsoftmax.inference import (
    greedy_lineage_probabilities,
    node_probabilities,
    greedy_predictions,
)
from barbet.data import RANKS


class BarbetLightningModule(GeneralLightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_prediction(self, barbet, names:list[str]|str, threshold:float=0.0, save_probabilities:bool=False):
        self.names = names
        self.classification_tree = self.hparams.classification_tree
        self.logits = defaultdict(lambda: 0.0)
        self.counts = defaultdict(lambda: 0)
        self.counter = 0
        self.category_names = [
            barbet.node_to_str(node) for node in self.classification_tree.node_list_softmax if not node.is_root
        ]
        self.barbet = barbet
        self.threshold = threshold
        self.save_probabilities = save_probabilities

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
        print("Consolidating results per genome...")
        names = list(self.logits.keys())
        logits = torch.stack([
            self.logits[name] / self.counts[name] for name in names
        ], dim=0)
        del self.logits
        del self.counts
        gc.collect()

        # Prepare column names and initialize empty lists
        output_columns = ['name']
        new_cols = {}
        new_cols['name'] = names

        for rank in RANKS:
            pred_col = f"{rank}_prediction"
            prob_col = f"{rank}_probability"
            output_columns += [pred_col, prob_col]
            new_cols[pred_col] = []
            new_cols[prob_col] = []

        # Convert to probabilities
        if self.save_probabilities:
            print("Converting to probabilities...")
            probabilities = node_probabilities(
                logits, 
                root=self.classification_tree,
                progress_bar=True,
            )

            del logits
            gc.collect()

            print("Saving in dataframe...")
            self.results_df = pl.DataFrame(
                data=probabilities,
                schema=self.category_names
            ).with_columns([
                pl.Series("name", names, dtype=pl.Utf8)
            ]).with_columns([
                pl.col("name").cast(pl.Utf8)
            ]).select(["name", *self.category_names])
            
            # get greedy predictions which can use the raw activation or the softmax probabilities
            print("Getting greedy predictions...")
            predictions = greedy_predictions(
                probabilities,
                root=self.classification_tree,
                threshold=self.threshold,
                progress_bar=True,
            )

            del probabilities
            gc.collect()

            # Prepare essentials
            num_rows = self.results_df.height

            for i in range(num_rows):
                prediction_node = predictions[i]
                lineage = prediction_node.ancestors[1:] + (prediction_node,)
                probability = 1.0

                for rank, lineage_node in zip(RANKS, lineage):
                    node_name = self.barbet.node_to_str(lineage_node)
                    pred_col = f"{rank}_prediction"
                    prob_col = f"{rank}_probability"

                    new_cols[pred_col].append(node_name)

                    if node_name in self.results_df.columns:
                        probability = self.results_df[node_name][i]

                    new_cols[prob_col].append(probability)

            # Add new columns to the Polars DataFrame
            self.results_df = self.results_df.with_columns(
                [pl.Series(name, values) for name, values in new_cols.items()]
            )
            output_columns += self.category_names
            self.results_df = self.results_df[output_columns]
        else:
            print("Finding greedy predictions...")
            results = greedy_lineage_probabilities(
                logits, 
                root=self.classification_tree,
                threshold=self.threshold,
                progress_bar=True,
            )

            del logits
            gc.collect()

            for row in results:
                if not len(row) == len(RANKS):
                    breakpoint()
                assert len(row) == len(RANKS), f"Row length {len(row)} does not match number of ranks {len(RANKS)}"
                for rank_index, (node, probability) in enumerate(row):
                    rank = RANKS[rank_index]
                    node_name = self.barbet.node_to_str(node)
                    pred_col = f"{rank}_prediction"
                    prob_col = f"{rank}_probability"

                    new_cols[pred_col].append(node_name)
                    new_cols[prob_col].append(probability)

            # Create the DataFrame
            self.results_df = pl.DataFrame(
                data=new_cols,
                schema=output_columns
            ).with_columns([
                pl.col("name").cast(pl.Utf8)
            ]).select(output_columns)




        




