import torch
from torch import nn
# from esm.model.esm2 import ESM2
# from esm.data import Alphabet
from hierarchicalsoftmax import HierarchicalSoftmaxLinear, HierarchicalSoftmaxLazyLinear, SoftmaxNode


ESM2_LAYERS_TO_MODEL_NAME = {
    48 : "esm2_t48_15B_UR50D",
    36 : "esm2_t36_3B_UR50D",
    33 : "esm2_t33_650M_UR50D",
    30 : "esm2_t30_150M_UR50D",
    12 : "esm2_t12_35M_UR50D",
    6  : "esm2_t6_8M_UR50D",
}


def get_esm2_model_alphabet(layers:int) -> tuple["ESM2", "Alphabet"]:
    assert layers in ESM2_LAYERS_TO_MODEL_NAME
    model_name = ESM2_LAYERS_TO_MODEL_NAME[layers]
    return torch.hub.load("facebookresearch/esm:main", model_name)


def get_esa_representations(model, alphabet, esm_layers, batch_tokens):
    batch_lengths = (batch_tokens != alphabet.padding_idx).sum(1)
    esm_results = model(batch_tokens, repr_layers=[esm_layers], return_contacts=False)
    token_representations = esm_results['representations'][esm_layers]
    sample_representations = torch.zeros(batch_tokens.shape[0], token_representations[0].shape[-1])
    for i, tokens_count in enumerate(batch_lengths):
        sample_representations[i] = token_representations[i, 1 : tokens_count - 1].mean(0)

    return sample_representations


class GambitModel(nn.Module):
    def __init__(self, classification_tree:SoftmaxNode):
        super().__init__()
        self.classification_tree = classification_tree
        self.softmax_layer = HierarchicalSoftmaxLazyLinear(root=classification_tree)

    def forward(self, embeddings):
        result = self.softmax_layer(embeddings)
        return result
        