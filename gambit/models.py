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


class GambitESMModel(nn.Module):
    esm: nn.Module
    esm_layers: int

    def __init__(self, esm:nn.Module, esm_layers:int, classification_tree:SoftmaxNode):
        self.esm = esm
        self.esm_layers = esm_layers
        self.classification_tree = classification_tree

        self.softmax_layer = HierarchicalSoftmaxLazyLinear(root=classification_tree)

    def forward(self, x):
        esm_results = self.esm(x, repr_layers=[self.esm_layers], return_contacts=False)
        representations = esm_results['representations'][self.esm_layers]
        return self.softmax_layer(representations)
        