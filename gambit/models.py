from torch import nn
from hierarchicalsoftmax import HierarchicalSoftmaxLinear, HierarchicalSoftmaxLazyLinear, SoftmaxNode



class GambitModel(nn.Module):
    def __init__(self, classification_tree:SoftmaxNode):
        super().__init__()
        self.classification_tree = classification_tree
        self.softmax_layer = HierarchicalSoftmaxLazyLinear(root=classification_tree)

    def forward(self, embeddings):
        result = self.softmax_layer(embeddings)
        return result
        