from torch import nn
from hierarchicalsoftmax import HierarchicalSoftmaxLazyLinear, SoftmaxNode



class GambitModel(nn.Module):
    def __init__(self, classification_tree:SoftmaxNode, features:int=5120):
        super().__init__()
        self.classification_tree = classification_tree
        self.sequential = nn.Sequential(
            nn.LazyLinear(out_features=features),
            nn.PReLU(),
            HierarchicalSoftmaxLazyLinear(root=classification_tree),
        )

    def forward(self, embeddings):
        result = self.sequential(embeddings)
        return result
        