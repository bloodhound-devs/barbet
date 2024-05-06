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
        

from torch import nn
from hierarchicalsoftmax import HierarchicalSoftmaxLinear, SoftmaxNode



class GambitModel2(nn.Module):
    def __init__(self, classification_tree:SoftmaxNode, features:int=5120, intermediate_layers:int=0, growth_factor:float=2.0):
        super().__init__()
        self.classification_tree = classification_tree
        modules = [nn.LazyLinear(out_features=features), nn.PReLU()]
        for _ in range(intermediate_layers):
            out_features = int(features * growth_factor + 0.5)
            modules += [nn.LazyLinear(out_features=out_features), nn.PReLU()]
            features = out_features

        modules.append(HierarchicalSoftmaxLazyLinear(root=classification_tree))

        self.sequential = nn.Sequential(*modules)

    def forward(self, embeddings):
        result = self.sequential(embeddings)
        return result
        
