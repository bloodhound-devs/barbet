import torch
from torch import nn
from hierarchicalsoftmax import HierarchicalSoftmaxLazyLinear, SoftmaxNode


class BloodhoundModel(nn.Module):
    def __init__(self, classification_tree:SoftmaxNode, gene_family_count:int, family_embedding_size:int=64, features:int=5120, intermediate_layers:int=0, growth_factor:float=2.0):
        super().__init__()
        self.classification_tree = classification_tree
        modules = [nn.LazyLinear(out_features=features), nn.PReLU()]
        for _ in range(intermediate_layers):
            out_features = int(features * growth_factor + 0.5)
            modules += [nn.LazyLinear(out_features=out_features), nn.PReLU()]
            features = out_features

        modules.append(HierarchicalSoftmaxLazyLinear(root=classification_tree))

        self.family_embedding = nn.Embedding(gene_family_count, family_embedding_size) if family_embedding_size else None

        self.sequential = nn.Sequential(*modules)

    def forward(self, x, gene_family_id):
        # Concatenate an embedding for the marker gene family if the embedding module exists
        if self.family_embedding:
            family_embeddings = self.family_embedding(gene_family_id)
            x = torch.cat([x, family_embeddings], dim=1)
            
        result = self.sequential(x)
        return result
        
