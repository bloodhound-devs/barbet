import torch
from torchmetrics import Metric


class AvgSmoothLoss(Metric):
    def __init__(self, beta=0.98, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.beta = beta
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("val", default=torch.tensor(0.), dist_reduce_fx="sum")

    def reset(self):
        self.count = torch.tensor(0)
        self.val = torch.tensor(0.)

    def update(self, loss):
        # Ensure loss is detached to avoid graph-related issues
        loss = loss.detach()
        self.count += 1
        self.val = torch.lerp(loss.mean(), self.val, self.beta)

    def compute(self):
        # Return the smoothed loss value
        return self.val / (1 - self.beta**self.count)


