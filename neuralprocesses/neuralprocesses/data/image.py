import lab as B
import numpy as np
from plum import convert
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .data import DataGenerator
from ..dist import AbstractDistribution
from ..dist.uniform import UniformDiscrete

__all__ = ["ImageGenerator"]


class ImageGenerator(DataGenerator):

    def __init__(
        self,
        dtype,
        rootdir,
        dataset,
        seed=0,
        num_tasks=2**14,
        batch_size=16,
        num_context=UniformDiscrete(10, 500),
        subset="train",
        device="cpu",
    ):
        super().__init__(dtype, seed, num_tasks, batch_size=batch_size, device=device)

        self.num_context = convert(num_context, AbstractDistribution)
        self.numpygen = np.random.default_rng(seed=seed)

        assert subset in ["train", "test"]

        if dataset == "mnist":
            data = datasets.MNIST(
                root=rootdir,
                train=(subset == "train"),
                download=True,
                transform=transforms.ToTensor()
            )
        else:
            raise ValueError(f'Unknown dataset {dataset}.')

        #torch.manual_seed(seed)
        #self.dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.data = data
        self.data_inds = UniformDiscrete(0, len(data)-1)

        image_size = data[0][0].shape[-1]
        self.n_tot = image_size**2
        axis = torch.arange(image_size, dtype=torch.float32)
        grid = torch.stack((axis.repeat_interleave(image_size), axis.repeat(image_size)))
        self.grid = grid.to(device)

    def generate_batch(self):
        with B.on_device(self.device):
            #features, labels = next(iter(self.dataloader))
            self.state, inds = self.data_inds.sample(self.state, self.int64, self.batch_size)
            features = torch.cat([self.data[ind][0] for ind in inds])

            # target features
            target_x = self.grid.repeat([self.batch_size, 1, 1])  # bs x 2 x ntot
            target_y = features.reshape(self.batch_size, 1, -1).to(self.device)  # bs x 1 x ntot

            # context features
            self.state, n_ctx = self.num_context.sample(self.state, self.int64)
            n_ctx = int(n_ctx)

            context_x = torch.zeros(self.batch_size, 2, n_ctx).to(self.device)
            context_y = torch.zeros(self.batch_size, 1, n_ctx).to(self.device)

            for b in range(self.batch_size):
                inds = self.numpygen.choice(self.n_tot, n_ctx, replace=False)
                
                context_x[b] = target_x[b, :, inds]  # 2 x ntcx
                context_y[b] = target_y[b, :, inds]  # 1 x nctx

            # make batch dict
            batch = {}
            batch["contexts"] = [(context_x, context_y)]
            batch["xt"] = target_x
            batch["yt"] = target_y

            return batch
