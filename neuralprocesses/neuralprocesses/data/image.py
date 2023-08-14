import lab as B
import numpy as np
from plum import convert
import torch
import torchvision

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .data import DataGenerator
from ..dist import AbstractDistribution
from ..dist.uniform import UniformDiscrete

__all__ = ["ImageGenerator"]


class ImageGenerator(DataGenerator):
    """Image data generator.

    Args:
        dtype (dtype): Data type.
        rootdir (str): Root or image dataset directory.
        dataset (str): Image data used to construct the interpolation tasks. Must be in
            "mnist", "mnist_trans", "mnist16", "mnist16_trans", "celeba16", "celeba32".
        seed (int, optional): Seed. Defaults to 0.
        num_tasks (int, optional): Number of batches in an epoch. Defaults to 2^14.
        batch_size (int, optional): Number of tasks per batch. Defaults to 16.
        num_context (:class:`neuralprocesses.dist.dist.AbstractDistribution`, optional):
            Distribution of the number of context points. Defaults to a uniform
            distribution over $[n/100, n/2]$, where n denotes the image size in pixels.
        load_data (bool, optional): Load image data to dataset directory if not loaded.
            Defaults to `True`.
        subset (str, optional): Dataset partition. Must be in "train", "valid", "test".
            Defaults to "train".
        device (str, optional): Device. Defaults to "cpu".

    Attributes:
        dtype (dtype): Data type.
        float64 (dtype): Floating point version of the data type.
        int64 (dtype): Integral version of the data type.
        seed (int): Seed.
        batch_size (int): Number of tasks per batch.
        num_batches (int): Number of batches in an epoch.
        device (str): Device.
        state (random state): Random state.
        numpygen (:class:`numpy.random.Generator`): Random number generator.
        num_context (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Distribution of the number of context points.
        data (:class:`torchvision.datasets.VisionDataset`): Loaded image data.
        data_inds (:class:`neuralprocesses.dist.uniform.UniformDiscrete`):
            Distribution used in image selection.
        dim_y (int): Number of channels.
        n_tot (int): Number of pixels.
        grid (`torch.Tensor`): Pixel coordinates.
    """

    def __init__(
        self,
        dtype,
        rootdir,
        dataset,
        seed=0,
        num_tasks=2**14,
        batch_size=16,
        num_context=None,
        load_data=True,
        subset="train",
        device="cpu",
    ):
        super().__init__(dtype, seed, num_tasks, batch_size=batch_size, device=device)

        # load image data
        assert subset in ["train", "valid", "test"]

        if dataset == "mnist":
            data = datasets.MNIST(
                root=rootdir,
                train=not(subset == "test"),
                download=load_data,
                transform=transforms.ToTensor()
            )
        elif dataset == "mnist_trans":
            cropper = torchvision.transforms.RandomCrop(size=(28, 28))
            padder = torchvision.transforms.Pad(padding=5)
            totensor = transforms.ToTensor()
            func = lambda x: totensor(cropper(padder(x)))
            data = datasets.MNIST(
		root=rootdir,
		train=not(subset == "test"),
		download=load_data,
		transform=func
	    )
        elif dataset == "mnist16":
            transforms_list = [transforms.Resize(16), transforms.ToTensor()]
            data = datasets.MNIST(
                root=rootdir,
                train=not(subset == "test"),
                download=load_data,
                transform=transforms.Compose(transforms_list)
            )
        elif dataset == "mnist16_trans":
            transforms_list = [transforms.Pad(padding=5), transforms.RandomCrop(size=(28, 28)), transforms.Resize(16), transforms.ToTensor()]
            data = datasets.MNIST(
                root=rootdir,
                train=not(subset == "test"),
                download=load_data,
                transform=transforms.Compose(transforms_list)
            )
        elif dataset == "celeba32":
            transforms_list = [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]
            data = datasets.CelebA(
                root=rootdir,
                split=subset,
                download=load_data,
                transform=transforms.Compose(transforms_list)
            )
        elif dataset == "celeba16":
            transforms_list = [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
            data = datasets.CelebA(
                root=rootdir,
                split=subset,
                download=load_data,
                transform=transforms.Compose(transforms_list)
            )
        else:
            raise ValueError(f'Unknown dataset {dataset}.')

        # random image selection setup
        #torch.manual_seed(seed)
        #self.dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.data = data
        self.data_inds = UniformDiscrete(0, len(data)-1)

        # image properties
        self.dim_y, _, image_size = data[0][0].shape
        self.n_tot = image_size**2
        axis = torch.arange(image_size, dtype=torch.float32)/(image_size-1)
        grid = torch.stack((axis.repeat_interleave(image_size), axis.repeat(image_size)))
        self.grid = grid.to(device)

        # random context selection setup
        num_context = num_context or UniformDiscrete(int(self.n_tot/100), int(self.n_tot/2))
        self.num_context = convert(num_context, AbstractDistribution)
        self.numpygen = np.random.default_rng(seed=seed)

    def generate_batch(self):
        with B.on_device(self.device):
            #features, labels = next(iter(self.dataloader))
            self.state, inds = self.data_inds.sample(self.state, self.int64, self.batch_size)
            features = torch.cat([self.data[ind][0] for ind in inds])

            # target features
            target_x = self.grid.repeat([self.batch_size, 1, 1])
            target_y = features.reshape(self.batch_size, self.dim_y, -1).to(self.device)

            # context features
            self.state, n_ctx = self.num_context.sample(self.state, self.int64)
            n_ctx = int(n_ctx)

            context_x = torch.zeros(self.batch_size, 2, n_ctx).to(self.device)
            context_y = torch.zeros(self.batch_size, self.dim_y, n_ctx).to(self.device)

            for b in range(self.batch_size):
                inds = self.numpygen.choice(self.n_tot, n_ctx, replace=False)
                context_x[b] = target_x[b, :, inds]
                context_y[b] = target_y[b, :, inds]

            # make batch dict
            batch = {}
            batch["contexts"] = [(context_x, context_y)]
            batch["xt"] = target_x
            batch["yt"] = target_y

            return batch
