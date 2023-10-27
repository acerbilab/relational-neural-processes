import lab as B
import numpy as np
import wbml.util
from plum import convert
import random
import pickle
import torch

from .data import DataGenerator, apply_task
from ..dist import AbstractDistribution
from ..dist.uniform import UniformDiscrete, UniformContinuous

__all__ = ["CancerJointGenerator"]


class CancerJointGenerator(DataGenerator):
    """Simulations from the .

    Args:
        dtype (dtype): Data type to generate.
        dataset: "small" or "large" depending on the wanted file. Default to "small"
        obs_type: "sane", "cancer" or "diff" the nature of the observations. Default to "sane"
        num_context (int, optional): Number of tasks to generate per epoch. Must be an
            integer multiple of `batch_size`. Defaults to 2^.
        num_target (:class:`neuralprocesses.dist.dist.AbstractDistribution`, optional):
            Distribution of the number of target inputs. Defaults to the fixed number
            100.
        forecast_start (:class:`neuralprocesses.dist.dist.AbstractDistribution`,
            optional): Distribution of the start of the forecasting task. Defaults to
            a uniform distribution over $[25, 75]$.
        mode (str, optional): Mode. Must be one of `"interpolation"`, `"forecasting"`,
            `"reconstruction"`, or `"random"`.
        device (str, optional): Device on which to generate data. Defaults to `"cpu"`.

    Attributes:
        dtype (dtype): Data type.
        float64 (dtype): Floating point version of `dtype` with 64 bits.
        int64 (dtype): Integer version of `dtype` with 64 bits.
        num_tasks (int): Number of tasks to generate per epoch. Is an integer multiple
            of `batch_size`.
        batch_size (int): Batch size.
        forecast_start (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Distribution of the start of the forecasting task.
        num_target (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Distribution of the number of target inputs.
        mode (str): Mode.
        state (random state): Random state.
        device (str): Device.
    """

    def __init__(
            self,
            dtype,
            seed=0,
            dataset="small",
            num_tasks=10 ** 3,
            batch_size=16,
            num_context=UniformDiscrete(50, 200),
            num_target=UniformDiscrete(100, 200),
            forecast_start=2,
            mode="completion",
            device="cpu",
    ):
        super().__init__(dtype, seed, num_tasks, batch_size=batch_size, device=device)

        self.num_context = convert(num_context, AbstractDistribution)
        self.num_target = convert(num_target, AbstractDistribution)
        self.forecast_start = forecast_start
        self.mode = mode

        # Load the data.
        if dataset == "small":
            with open('experiment/data/dataset_cancer/datasetcancer_small.pkl', 'rb') as f:
                trajectories_data = pickle.load(f)
        elif dataset == "large":
            with open('./experiment/data/dataset_cancer/datasetcancer.pkl', 'rb') as f:
                trajectories_data = pickle.load(f)
        elif dataset == "small_test":
            with open('./experiment/data/dataset_cancer/datasetcancer_small_test.pkl', 'rb') as f:
                trajectories_data = pickle.load(f)
        elif dataset == "small_train":
            with open('./experiment/data/dataset_cancer/datasetcancer_small_train.pkl', 'rb') as f:
                trajectories_data = pickle.load(f)
        else:
            raise ValueError(f'Bad datasel "(dataset)".')

        # check data dimensinons
        ntime, nspace = trajectories_data[0][0].shape
        nx = int(B.sqrt(nspace))

        # choose observation type and reshape and scale
        max_value = 10
        self.trajectories1=[tr[0].reshape(ntime, nx, nx) / max_value for tr in trajectories_data]
        self.trajectories2=[tr[2].reshape(ntime, nx, nx) / max_value for tr in trajectories_data]

        nsamples = len(self.trajectories1)
        self.trajectories_ind = UniformDiscrete(0, nsamples - 1)
        self.x_ind = UniformDiscrete(1, nx - 1)
        if self.mode == "forecasting":
            self.time_ind_train = UniformDiscrete(-2, -1)
            self.time_ind_test = UniformDiscrete(2, 5)
        else:
            self.time_ind_train = UniformDiscrete(-1, 1)
            self.time_ind_test = UniformDiscrete(1, 4)

    def generate_batch(self):
        with B.on_device(self.device):

            # batch setup
            self.state, n_ctx = self.num_context.sample(self.state, self.int64)
            self.state, n_trg = self.num_target.sample(self.state, self.int64)
            n_ctx = int(n_ctx)
            n_trg = int(n_trg)

            self.state, inds = self.trajectories_ind.sample(self.state, self.int64, self.batch_size)

            self.state, test_time = self.time_ind_test.sample(self.state, self.int64,
                                                              self.batch_size)  # we sample one time, and all the task will be around this time.

            # random targets
            target_x = torch.zeros(self.batch_size, 3, n_trg).to(self.device)
            target_y = torch.zeros(self.batch_size, 2, n_trg).to(self.device)
            for b in range(self.batch_size):
                self.state, x1 = self.x_ind.sample(self.state, self.int64, n_trg)
                self.state, x2 = self.x_ind.sample(self.state, self.int64, n_trg)

                x = B.concat(x1.reshape(1, -1), x2.reshape(1, -1), test_time[b].repeat(x1.shape[0]).reshape(1, -1))  # check these size
                y = self.trajectories2[inds[b]][test_time[b].cpu().detach().numpy(), x1.cpu().detach().numpy(), x2.cpu().detach().numpy()]
                target_x[b] = x
                target_y[b] = torch.from_numpy(y).to(self.device)
            # random context
            context_x = torch.zeros(self.batch_size, 3, n_ctx).to(self.device)
            context_y = torch.zeros(self.batch_size, 2, n_ctx).to(self.device)
            # print(type(context_x))
            # print(type(context_y))

            for b in range(self.batch_size):
                self.state, x1 = self.x_ind.sample(self.state, self.int64, n_ctx)
                self.state, x2 = self.x_ind.sample(self.state, self.int64, n_ctx)
                self.state, time1 = self.time_ind_train.sample(self.state, self.int64, n_ctx)

                time2 = time1 + test_time[b]  # we sample around the target time

                x = B.concat(x1.reshape(1, -1), x2.reshape(1, -1), time2.reshape(1, -1))
                # print(type(x))

                y = self.trajectories1[inds[b]][time2.cpu().detach().numpy(), x1.cpu().detach().numpy(), x2.cpu().detach().numpy()]
                # print(type(y))
                context_x[b] = x
                context_y[b] = torch.from_numpy(y).to(self.device)

            # make batch dict
            batch = {}
            batch['contexts'] = [(context_x, context_y)]
            batch['xt'] = target_x
            batch['yt'] = target_y

            return batch