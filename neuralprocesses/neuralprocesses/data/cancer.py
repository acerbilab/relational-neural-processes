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

__all__ = ["CancerGenerator"]


class CancerGenerator(DataGenerator):
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
        obs_type="sane",
        num_tasks=10**3,
        num_context=UniformDiscrete(150, 150),
        num_target=UniformDiscrete(50, 50),
        forecast_start=UniformContinuous(4, 5),
        mode="completion",
        device="cpu",
    ):
        super().__init__(dtype, seed, num_tasks, batch_size=1, device=device)
        
        self.num_context = convert(num_context, AbstractDistribution)
        self.num_target = convert(num_target, AbstractDistribution)
        self.forecast_start = convert(forecast_start, AbstractDistribution)
	
        # Load the data.
        if dataset=="small":
            with open('experiment/data/dataset_cancer/datasetcancer_small.pkl', 'rb') as f:
                self.trajectories = pickle.load(f)
        elif dataset=="large":
            with open('./experiment/data/datasetcancer/datasetcancer.pkl', 'rb') as f:
                self.trajectories = pickle.load(f)
        else:
            raise ValueError(f'Bad datasel "(dataset)".')
        self.forecast_start=forecast_start
        self.mode=mode
        self.obs_type=obs_type
        
        
        ntime=self.trajectories[0][0].shape[0]
        nspace=self.trajectories[0][0].shape[1]
        
        N=random.choices(range(len(self.trajectories)),k=num_tasks)
        #Attention to this, I don't know how to have a maximum of both quantities
        xt1=np.zeros(shape=[num_tasks,1,200])
        xt2=np.zeros(shape=[num_tasks,1,200])
        yt=np.zeros(shape=[num_tasks,1,200])
        
        for i in range(num_tasks):
            if obs_type == "sane":
                traj_transf=self.trajectories[N[i]][0]
            elif obs_type == "cancer":
                traj_transf=self.trajectories[N[i]][1]
            elif obs_type == "diff":
                traj_transf=self.trajectories[N[i]][0]-self.trajectories[N[i]][0]
            else:
                raise ValueError(f'Bad obs_type "(obs_type)".')
            for j in range(200):
                tps=random.sample(range(ntime),1)[0]
                pos=random.sample(range(nspace),1)[0]
                xt1[i][0][j]=tps
                xt2[i][0][j]=pos
                yt[i][0][j]=traj_transf[tps][pos]
        		
        
        self.xt=list([torch.tensor(xt1,dtype=dtype),torch.tensor(xt2,dtype=dtype)])
        self.yt=list([torch.tensor(yt,dtype=dtype)])

    def generate_batch(self):
        with B.on_device(self.device):
            self.state, batch = apply_task(
                self.state,
                self.dtype,
                self.int64,
                self.mode,
                self.xt,
                self.yt,
                self.num_target,
                self.forecast_start,
            )
            return batch
