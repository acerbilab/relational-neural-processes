import lab as B
import stheno
import warnings
import math
import numpy as np


from .data import SyntheticGenerator, new_batch

# from .sample_theno import sample_kernel

__all__ = ["GPGenerator"]


class GPGenerator(SyntheticGenerator):
    """GP generator.

    Further takes in arguments and keyword arguments from the constructor of
    :class:`.data.SyntheticGenerator`. Moreover, also has the attributes of
    :class:`.data.SyntheticGenerator`.

    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel of the GP. Defaults to an
            EQ kernel with length scale `0.25`.
        pred_logpdf (bool, optional): Also compute the logpdf of the target set given
            the context set under the true GP. Defaults to `True`.
        pred_logpdf_diag (bool, optional): Also compute the logpdf of the target set
            given the context set under the true diagonalised GP. Defaults to `True`.

    Attributes:
        kernel (:class:`stheno.Kernel`): Kernel of the GP.
        pred_logpdf (bool): Also compute the logpdf of the target set given the context
            set under the true GP.
        pred_logpdf_diag (bool): Also compute the logpdf of the target set given the
            context set under the true diagonalised GP.
    """

    def __init__(
        self,
        *args,
        kernel=stheno.EQ().stretch(0.25),
        pred_logpdf=True,
        pred_logpdf_diag=True,
        **kw_args,
    ):
        self.kernel = kernel
        self.pred_logpdf = pred_logpdf
        self.pred_logpdf_diag = pred_logpdf_diag
        super().__init__(*args, **kw_args)

    def generate_batch(self):
        """Generate a batch.

        Returns:
            dict: A batch, which is a dictionary with keys "xc", "yc", "xt", and "yt".
                Also possibly contains the keys "pred_logpdf" and "pred_logpdf_diag".
        """
        with B.on_device(self.device):
            set_batch, xcs, xc, nc, xts, xt, nt = new_batch(self, self.dim_y)
            import torch as th

            # xc = xc.type(th.float32)
            # xt = xt.type(th.float32)
            # self.noise = self.noise.type(th.float32)

            # If `self.h` is specified, then we create a multi-output GP. Otherwise, we
            # use a simple regular GP.
            import torch as th

            if self.h is None:
                # TODO: Simple way to ignore the todense warnings
                #   Should be made more specific later
                #   -> Does not work yet, fix later
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with stheno.Measure() as prior:
                        # self.kernel = sample_kernel(True)
                        # self.kernel = sample_kernel(single=False)
                        # Dutordoir2022 claims to sample lengthscales from
                        #   log N(log(0.5),sqrt(0.5)) but that makes no sense
                        #   so we sample from exp N(log(0.5),0.5) (the sqrt would work as well
                        #   but looks strange, we can change that later if we want to
                        self.kernel = stheno.Matern32().stretch(
                            np.exp(math.log(0.5) + math.sqrt(0.5) * np.random.randn())
                        )
                        # self.kernel = stheno.EQ()
                        # self.kernel = stheno.Matern32()
                        f = stheno.GP(self.kernel)
                        # Construct FDDs for the context and target points.
                        fc = f(xc, self.noise)
                        ft = f(xt, self.noise)
            else:
                raise NotImplementedError
                # with stheno.Measure() as prior:
                #     # Construct latent processes and initialise output processes.
                #     us = [stheno.GP(self.kernel) for _ in range(self.dim_y_latent)]
                #     fs = [0 for _ in range(self.dim_y)]
                #     # Perform matrix multiplication.
                #     for i in range(self.dim_y):
                #         for j in range(self.dim_y_latent):
                #             fs[i] = fs[i] + self.h[i, j] * us[j]
                #     # Finally, construct the multi-output GP.
                #     f = stheno.cross(*fs)
                #     # Construct FDDs for the context and target points.
                #     fc = f(
                #         tuple(fi(xci) for fi, xci in zip(fs, xcs)),
                #         self.noise,
                #     )
                #     ft = f(
                #         tuple(fi(xti) for fi, xti in zip(fs, xts)),
                #         self.noise,
                #     )

            # Sample context and target set.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.state, yc, yt = prior.sample(self.state, fc, ft)
            # print(yc.mean(), yc.std())
            # print(xc.shape)

            # Make the new batch.
            batch = {}
            set_batch(batch, yc, yt)

            # Compute predictive logpdfs.
            if self.pred_logpdf or self.pred_logpdf_diag:
                post = prior | (fc, yc)
            if self.pred_logpdf:
                batch["pred_logpdf"] = post(ft).logpdf(yt)
            if self.pred_logpdf_diag:
                batch["pred_logpdf_diag"] = post(ft).diagonalise().logpdf(yt)

            return batch
