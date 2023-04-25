# An adaptation of Wessel's GP generator using GPytorch
# For now primarily a hacky process to get things started
import lab as B
import stheno
import gpytorch as gpy
import torch as th
import numpy as np

from .data import SyntheticGenerator, new_batch

from .kernelgrammar import sample_kernel

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
        kernel=None,
        noise=0.05**2,
        pred_logpdf=False,
        pred_logpdf_diag=False,
        verbose=False,
        **kw_args,
    ):
        if verbose and kernel is not None:
            print("WARNING: The chosen kernel will be ignored below")
        self.kernel = kernel
        # TODO: Change that
        if verbose and (pred_logpdf or pred_logpdf_diag):
            print("WARNING: pred_logpdf is not yet available")
        self.pred_logpdf = False
        self.pred_logpdf_diag = False
        super().__init__(*args, **kw_args)
        self.noise = noise

    def generate_batch(self):
        """Generate a batch.

        Returns:
            dict: A batch, which is a dictionary with keys "xc", "yc", "xt", and "yt".
                Also possibly contains the keys "pred_logpdf" and "pred_logpdf_diag".
        """
        with B.on_device(self.device):
            # COMMENT MH: Xcs, Xts are only necessary for multioutput
            set_batch, xcs, xc, nc, xts, xt, nt = new_batch(self, self.dim_y)

            # For now we do not rely on stheno
            # # If `self.h` is specified, then we create a multi-output GP. Otherwise, we
            # # use a simple regular GP.
            # if self.h is None:
            #     with stheno.Measure() as prior:
            #         f = stheno.GP(self.kernel)
            #         # Construct FDDs for the context and target points.
            #         # COMMENT MH: These are FDD objects
            #         fc = f(xc, self.noise)
            #         ft = f(xt, self.noise)
            # else:
            #     raise NotImplementedError("Focus for now on scalar output")
            #     with stheno.Measure() as prior:
            #         # Construct latent processes and initialise output processes.
            #         us = [stheno.GP(self.kernel) for _ in range(self.dim_y_latent)]
            #         fs = [0 for _ in range(self.dim_y)]
            #         # Perform matrix multiplication.
            #         for i in range(self.dim_y):
            #             for j in range(self.dim_y_latent):
            #                 fs[i] = fs[i] + self.h[i, j] * us[j]
            #         # Finally, construct the multi-output GP.
            #         f = stheno.cross(*fs)
            #         # Construct FDDs for the context and target points.
            #         fc = f(
            #             tuple(fi(xci) for fi, xci in zip(fs, xcs)),
            #             self.noise,
            #         )
            #         ft = f(
            #             tuple(fi(xti) for fi, xti in zip(fs, xts)),
            #             self.noise,
            #         )
            #
            # # Sample context and target set.
            # # self.state, yc, yt = prior.sample(self.state, fc, ft)
            # COMMENT MH: We ignore the state for now

            # TODO: This is currently full of messy tricks, fix them
            self.kernel = sample_kernel(single=True)
            # The noise gets turned in an array somewhere in the library
            #   and I currently don't have the nerve to check where
            if self.noise is not None:
                self.noise = np.float64(self.noise)
            xc = xc.float()
            xt = xt.float()

            yc = (
                gpy.distributions.MultivariateNormal(
                    th.zeros_like(xc.squeeze(-1)), self.kernel(xc)
                )
                .sample()
                .unsqueeze(-1)
            )
            yt = (
                gpy.distributions.MultivariateNormal(
                    th.zeros_like(xt.squeeze(-1)), self.kernel(xt)
                )
                .sample()
                .unsqueeze(-1)
            )
            yc = (yc + self.noise * th.randn_like(yc)).numpy()
            yt = (yt + self.noise * th.randn_like(yt)).numpy()

            # Make the new batch.
            batch = {}
            set_batch(batch, yc, yt)
            # set_batch(batch, yc.numpy(), yt.numpy())

            # # Compute predictive logpdfs.
            # if self.pred_logpdf or self.pred_logpdf_diag:
            #     post = prior | (fc, yc)
            # if self.pred_logpdf:
            #     batch["pred_logpdf"] = post(ft).logpdf(yt)
            # if self.pred_logpdf_diag:
            #     batch["pred_logpdf_diag"] = post(ft).diagonalise().logpdf(yt)

            return batch
