import warnings

import lab as B
import stheno

from .data import SyntheticGenerator, new_batch
from .kernelgrammar_stheno import sample_kernel, sample_basic_kernel

__all__ = ["GPGeneratorSample"]


class GPGeneratorSample(SyntheticGenerator):
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
        kernel_struct="fixed",
        kernel=None,
        pred_logpdf=True,
        pred_logpdf_diag=True,
        **kw_args,
    ):
        assert kernel is None, "Kernel argument is irrelevant"
        self.kernel_struct = kernel_struct
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

            if self.h is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with stheno.Measure() as prior:
                        if self.kernel_struct == "fixed":
                            self.kernel = stheno.Matern52()
                        elif self.kernel_struct == "matern":
                            self.kernel = sample_basic_kernel("matern52", scale=True)
                        elif self.kernel_struct == "single":
                            self.kernel = sample_kernel(True)
                        elif self.kernel_struct == "sumprod":
                            self.kernel = sample_kernel(False)
                        f = stheno.GP(self.kernel)
                        # Construct FDDs for the context and target points.
                        fc = f(xc, self.noise)
                        ft = f(xt, self.noise)
            else:
                raise NotImplementedError

            # Sample context and target set.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.state, yc, yt = prior.sample(self.state, fc, ft)

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
