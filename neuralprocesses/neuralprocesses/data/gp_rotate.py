import lab as B
import stheno
import torch

from .data import SyntheticGenerator, new_batch

__all__ = ["GPGeneratorRotate"]


class GPGeneratorRotate(SyntheticGenerator):
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
        pred_logpdf=False,
        pred_logpdf_diag=False,
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
            

            # If `self.h` is specified, then we create a multi-output GP. Otherwise, we
            # use a simple regular GP.
            if self.h is None:
                with stheno.Measure() as prior:
                    f = stheno.GP(self.kernel)
                    # Construct FDDs for the context and target points.
                    fc = f(xc, self.noise)
                    ft = f(xt, self.noise)
            else:
                with stheno.Measure() as prior:
                    # Construct latent processes and initialise output processes.
                    us = [stheno.GP(self.kernel) for _ in range(self.dim_y_latent)]
                    fs = [0 for _ in range(self.dim_y)]
                    # Perform matrix multiplication.
                    for i in range(self.dim_y):
                        for j in range(self.dim_y_latent):
                            fs[i] = fs[i] + self.h[i, j] * us[j]
                    # Finally, construct the multi-output GP.
                    f = stheno.cross(*fs)
                    # Construct FDDs for the context and target points.
                    fc = f(
                        tuple(fi(xci) for fi, xci in zip(fs, xcs)),
                        self.noise,
                    )
                    ft = f(
                        tuple(fi(xti) for fi, xti in zip(fs, xts)),
                        self.noise,
                    )

            dim_x = xc.shape[-1]
            normal_sampler = stheno.Normal(1)

            rotate = B.cast(xc.dtype, B.zeros(dim_x, dim_x))

            rotate[0, :] = B.cast(xc.dtype, normal_sampler.sample(dim_x))
            rotate[0, :] = rotate[0, :] / B.sqrt(B.sum(rotate[0, :] ** 2))
            for j in range(1, dim_x):
                x_temp = B.cast(xc.dtype, normal_sampler.sample(dim_x))
                for jj in range(0, j):
                    x_temp = x_temp - rotate[jj, :] * B.sum(x_temp * rotate[jj, :]) / B.sum(rotate[jj, :] ** 2)
                x_temp = x_temp / B.sqrt(B.sum(x_temp ** 2))
                rotate[j, :] = x_temp
            #rotate = B.matmul(rotate, rotate)
            if torch.det(rotate) < 0:
                rotate[0,:]=-rotate[0,:]

            xc_rotate = B.matmul(xc, rotate)
            xt_rotate = B.matmul(xt, rotate)

            # Sample context and target set.
            self.state, yc_temp, yt_temp = prior.sample(self.state, fc, ft)
            mean_function_length_scale = B.cast(xc.dtype, B.linspace(0.5, 2, dim_x))

            yc = yc_temp + torch.sum(xc_rotate.unsqueeze(2) ** 2 / mean_function_length_scale ** 2, axis=-1)
            yt = yt_temp + torch.sum(xt_rotate.unsqueeze(2) ** 2 / mean_function_length_scale ** 2, axis=-1)

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
