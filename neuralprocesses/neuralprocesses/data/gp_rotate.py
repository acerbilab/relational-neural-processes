import lab as B
import stheno
import torch

from .data import SyntheticGenerator, new_batch

__all__ = ["GPGeneratorRotate"]

"""

def transform_to_euclidean(x):
        r=torch.zeros_like(x)
        r[:,0,:]=x[:,0,:]*torch.sin(x[:,1,:])*torch.cos(x[:,2,:])
        r[:,1,:]=x[:,0,:]*torch.sin(x[:,1,:])*torch.sin(x[:,2,:])
        r[:,2,:]=x[:,0,:]*torch.cos(x[:,1,:])
        return r
        
def transform_to_spherical(x):
        r=torch.zeros_like(x)
        r[:,0,:]=torch.sqrt(torch.sum(x**2,axis=-2))
        r[:,1,:]=torch.arccos(x[:,2,:]/r[:,0,:])/2*np.pi
        r[:,2,:]=torch.sign(x[:,1,:])*torch.arccos(xc[:,0,:]/r[:,0,:])/np.pi
        return r

"""


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

            # dim_x = xc.shape[-1]
            #
            # M = torch.randn(dim_x, dim_x)
            # rotate, _ = torch.linalg.qr(M)
            #
            # if torch.det(rotate) < 0:
            #     rotate[:, 0] = -rotate[:, 0]
            #
            # rotate = B.cast(xc.dtype, rotate).to(self.device)
            #
            # xc_rotate = B.matmul(xc, rotate)
            # xt_rotate = B.matmul(xt, rotate)

            # Sample context and target set.
            self.state, yc_temp, yt_temp = prior.sample(self.state, fc, ft)
            
            
            dimx=xc.shape[-1]
            normalsampler=Normal(1)
            #sample a random rotation:
            #Idea: produce a random base of the space. For that, we take random vectors and apply gram schmidt. I think this law is uniform over the bases, as we are always sampling uniformly on the sphere and then projecting everything on the subspace generated by previous vectors. I don't think it's very important.
            X=torch.zeros(dimx,dimx,dtype=torch.float64)#there seems to be some issues with the types. So I just added a bunch of them
            X[0,:]=torch.from_numpy(normalsampler.sample(dimx).astype(float))#the first vector is just normal, I'm using the normal distribution from the package to prevent compatbility issues.
            X[0,:]=X[0,:]/B.sqrt(torch.sum(X[0,:]**2))
            for j in range(1,dimx):
                Xtemp=torch.from_numpy(normalsampler.sample(dimx).astype(float))
                for jj in range(0,j):
                    Xtemp=Xtemp-X[jj,:]*B.sum(Xtemp*X[jj,:])/B.sum(X[jj,:]**2)#gram schmidt
                Xtemp=Xtemp/B.sum(Xtemp**2)#renormalise
                X[j,:]=Xtemp
            #print(xc.shape)
            #print(xc[0,0,0])
            #print(X[0,0])
            xc_rotate=torch.matmul(xc,X)#we rotate the x to apply the function, for whatever reason, this works. Maybe the true rotation matrix is not X but X^t=X^-1, but in this case we just have a definition of the distribution over X^t, it's not important imo, but we can use transpose on X.
            xt_rotate=torch.matmul(xt,X)#we rotate the x to apply the function
            #in the end, the matrix X is a random rotation matrix
            self.state, yc_temp, yt_temp = prior.sample(self.state, fc, ft)
            if dimx==2:
                factor=[4,0.5]
            elif dimx=3:
                factor=[4,0.5,2]
            elif dimx>3:
                if dimx % 2 ==0:
                    factor=B.concat[torch.tensor([4,0.5,2,1]),torch.tensor([0.3,3]).repeat((dimx-4)/2),0)
                else:
                    factor=B.concat[torch.tensor([4,0.5,2]),torch.tensor([0.3,3]).repeat((dimx-3)/2),0)
            yc = yc_temp + 3*torch.sin(torch.sum(xc_rotate.unsqueeze(2)**2/factor,axis=-2)/3)
            yt = yt_temp + 3*torch.sin(torch.sum(xt_rotate.unsqueeze(2)**2/factor,axis=-2)/3)
            
            #yc = yc_temp + 3*torch.sin(torch.sum(xc.unsqueeze(2)**2,axis=-2)/3)
            #yt = yt_temp + 3*torch.sin(torch.sum(xt.unsqueeze(2)**2,axis=-2)/3)
            
            #yc = yc_temp + torch.sum(xc.unsqueeze(2)**2,axis=-2)**2/2
            #yt = yt_temp + torch.sum(xt.unsqueeze(2)**2,axis=-2)**2/2

            # Make the new batch.
            batch = {}
            set_batch(batch, yc, yt)

            # Compute predictive logpdfs.
            if self.pred_logpdf or self.pred_logpdf_diag:
                post = prior | (fc, yc_temp)
            if self.pred_logpdf:
                batch["pred_logpdf"] = post(ft).logpdf(yt_temp)
            if self.pred_logpdf_diag:
                batch["pred_logpdf_diag"] = post(ft).diagonalise().logpdf(yt_temp)

            return batch


