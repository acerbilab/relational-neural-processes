# Tasks

#### Synthetic 1-2D regression using AR-CNP paper code (Ti, Daolang):

- Add RCNP model to the model set.
- Add RGNP (kvv) model to the model set where Cov(x, x') = K(phi(r(x)), phi(r(x'))) v(phi(r(x))) v(phi(r(x'))).
- Run the benchmark from the AR-CNP paper with RCNP and RGNP (kvv).

#### Synthetic "high" D regression using AR-CNP paper code :

- As above but we scale to $D>2$ (non-isotropic to $D=3-4$, maybe 5; isotropic to whatever we can manage, 20, 100?)
- We need to run other baselines: which ones?
- ConvCNPs et al. won't work above D=2
- Baselines: We can try CNP, GNP, AttnCNP, etc. - how will they work? (OOD not going to work, ID ? - it depends on training etc.)

#### Kernel and hyperparameter learning

- Similar to KITT in terms of training [here](https://arxiv.org/abs/2106.08185)
- Train on a "grammar" of kernels (base kernels plus linear combinations or products, random samples from hyperparameter priors)
- Show this at work on regression datasets or tasks like Bayesian optimization (see below)

#### Bayesian optimization

- See the diffusion process paper for an example
- BO in 3-6 D, train on a GP, test on some standard BO function (e.g., Hartmann, Ackley)
- Acquisition function? Thompson sampling (you need correlated samples), but for us it might be convenient just to use Expected improvement (either with gradient or even without)
- Also consider BO in higher dimension e.g. 20-100 (with isotropic process and an appropriate target).

#### Simulator example to directly predict the predictive distribution (sim2real) and then apply it to data

- Typically people peform inference over model parameters to get the posterior
- Sometimes then the posterior is used for posterior predictions ("fits" for existing data, and predictions for new data)
- Can we find interesting models for which this is useful? data -> posterior prediction (no intermediate posterior inference step)
- This is like learning an emulator of the simulator with marginalized model parameters (with no context = marginalize over the prior, etc.)
- See Lotka-Volterra example (sim2real)
- Some version of continous control of robot arm (double-pendulum, etc.)
- Crucially, we want to go above $D=1$ inputs

#### Example in which we showcase actual usage case of neural processes i.e. meta-learning (many small datasets)

- Think of something.

#### Figure out permutation invariance of input dimensions (Luigi, everyone)

- How do we implement that, if we can? (see [this](https://github.com/PrincetonLIPS/AHGP) as an example).
- Otherwise, simple approach is "data augmentation" (or summing over all permutations after applying the relational representation); this is impractical above $D \approx 4$ as it grows as $D!$.

#### RL benchmark (Kevin)

- Figure out some tasks for which GPs work, simple implementation
- Find out additional baseline besides GP (e.g, NN based ensemble)
- The idea is then that we can plug-in as a replacement our RCNP/RGNP
- Dataset with motor babbling, test set on trajectories that leave the initial state space
