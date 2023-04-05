# Experiments

## CNP paper ([Garnelo et al., 2018](https://arxiv.org/abs/1807.01613))

The experiments in this early paper are somewhat limited. Probably best to use examples from later papers which perform more comprehensive comparisons.

#### Experiments

1. **Function regression:** Synthetic 1D regression experiments with GPs.
2. **Image Completion:** MNIST and CelebA.
3. **One-shot classification:** Omniglot dataset (1,623 classes of characters from 50 different alphabets, 20 examples per class).


## ConvCNP paper ([Gordon et al., 2020](https://arxiv.org/abs/1910.13556))

Main competitor we need to compare against.

#### Metrics used in the experiments
- Predictive log-likelihood over multiple (e.g., 1000) test regression tasks (a task is a new dataset);
- Mean squared error (MSE).

#### Experiments

1. **Synthetic 1D experiments:** 1D regression problems with data sampled from GPs with different kernel functions
   - Regression problem: 1D input, 1D output.
   - Kernels: EQ (exponentiated quadratic), Weak periodic, Matern 5/2, Sawtooth. Kernel hyperparameters are fixed, not from a distribution.
   - Methods: CNP, AttnCNP, ConvCNP, ConvCNPXL (larger network/receptive field).
   - Comment: The presented metrics are for data inside the range. They also qualitatively show results for context points outside the range (showcasing that only translational-invariant models work).
2. **PLAsTiCC experiment:** Simulation of transients observed by the LSST telescope under realistic observational conditions ([here](https://www.kaggle.com/c/PLAsTiCC-2018)).
   - Regression problem: Six dimensional time series, treated as a multi-input multi-output problem.
   - Methods: Multi-input multi-output ConvCNP, GP from [Avocado](https://www.kaggle.com/c/PLAsTiCC-2018).
   - Comment: Might be too involved to set up.
3. **Predator-Prey models:** Train model on synthetic data from Lotka-Volterra model, condition on real data from Hudson's Bay lynx-hare data set ("Sim2Real").
   - Regression problem: 1D input (time), 2D output (predator and prey population size).
   - Methods: ConvCNP, AttnCNP (fails).
4. **2D image completion experiments:** Predict pixel intensities/color conditioned on a context set of other pixels.
   - Regression problem: 2D input, 1D output (greyscale) or 3D output (RGB).
   - Standard benchmarks: MNIST, SVHN, 32x32 and 64x64 CelebA.
   - Methods: AttnCNP, ConvCNP, ConvCNPXL (larger network/receptive field).
   - Additional tests: Generalization to multiple non-centered objects with *zero shot multi-MNIST* (ZSMM); CelebA celebrity photo.
   - Computational efficiency: ConvCNP is more efficient than AttnCNP (we may have trouble with this).

## GNP paper ([Markou et al., 2022](https://arxiv.org/abs/2203.08775))

#### Metrics used in the experiments
- Predictive log-likelihood over multiple (e.g., 1000) test regression tasks (a task is a new dataset);

#### Experiments

1. **Gaussian synthetic experiments:** 1D and 2D regression problems with data sampled from GPs.
   - Regression problems: 1D or 2D input, 1D output.
   - Kernels: EQ, Matern 5/2, Noisy mixture, Weakly periodic. 
   - Methods (1D): GP oracle, GP oracle (diag), GNP ($\times 3$), AGNP ($\times 3$), ConvGNP ($\times 3$), ANP, ConvNP, FullConvGNP.
   - Methods (2D): GP oracle, GP oracle (diag), ConvGNP ($\times 3$), ConvNP.
   - $\times 3$: Each of these methods above is implemented in three variants: meanfield, linear, kvv.
   - Comment: I am not sure if ConvNP is the same as ConvCNP.
2. **Predator-prey experiments:** Train on synthetic data from Lotka-Volterra model, condition on real data (see ConvCNP paper).
   - Regression problem: 1D input (time), 2D output (predator and prey population size).
   - Methods: ConvGNP ($\times 3$), ConvGCNP ($\times 3$).
   - ConvGCNP: A marginal CDF transformation of the output is introduced to enforce positivity of the output. I think the "C" in ConvGCNP doesn't stand for conditional (all GNPs are GCNPs, "Gaussian conditional neural processes"), but for CDF transformation of the marginal. The notation is not only super-confusing, but it's *also* used inconsistently in this same section (both "ConvCGNPs" and "ConvGCNPs" appear, *presumably* to mean the same thing).
3. **EEG experiments:** Real EEG dataset (7632 multivariate time series).
   - Regression problem: Multi-input, multi-output problem? To clarify.
   - Methods: ConvGNP ($\times 3$), ConvNP, MOGP (multi-output GP).
4. **Temperature downscaling for environmental modelling**. Multiple experiments with climate based data.
   - Regression problem: Unclear, some multi-input multi-output thing (perhaps 1D output, being temperature).
   - Methods: ConvGNP ($\times 3$), ConvNP, Value baseline (MAE - whatever MAE stands for).
   - Comment: This set of experiments is very complex and near-impossible to figure out the exact details just from the paper, even considering the appendices. In practice, we would need to read the other paper that focuses on this, [Vaughan et al. (2021)](https://arxiv.org/abs/2101.07950); probably not worth it unless the code is available and runs out of the box.  
_Should be the same as used in Bruinsma et al., in which case it is available in the repository they provide._


## AR-CNP paper ([Bruinsma et al., 2022](https://arxiv.org/abs/2303.14468))

This paper is a must-read as it systematizes the NP literature, with generally better definitions, notations, and descriptions of experiments and implementations (e.g., see Appendix F for the nice catalog of NP models).

#### Metrics used in the experiments
- Predictive log-likelihood over multiple (e.g., 1000) test regression tasks (a task is a new dataset);

#### Experiments

The experiments look the same as the GNP paper, with possibly a better explanation for both the EEG and the environmental modelling setup. The synthetic experiments are way more comprehensive, including more functions and methods. The code is here: https://github.com/wesselb/neuralprocesses

1. **Synthetically generated Gaussian and non-Gaussian data:** Twenty different regression tasks.
   - Regression problems: 1D or 2D input, 1D output.
   - Functions: Draws from GPs with different kernels (EQ, Matern 5/2, weakly periodic), plus non-Gaussian functions (sawtooth, mixture of GP + sawtooth).
   - Methods: ConvCNP, ConvCNP (AR), ConvGNP, FullConvGNP, ConvNLP (ML), ConvLNP (ELBO), GP (diag), trivial (1D normal with empirical mean and variance) in the main text; and many more in the appendix (pretty much the whole NP zoo).
2. **Predator-prey experiments.**
3. **EEG experiments.**
4. **Environmental modelling.**
