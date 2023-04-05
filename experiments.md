# Experiments

## CNP paper ([Garnelo et al., 2018](https://arxiv.org/abs/1807.01613))

The experiments in this paper are somewhat limited. Probably best to use examples from later papers which perform more comprehensive comparisons.

#### Experiments

1. **Function regression:** Synthetic 1D regression experiments with GPs.
2. **Image Completion:** MNIST and CelebA.
3. **One-shot classification:** Omniglot dataset (1,623 classes of characters from 50 different alphabets, 20 examples per class).


## ConvCNP paper ([Gordon et al., 2020](https://arxiv.org/abs/1910.13556))

#### Metrics used in the experiments
- Log-likelihood over multiple (e.g., 1000) test regression tasks (a task is a new dataset);
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


   
