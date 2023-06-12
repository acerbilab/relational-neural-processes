# Practical Equivariances via Relational Conditional Neural Processes

_____
**Note:** Our implementations build on the [neuralprocesses](http://github.com/wesselb/neuralprocesses) library by Wessel Bruinsma. 
You can find further details on its structure and the dependencies at the github website.

_______

## Installation
```
git clone https://github.com/acerbilab/relational-neural-processes.git

cd relational-neural-processes

conda create --name np python=3.10.10

conda activate np

pip install -r requirements.txt
```


To use our RCNP models, you can specify `--model` as one of `[rcnp, rgnp, fullrcnp, fullrgnp]`, where the latter two represents the full version of our models. You can also choose comparison function for relational encoding by using `--comparison-function`, where `difference` stands for encoding translational equivariance, and `distance` stands for encoding rigid transformations.

We also support partial encoding, where some dimensions of the data do not need relational encoding, you can specify the dimensions that do not require relational encoding through `--non-equivariant-dim`, e.g., `--non-equivariant-dim=“0,1”` means the first and the second dimension will not be encoded by relational encoder.


## Synthetic Regression
To run the synthetic regression experiment, you can use the following template command lines:
```angular2html
python train.py \
    --model=“rcnp” \
    --data=“eq” \
    --dim-x=2 \
    --dim-y=1 \
    --seed=1 \
    --comparison-function=“difference”
```
For the above template command, we use RCNP model with `difference` comparison function on x2_y1 eq data.


## Bayesian Optimization
To run the Bayesian optimization experiment first run 
```
python bayesopt/bo_train.py \
    --target=hartmann3d \
    --exp="bo_fixed" \
    --model="rcnp"
 ```
to pretrain a model with the desired set of kernels. Here, `--target` is one of `[hartmann3d, hartmann6d]`, `--exp` is one of `[bo_fixed, bo_matern, bo_single, bo_sumprod]` corresponding to scenario (i)-(iv) in the paper. `--model` is one of `[cnp, gnp, acnp, agnp, rcnp, rgnp]`.

Once trained, apply it to the BO task as
```
python bayesopt/bo_apply.py \
    --exp="bo_fixed" \
    --target_name="hartmann3d" \
    --n_rep=10 \
    --n_steps=50
 ```
where `--exp` and `--target_name` as above and `--n_rep`, `--n_steps` specify the number of replications and the number of query steps respectively.


## Lotka-Volterra Model
For the Lotka-Volterra model experiments, we used the "predprey" experiment available in the original package.
To rerun our CNP experiments, use
```
python train.py \
    --data="predprey" \ 
    --seed=x \ 
    --model=m \
    --enc-same
```

with x in `(1, 10)` and m in `(rcnp, cnp, acnp, convcnp)`.

To rerun our GNP experiments, use
```
python train.py \
    --data="predprey" \
    --seed=x \
    --model=m \
    --enc-same \
    --num-basis-functions=32
```
with x in `(1, 10)` and m in `(rgnp, gnp, agnp, convgnp)`



## Reaction-Diffusion Model
For the Reaction-Diffusion example, all the parameters selected for the experiments are listed in `neuralprocesses/experiments/data/cancer.py`. 

The dataset used from the Reaction-Diffusion simulator is stored in the `neuralprocesses/experiments/data/dataset_cancer` folder, along with the python file that produces simulations (`RD-simulator.py`).
To run experiments, use, with x the seed chosen and m the method chosen `(rcnp, rgnp, cnp, gnp, acnp, agnp)`:
```
python train.py \
    --data="cancer" \
    --seed=x \
    --model=m
```
For the additional study, run:
```
python train.py \
    --data="cancer_latent" \
    --seed=x \
    --model=m \ 
    --non-equivariant-dim="3" \
    --comparison-function="difference"
```
with x in `(1, 10)` and m in `(rgnp, gnp, agnp, convgnp)`

## Customize your own comparison function
Our repository provide an easy way to customize your own comparison function. You can define your own comparison function in `comparison_functions.py`, and then use it in `train.py` by specifying `--comparison-function=“your_comparison_function”`.


## Citation
```
@article{huang2023practical,
  title={Practical Equivariances via Relational Conditional Neural Processes},
  author={Huang, Daolang and Haussmann, Manuel and Remes, Ulpu and John, ST and Clart{\'e}, Gr{\'e}goire and Luck, Kevin Sebastian and Kaski, Samuel and Acerbi, Luigi},
  journal={arXiv preprint arXiv:2306.10915},
  year={2023}
}
```

## License
This code is under the MIT License.



