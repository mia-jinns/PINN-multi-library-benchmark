# PINN multi-library benchmark

This repository compares several open-source Physics Informed Neural Network libraries on standard PDE problems both on forward and inverse problems. To the best of our knowledge it is the first PINN benchmark comparing together several libraries!

The following libraries are included:

- `jinns` [https://mia_jinns.gitlab.io/jinns/index.html](https://mia_jinns.gitlab.io/jinns/index.html)

- `deepXDE` [https://deepxde.readthedocs.io/en/latest/](https://deepxde.readthedocs.io/en/latest/)

- `PINA` [https://mathlab.github.io/PINA/](https://mathlab.github.io/PINA/)

- `Nvidia Modulus` [https://developer.nvidia.com/modulus](https://developer.nvidia.com/modulus)

The following PDE problems are included:

- From [PINNacle benchmark](https://arxiv.org/pdf/2306.08827): Burgers1D, 2D Navier-Stokes lid-driven flow, Poisson Inverse.

- From [deepXDE documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse.html): Diffusion Reaction inverse, Navier-Stokes inverse problem.


# Instructions

## To run the notebooks (`jinns`, `deepXDE`, `PINA`)

We resort to a conda environment where JAX and Pytorch can work alongside with first `pip install jax[cuda12]` and then `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`. 
Then you can install the three first libraries with `pip install jinns`, `pip install deepxde` and `pip install pina-mathlab`.

The versions of the relevant packages are:

- `jax==0.4.36` and `jaxlib==0.4.36`

- `equinox==0.11.10` and `optax==0.2.2`

- `torch==2.5.1+cu121`

- `jinns==1.2.0`

- `deepxde==1.12.1`

- `pina-mathlab==0.1.2.post2412` 


## To run the NVIDIA Modulus scripts (`NVIDIA modulus`)

We followed the docker tutorial available [here](https://docs.nvidia.com/deeplearning/modulus/getting-started/index.html#modulus-with-docker-image-recommended). Then `docker pull nvcr.io/nvidia/modulus/modulus:24.12` to download the image. Then to run the docker, contrary to what the instruction given on the website we need to execute `sudo docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v ${PWD}:/benchmark -it --rm nvcr.io/nvidia/modulus/modulus:24.12 bash` (because of newer docker version).


To correctly run all the scripts we need two corrections in the `modulus-sym` code source:

- Allow a constant learning rate by default. There seems to be a problem with scheduler default missing value. A simple modification of the `instantiate_sched` function in the file `modulus/sym/hydra/utils.py` can do the trick. For example:

```python
def instantiate_sched(
    cfg: DictConfig, optimizer: torch.optim
) -> torch.optim.lr_scheduler:
    try:
        # Function for instantiating a scheduler with hydra
        sched_cfg = copy.deepcopy(cfg.scheduler)

        # Default is no scheduler, so just make fixed LR
        if sched_cfg is MISSING:
            sched_cfg = {
                "_target_": "torch.optim.lr_scheduler.ConstantLR",
                "factor": 1.0,
            }
        # Handle custom cases
        if sched_cfg._target_ == "custom":
            if "tf.ExponentialLR" in sched_cfg._name_:
                sched_cfg = {
                    "_target_": "torch.optim.lr_scheduler.ExponentialLR",
                    "gamma": sched_cfg.decay_rate ** (1.0 / sched_cfg.decay_steps),
                }
            else:
                logger.warn("Detected unsupported custom scheduler", sched_cfg)
    except Exception as e:
        sched_cfg = {
            "_target_": "torch.optim.lr_scheduler.ConstantLR",
            "factor": 1.0,
        }

    try:
        scheduler = hydra.utils.instantiate(sched_cfg, optimizer=optimizer)
    except Exception as e:
        fail = colored("Failed to initialize scheduler: \n", "red")
        logger.error(fail + to_yaml(sched_cfg))
        raise Exception(fail) from e

    return scheduler
```

- Comment the call `var_to_polyvtk(save_var, filename)` in `modulus/sym/domain/constraint/continuous.py`.

# Results
See [results.md](results.md)

# Contributing

Contributions are very welcome! There are a lot of work that can be done to improve this repository. For example:

- Check for mistakes and enhancements: since no-one can have expert knowledge of all libraries and a few things could probably be improved. Be sure to improve the code of your favorite library.

- Update code: all the libraries will keep on being updated, hence the code here will need regular updates.

- Improve the benchmark coverage: more PDE problems, more libraries, more tests on different devices would be welcome. 
