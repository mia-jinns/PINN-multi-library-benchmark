# Instructions

## To run the notebooks

We resort to a conda environment where JAX and Pytorch can work alongside with first `pip install jax[cuda12]` and then `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`. 

Then we simply installed the libraries `jinns` `deepxde` and `mathlab-pina` via `pip`

## To run the NVIDIA Modulus scripts

We followed the docker tutorial available [here](https://docs.nvidia.com/deeplearning/modulus/getting-started/index.html#modulus-with-docker-image-recommended). Then `docker pull nvcr.io/nvidia/modulus/modulus:24.12` to download the image. Then to run the docker, contrary to what the instruction given on the website we need to execute `sudo docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v ${PWD}:/benchmark -it --rm nvcr.io/nvidia/modulus/modulus:24.12 bash` (because of newer docker version.
