FROM nvcr.io/nvidia/pytorch:22.03-py3
# RUN apt-get update && apt-get install -y git
# RUN git+https://github.com/username/nerfax.git#egg=nerfax[optional]
COPY . /nerfax
RUN pip install /nerfax[optional]
RUN pip install --upgrade "jax[cuda]==0.3.0" "jaxlib[cuda11_cudnn805]==0.3.0" -f https://storage.googleapis.com/jax-releases/jax_releases.html
