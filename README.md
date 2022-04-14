# NeRFax
<img src="https://github.com/PeptoneInc/nerfax/blob/main/data/NeRFax-cover.jpg" width="801" height="390">   
In this work we implement NeRF and, to the best of our knowledge, the first fully parallel implementation of pNeRF in an emerging framework, JAX. We demonstrate speedups in the range 35-175x in comparison to the fastest public implementation for single chain proteins and utilising the frameworks ability to trivially parallelise functions we show a >10,000x speedup relative to using mp-NeRF serially for a biomolecular condensate of 1,000 chains of 163 residues.

# Benchmarks
## Single chain
### Runtime of different computational methods for single chains  
<img src="https://github.com/PeptoneInc/nerfax/blob/main/data/speedup.png" width="300" height="200">     

### Speedup, relative to the CPU mp_nerf implementation, of different computational methods for single chains  
<img src="https://github.com/PeptoneInc/nerfax/blob/main/data/timings.png" width="300" height="200">   

This can be reproduced with `notebooks/benchmark_single_chain_reconstruction.ipynb`.
## Multiple chains: Biomolecular condensate reconstruction
Leveraging the automatic vectorization feature of JAX the reconstruction was parallelized, running in **3.4 ms** on GPU. Extrapolation of the torch implementation gives **~60 seconds** in previous implementations, approximately 17,000x faster as the torch has no parallel chain implementation so has to be computed serially. This can be reproduced with `notebooks/benchmark_multiple_chain_reconstruction.ipynb`.

# Installation
## Pip
```bash
git clone https://github.com/PeptoneInc/nerfax.git && pip install ./nerfax[optional]
```
Note: for running on GPU, a GPU version of JAX must be installed, please follows the instructions at [JAX GPU compatibility instructions](https://github.com/google/jax#pip-installation-gpu-cuda)
## Docker image
We also provide a Dockerfile which can be used to install NerFax. The dockerfile includes the GPU version of JAX.








<!-- # Improvements on mp_nerf
- mp_nerf in pytorch places each residue relative to a non unit reference frame, this adds an extra matrix-matrix multiplication relative to SMP-NeRF which places relative to the unit reference frame.
- Most of the NeRF algorithm is trivially parallel, except the calculation of rotation matrices and translational offset which relate each residue to the previous. MP_NeRF implements the rotation matrix calculation as a for loop, causing O(N) runtime. SMP_NeRF leverages the fact matrix-matrix multiplication is associative A(BC)==(AB)C so an associative scan, a simple primitive in JAX, can be used which scales with O(log(N)).
- JAX has convenient just-in-time compilation of functions, accelerating evaluation
- The atoms are placed N->CA->C rather than CA->C->N. This means the N of the first residue is placed, which is absent in mp_nerf
- The CB of the first residue is incorrectly placed in mp_nerf, this bug is fixed here.
 -->
