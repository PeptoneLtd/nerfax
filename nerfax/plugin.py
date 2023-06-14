from functools import partial
from jax import random, numpy as jnp, vmap, jit, grad
import jax
from nerfax.reconstruct import reconstruct_from_internal_coordinates, mp_nerf_jax
import numpy as np

roll_first_col_in_last_axis = lambda x, roll=1: jnp.concatenate([jnp.roll(x[...,:1], roll, axis=-2), x[...,1:]], axis=-1)
# Convert between scnet odd conventions and 'natural' ones. #FIXME make bijector.
def converter(input, roll):
    '''
    if a dict, it's assumed to have `angles_mask` key which is altered.
    if an array, assumed to be the `angles_mask` array in mp_nerf to alter
    '''
    def _converter(angles_mask):
        # Roll the first angle and dihedral round by 1 to match 'natural' syntax
        angles_mask = roll_first_col_in_last_axis(angles_mask, roll=roll)

        # Fix difference in how angle is specified
        angles, torsions = angles_mask
        angles = jnp.pi-angles # due to historical scnet reasons, the scnet angle is defined as pi-angle
        angles_mask = jnp.stack([angles, torsions])
        return angles_mask
    if isinstance(input, dict):
        # is scaffolds dict, where we fix angles mask
        return {**input, 'angles_mask': _converter(input['angles_mask'])}
    else:
        # Is angles_mask tensor of shape (2,L,14)
        return _converter(input)
convert_scnet_to_natural = partial(converter, roll=1)
convert_natural_to_scnet = partial(converter, roll=-1)

def protein_fold(cloud_mask, point_ref_mask, angles_mask, bond_mask, only_backbone=False, reconstruct_fn=reconstruct_from_internal_coordinates):
    bb_coords = reconstruct_fn(bond_mask[...,:3], *angles_mask[...,:3]).reshape(-1,3,3)
    if only_backbone: return bb_coords;
    n = bb_coords.shape[0]
    coords = jnp.concatenate([bb_coords, jnp.zeros((n,11,3))], axis=-2)

    ########
    # parallel sidechain - do the oxygen, c-beta and side chain
    ########

    '''
    Fix indexing for residue 0 C-beta. Normally for CB(i) is placed C(i-1)-N(i)-CA(i)->CB(i)
    but first residue is N(1)-C(0)-CA(0)->CB(0) as C(-1) does not exist. The point ref mask here
    is current incorrect so we fix numbering here. indexing i is dealt with later'''
    point_ref_mask = point_ref_mask.at[:,0,1].set(jnp.array([0,2,1])) # [0,2,1] == N, C, CA

    for i in range(3,14):
        level_mask = cloud_mask[:, i]
        thetas, dihedrals = angles_mask[:, level_mask, i]
        idx_a, idx_b, idx_c = point_ref_mask[:, level_mask, i-3]

        # to place C-beta, we need the carbons from prev res - not available for the 1st res
        if i == 4:
            # the C requested is from the previous residue, so we shift all by 1. Expect special case for 0.
            i_ = level_mask.nonzero()[0]-1
            # if first residue is not glycine, 
            # for first residue, use position of the second residue's N
            if level_mask[0].item():
                i_ = i_.at[0].set(1)
            coords_a = coords[i_, idx_a]
        else:
            coords_a = coords[level_mask, idx_a]
        coords = coords.at[level_mask, i].set(vmap(mp_nerf_jax)(coords_a, 
                                              coords[level_mask, idx_b],
                                              coords[level_mask, idx_c],
                                              bond_mask[level_mask, i], 
                                              thetas, dihedrals)
                                    )
    return coords

def get_jax_protein_fold(scaffolds, only_backbone=False, reconstruct_fn=reconstruct_from_internal_coordinates):
    '''
        Takes scaffolds, a dict of torch tensors and returns a 
        jit-compiled op converting internal->cartesian coords
        for that specific protein
    '''
    cloud_mask, point_ref_mask= [(jnp.array(np.array(x)) if not isinstance(x, jax.Array) else x) for x in map(scaffolds.__getitem__, ['cloud_mask', 'point_ref_mask'])]
    def _fold(angles_mask, bond_mask, only_backbone=only_backbone):
        with jax.ensure_compile_time_eval():
            return protein_fold(cloud_mask, point_ref_mask, angles_mask, bond_mask, only_backbone=only_backbone, reconstruct_fn=reconstruct_fn)
    return _fold