from typing import Any, Callable, Sequence, Optional
from functools import partial
import jax
from jax import jit, vmap, numpy as jnp, lax
from jaxlib.xla_extension import Device
from nerfax.host_callback_utils import call_jax_other_device

Array = jnp.ndarray

insert_zero = lambda x: jnp.concatenate([jnp.zeros((1,)+x.shape[1:], dtype=x.dtype), x])

def mp_nerf_jax(a,b,c,l,theta,chi):
    # Credit to https://github.com/EleutherAI/mp_nerf
    ba = b-a
    cb = c-b
    n_plane = jnp.cross(ba, cb)
    n_plane_ = jnp.cross(n_plane, cb)
    rotate = jnp.stack([cb, n_plane_, n_plane], axis=-1)
    rotate /= jnp.linalg.norm(rotate, axis=-2, keepdims=True)
    direction = jnp.array([
        jnp.cos(theta),
        jnp.sin(theta) * jnp.cos(chi),
        jnp.sin(theta) * jnp.sin(chi)
    ])
    return c + l * jnp.einsum('ji,i', rotate, direction)

normalise = lambda x: x/jnp.linalg.norm(x, axis=-1, keepdims=True)

def get_axis_matrix(a, b, c, norm=True):
    """ Gets an orthonomal basis as a matrix of [e1, e2, e3]. 
        Useful for constructing rotation matrices between planes
        according to the first answer here:
        https://math.stackexchange.com/questions/1876615/rotation-matrix-from-plane-a-to-b
        Inputs:
        * a: (batch, 3) or (3, ). point(s) of the plane
        * b: (batch, 3) or (3, ). point(s) of the plane
        * c: (batch, 3) or (3, ). point(s) of the plane
        Outputs: orthonomal basis as a matrix of [e1, e2, e3]. calculated as: 
            * e1_ = (c-b)
            * e2_proto = (b-a)
            * e3_ = e1_ ^ e2_proto
            * e2_ = e3_ ^ e1_
            * basis = normalize_by_vectors( [e1_, e2_, e3_] )
        Note: Could be done more by Grahm-Schmidt and extend to N-dimensions
              but this is faster and more intuitive for 3D.
    """
    v1_ = c - b 
    v2_ = b - a
    v3_ = jnp.cross(v1_, v2_)
    v2_ready = jnp.cross(v3_, v1_)
    basis    = jnp.stack([v1_, v2_ready, v3_], axis=-2)
    # normalize if needed
    if norm:
        return normalise(basis)
    return basis

def chain_sequential_rotations(rotations: Array) -> Array:
    length = rotations.shape[0]
    for i in range(1, length):
        rotations = rotations.at[i].set(jnp.matmul(rotations[i], rotations[i-1]))
    return rotations

def chain_sequential_rotations_scan(rotations: Array) -> Array:
    def _body(carry, x):
        rotations_im1 = carry
        rotations_i = x
        rotation = jnp.matmul(rotations_i, rotations_im1)
        return (rotation, rotation)
    _, rotations = jax.lax.scan(
        f=_body,
        init=jnp.eye(3),
        xs=rotations
    )
    return rotations

def associative_rotate_coords(prev, current):
    rotations_im1 = prev[...,0,:,:] # R_(i-1)
    # sub_rotation_i, pre_rotated_coords_i = current
    return vmap(jnp.matmul)(current, rotations_im1) # returns stacked [R_i, rotated_coords_i]

def work_inefficient_all_prefix_sum(operator, x):
    # Hillis, W. D. and Steele, G. L. (1986). Data parallel algorithms. Communications of the ACM, 29(12), 1170–1183
    # log_2{n} latency with sufficient parallelism, but total FLOPS scales with N
    n = x.shape[0]
    j_max = jnp.ceil(jnp.log2(n)).astype(int)
    
    l = jax.lax.slice_in_dim(x, 0   ,n-2**0)
    r = jax.lax.slice_in_dim(x, 2**0,n)
    for j in range(0,j_max-1):
        prev_l=l
        n = r.shape[0]
        r_ = operator(l, r)
        l_max = 2**j
        r_max = 2**(j+1)
        if r_max > n:
            l_max = l_max-r_max+n
            l = jax.lax.slice_in_dim(l, 0, l_max)
        else:
            l = jnp.concatenate([jax.lax.slice_in_dim(l, 0, 2**j), jax.lax.slice_in_dim(r_, 0, n-2**(j+1))])
        r = jax.lax.slice_in_dim(r_, 2**j, n)
    n = r.shape[0]
    final_r_ = operator(jax.lax.slice_in_dim(l, 0, n), r)
    return jnp.concatenate([jax.lax.slice_in_dim(prev_l, 0, 2**j), jax.lax.slice_in_dim(r_, 0, 2**j), final_r_])

# def reconstruct_from_internal_coordinates_pure_sequential(l, theta, chi, ref_pos):
#     def _body(a,b,c, length, angle, dihedral):
#         d = mp_nerf_jax(a, b, c, length, angle, dihedral)
#         return (b,c,d), (d,)

#     def reconstruct_sequential(lengths, angles, dihedrals):
#         _, (coords,) = jax.lax.scan(
#             lambda carry, x: _body(*carry, *x),
#             init=tuple(ref_pos),
#             xs=(lengths, angles, dihedrals),
#             unroll=4
#         )
#         return coords
#     return reconstruct_sequential(l, theta, chi)

def mp_nerf_jax_pretrig_self_normalising(ba, cb, c, length, sin_angle, cos_angle, sin_dihedral, cos_dihedral):
    # Credit to https://github.com/EleutherAI/mp_nerf
    n_plane = normalise(jnp.cross(ba, cb))
    n_plane_ = jnp.cross(n_plane, cb) # this is already normalised as cb is orthogonal to n_plane by construction
    rotate = jnp.stack([cb, n_plane_, n_plane], axis=-1)
    direction = jnp.array([
        cos_angle,
        sin_angle * cos_dihedral,
        sin_angle * sin_dihedral
    ])
    dc = (rotate@direction) # this is already normalised
    return c + length * dc, dc

expand = lambda x: (jnp.sin(x), jnp.cos(x)) # expand x in sin,cos basis
def reconstruct_from_internal_coordinates_pure_sequential(lengths, angles, dihedrals):
    '''
    We follow the simplifications in Parsons et al, Practical conversion from torsion space to Cartesian space for in silico protein synthesis, 10.1002/jcc.20237
    detailed as SN-NeRF
    Implementation here computes the sin and cos of angles and dihedrals externally to the scan so vectorised trig instructions can be used
    
    '''
    def _body(ba, cb, c, length, sin_angle, cos_angle, sin_dihedral, cos_dihedral):
        d, dc = mp_nerf_jax_pretrig_self_normalising(ba, cb, c, length, sin_angle, cos_angle, sin_dihedral, cos_dihedral)
        return (cb,dc, d), (d,)

    _, (coords,) = jax.lax.scan(
        lambda carry, x: _body(*carry, *x),
        init=(jnp.array([0.,1.,0.]), jnp.array([1.,0.,0.]), jnp.array([0.,0.,0.])),
        xs=(lengths, *expand(angles), *expand(dihedrals)),
        unroll=4
    )
    return coords

def reconstruct_from_internal_coordinates(l, theta, chi, mode='associative', device: Optional[Device]=None):
    '''
    Note: the first three dihedrals, the first two angles and the first length 
        are dummy values as they are relative to ghost atoms
    '''
    if mode=='fully_sequential':
        '''Short circuit where everything is sequential'''
        return reconstruct_from_internal_coordinates_pure_sequential(*jax.tree_map(lambda x: x.reshape(-1), (l, theta, chi)))

    ref_pos = jnp.array([
        [-1,-1,0],
        [-1, 0,0],
        [ 0, 0,0]
    ], dtype=float) # This choice makes the axis matrix identity so we can skip a matmul
    # Note in contrast to https://github.com/EleutherAI/mp_nerf this is always
    # a ghost residue rather than the first residue
    ghost_N, ghost_CA, ghost_C = ref_pos

    split = lambda x, axis=-1: jax.tree_map(lambda x: x.squeeze(axis), jnp.split(x, x.shape[axis], axis=axis))
    d = jax.tree_map(split, [l, theta, chi])
    N_data, CA_data, C_data = [[d[i][j] for i in range(3)] for j in range(3)] # reorder [type][atom] to [atom][type]

    # Place N
    N = vmap(mp_nerf_jax, in_axes=(None,None,None, 0,0,0))(ghost_N, ghost_CA, ghost_C, *N_data)

    # Place CA
    CA = vmap(mp_nerf_jax, in_axes=(None,None,0, 0,0,0))(ghost_CA,  ghost_C, N, *CA_data)

    # Place C
    C = vmap(mp_nerf_jax, in_axes=(None,0,0,0,0,0))(ghost_C, N, CA, *C_data)

    rotations = get_axis_matrix(N, CA, C, norm=True)[:-1] # N, CA, C
    pre_rotated_coords = jnp.stack([N, CA, C], axis=-2)

    assert mode in ['combo_associative', 'associative', 'sequential_slow', 'sequential', 'associative_work_inefficient']
    if mode != 'combo_associative':
        # do rotation concatenation
        if mode == 'associative':
            f = lambda rotations: lax.associative_scan(lambda x,y: jnp.matmul(y,x), rotations)
        elif mode == 'sequential_slow':
            f = chain_sequential_rotations
        elif mode == 'sequential':
            f = chain_sequential_rotations_scan
        elif mode == 'associative_work_inefficient':
            f = partial(work_inefficient_all_prefix_sum, lambda x,y: jnp.matmul(y,x))
        if device != None:
            # host call back, automatically uses the first cpu device
            rotations = call_jax_other_device(f, rotations, device=device)
        else:
            # None means place on the default device
            rotations = f(rotations)

        # rotate all
        coords = jnp.matmul(pre_rotated_coords[1:], rotations)
        # add on the identity point
        coords = jnp.concatenate([pre_rotated_coords[:1], coords])
    else:
        # We rotate the coords and compute the combined rotations together in one associative op
        f = lambda rotations, pre_rotated_coords: lax.associative_scan(associative_rotate_coords, jnp.stack([rotations, pre_rotated_coords], axis=1))
        coords = f(rotations, pre_rotated_coords)[:,1]

    # apply translational offset
    offset = coords[:-1,2].cumsum(0)
    offset = jnp.concatenate([jnp.zeros((1,3)), offset])
    coords += offset[...,None,:]
    return coords.reshape(-1,3)