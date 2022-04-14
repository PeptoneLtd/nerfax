from nerfax.reconstruct import get_axis_matrix
from jax import numpy as jnp

def get_align_rigid_bodies_fn(mobile, ref):
    # We align to a reference frame defined by unit matrix and the first atom at the origin
    # As rigid body we just need the first 3 atoms of each to align
    mobile = mobile[:3]
    ref = ref[:3]

    ref_translation = ref[0]
    mobile_translation = mobile[0]
    mobile_rot = get_axis_matrix(*(mobile-mobile_translation))
    ref_rot = get_axis_matrix(*(ref-ref_translation))

    rotation = jnp.matmul(jnp.linalg.inv(mobile_rot), ref_rot)
    translation = ref_translation - jnp.matmul(mobile_translation, rotation)
    return lambda x: jnp.matmul(x, rotation) + translation