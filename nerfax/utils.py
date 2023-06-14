import numpy as np
from Bio.Data.IUPACData import protein_letters_1to3
from jax import numpy as jnp

from nerfax.reconstruct import get_axis_matrix
from nerfax.foldcomp_constants import ATOM_ORDER

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

def build_mdtraj_top(seq):
    import pandas as pd
    aas_3letter = np.vectorize(lambda aa: protein_letters_1to3[aa].upper())(list(seq))

    dfs = []
    for i, aa in enumerate(aas_3letter):
        df = pd.DataFrame({'resSeq': i+1, 'resName': aa, 'name': ATOM_ORDER[aa]})
        dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)
    df['serial'] = df.index+1
    df['chainID'] = 0
    df['segmentID'] = ''
    df['element'] = df['name'].apply(lambda s: s[:1])
    df = df[['serial', 'name', 'element', 'resSeq', 'resName', 'chainID',
        'segmentID']]
    return df