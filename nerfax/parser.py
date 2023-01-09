import numpy as np
import mdtraj as md
from Bio.Data import IUPACData
from collections import defaultdict
import mp_nerf
from jax import vmap, numpy as jnp, lax

AA = set(mp_nerf.kb_proteins.SC_BUILD_INFO.keys())-set('_')
aa_to_index = lambda seq: np.vectorize({v:k for k,v in enumerate(AA)}.__getitem__)(list(seq))


def calculate_angle(c1, c2, c3):
    # Credit to https://github.com/EleutherAI/mp_nerf
    u1 = c2 - c1
    u2 = c3 - c2
    # dont use acos since norms involved. 
    # better use atan2 formula: atan2(cross, dot) from here: 
    # https://johnblackburne.blogspot.com/2012/05/angle-between-two-3d-vectors.html
    return jnp.arctan2( jnp.linalg.norm(jnp.cross(u1,u2)), 
                        jnp.dot(u1,u2) ) 

@vmap
def compute_dihedrals(xyz):
    # assert xyz.shape == (4,3) 
    d = xyz[1:]-xyz[:-1]
    index = jnp.arange(3)
    cross_pairs = jnp.stack([index[:-1], index[1:]])
    c = jnp.cross(*d[cross_pairs]) # c is the cross products of consecutive displacements
    p1 = jnp.einsum('i,i', d[0],c[1]) * jnp.linalg.norm(d[1])
    p2 = jnp.einsum('i,i', c[0],c[1])
    return jnp.arctan2(p1, p2)

@vmap
def decompose_quad(xyz):
    a,b,c, real_pos = xyz
    l = jnp.linalg.norm(real_pos-c)
    theta = calculate_angle(b,c,real_pos)
    chi = compute_dihedrals(jnp.array([[a,b,c,real_pos]]))[0]
    return l, theta, chi

def xyz_to_internal_coords(xyz):
    ''' Expects just the backbone'''
    dih_quads = jnp.stack([lax.dynamic_slice(xyz, (i,0), (xyz.shape[0]-3,3)) for i in jnp.arange(4)], axis=-2)
    l, theta, chi = decompose_quad(dih_quads)
    theta, chi, l = [a.reshape((-1,3)) for a in [theta,chi,l]]
    return l, theta, chi # lengths, angles, torsions

def get_scnet_loader_fns(t):
    '''
    Requires ALL heavy sidechain atoms to be present
    Convert positions from pdb to (L,14,3) format
    '''
    sc_atomnames = {k:'N CA C O'.split()+v['atom-names'] for k,v in mp_nerf.kb_proteins.SC_BUILD_INFO.items() if k!='_'}
    sc_net_coord_dict = {f'{aa}-{atomname}':i for aa, atomnames in sc_atomnames.items() for i, atomname in enumerate(atomnames)}

    d = t.top.to_dataframe()[0]
    d['resSeq'] -= (d.resSeq.min()-1)
    onecode = {k.upper(): v for k, v in IUPACData.protein_letters_3to1_extended.items()}
    d['aa'] = np.vectorize(onecode.__getitem__)(d.resName)
    atom_idxs = np.vectorize(defaultdict(lambda: None, sc_net_coord_dict).__getitem__, otypes=[object])(d.aa+'-'+d.name)

    mask = atom_idxs!=None
    res_idxs = d.resSeq.values-1
    def _parse_coords(coords):
        '''Converts from mdtraj coords to sidechainnet (L,14,3)'''
        return jnp.zeros((res_idxs.max()+1, 14, 3)).at[(res_idxs[mask], atom_idxs[mask].astype(int))].set(coords[mask]) * 10 # account for nm->Angstrom unit conversion
    
    def _restrict_to_scnet_atoms(t):
        '''takes mdtraj traj with all atoms and returns ones with just ones in scnet representation'''
        return t.atom_slice(np.where(mask)[0])
    
    def _scnet_to_list(coords):
        '''Makes mdtraj format (N, 3) coords from scnet coords (L,14,3)'''
        return coords[(res_idxs[mask], atom_idxs[mask].astype(int))]/10
    return _parse_coords, _restrict_to_scnet_atoms, _scnet_to_list

def load_traj(path):
    t = md.load(path)
    t = t.restrict_atoms(t.top.select('protein and chainid 0'))
    return t

def load_to_sc_coord_format(path, first_frame_only=True):
    t = load_traj(path)
    _parse_coords, _, _ =  get_scnet_loader_fns(t)
    coords = _parse_coords(t.xyz[0]) if first_frame_only else vmap(_parse_coords)(t.xyz)
    seq = t.top.to_fasta()[0]
    return coords, seq

def get_point_ref_with_final(aa):
    ref_indices = mp_nerf.make_idx_mask(aa).astype(int)
    final_index = np.arange(11)+3
    mask = mp_nerf.make_cloud_mask(aa)[3:].astype(bool)
    final_index[~mask]=0
    return np.concatenate([ref_indices, final_index[...,None]], axis=-1)

def get_point_ref_and_cloud_mask(seq):
    # point ref here (4,L,11) differs from mp_nerf point_ref_mask (3,L,11) as it
    # includes the final index explicitly so is 4 long, not 3
    point_ref_per_aa = np.stack(list(map(get_point_ref_with_final,AA))).astype(int)
    point_ref = jnp.einsum('ijk->kij', point_ref_per_aa[aa_to_index(seq)])

    cloud_mask_per_aa = jnp.stack(list(map(mp_nerf.make_cloud_mask,AA))).astype(bool)
    cloud_mask = cloud_mask_per_aa[aa_to_index(seq)]
    return point_ref, cloud_mask

def get_data_masks(coords, point_ref):
    '''
    Note: outputs angles_mask in the natural format, not scnet format
    '''
    # We force [2,1,0,4] reference for CB to be sure to pull out all CB relevant positions
    # as we will use atoms from the previous residue here
    point_ref_mod = point_ref.at[:,:,1].set(jnp.array([2,0,1,4])[...,None])
    
    # coords - (L,14,3). Point_ref (4,L,11) [note, 4 in first axis, not 3 as in point_ref_mask as we use the placed position index here]
    ref_coords = vmap(lambda x,y: x[y], in_axes=(0,1))(coords, point_ref_mod).swapaxes(1,2)

    '''
    Fix indexing for residue 0 C-beta. Normally for CB(i) is placed C(i-1)-N(i)-CA(i)->CB(i)
    but first residue is N(1)-C(0)-CA(0)->CB(0) as C(-1) does not exist. The point ref mask here
    is current incorrect so we fix numbering here. indexing i is dealt with later'''
    # Fix for CB referencing (for every residue but the first one)
    C_ref = ref_coords.at[:,1,0].get()
    prev_C_ref = jnp.roll(C_ref, 1, axis=0)
    ref_coords = ref_coords.at[:,1,0].set(prev_C_ref)
    # Fix for CB referencing of first residue, use (N of second residue, C of first residue, CA of first residue)
    ref_for_first_CB = jnp.concatenate([coords[1,0][None], coords[0,jnp.array([2,1,4])]])
    ref_coords = ref_coords.at[0,1].set(ref_for_first_CB)

    ## Pull out the lengths, angles and dihedrals
    data_sc = vmap(decompose_quad)(ref_coords) # lengths, angles, dihedrals

    insert_zero = lambda x: jnp.concatenate([jnp.zeros((1,)+x.shape[1:], dtype=x.dtype), x])
    data_bb = xyz_to_internal_coords(insert_zero(coords[:,:3]).reshape(-1,3)) # Dummy reference residue used
    lengths, angles, dihedrals = [jnp.concatenate([bb,sc], axis=-1) for bb,sc in zip(data_bb, data_sc)]
    
    angles = angles.at[0,0].set(1.) # Non zero value for first angle, which is really a 'ghost' one
    bond_mask = lengths
    angles_mask = jnp.stack([angles, dihedrals])

    # fix incidental angles extraction for phantom CB on glycines
    mask = (point_ref[:,:,1]==0).all(0)
    bond_mask = bond_mask.at[mask, 4].set(0.)
    angles_mask = angles_mask.at[:,mask,4].set(0.)

    return bond_mask, angles_mask

def make_scaffolds(coords, seq):
    '''
    coords - (L,14,3)
    seq - one letter aa sequence, no non-standard amino acids
    '''
    point_ref, cloud_mask = get_point_ref_and_cloud_mask(seq)
    bond_mask, angles_mask = get_data_masks(coords, point_ref)
    scaffolds_natural = {'point_ref_mask':point_ref[:3], 'angles_mask':angles_mask, 'bond_mask': bond_mask, 'cloud_mask':cloud_mask}
    return scaffolds_natural

def load_pdb(path, first_frame_only=True):
    coords, seq = load_to_sc_coord_format(path, first_frame_only=first_frame_only)
    scaffolds_natural = make_scaffolds(coords, seq) if first_frame_only else vmap(make_scaffolds, in_axes=(0, None))(coords, seq)
    return scaffolds_natural