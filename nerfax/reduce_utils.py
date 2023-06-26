import mdtraj as md
import numpy as np
import pandas as pd
from nerfax.foldcomp_constants import ATOM_ORDER, ConvertIntToOneLetterCode
from jax import vmap, numpy as jnp
from collections import defaultdict
from nerfax.reduce_constants import HATOM_ORDER_PADDED, HATOM_DATA, HATOM_MASK, HATOM_NTERMINAL_DATA
from Bio.Data.IUPACData import protein_letters_3to1, protein_letters_1to3
from nerfax.reduce_reconstruct import reconstruct
ConvertOneLetterCodeToInt = np.vectorize({v:k for k,v in enumerate(ConvertIntToOneLetterCode[:20])}.__getitem__)

def get_scnet_loader_fns(t):
    '''
    Requires ALL heavy sidechain atoms to be present
    Convert positions from pdb to (L,14,3) format
    '''
    sc_net_coord_dict = {f'{aa}-{atomname}':i for aa, atomnames in ATOM_ORDER.items() for i, atomname in enumerate(atomnames)}

    d = t.top.to_dataframe()[0]
    d['resSeq'] -= (d.resSeq.min()-1)
    atom_idxs = np.vectorize(defaultdict(lambda: None, sc_net_coord_dict).__getitem__, otypes=[object])(d.resName+'-'+d.name)

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

def reorder_traj(t):
    d = t.top.to_dataframe()[0].reset_index(drop=True)
    d['index'] = d.index
    order = d.sort_values(['resSeq','index'])['index']
    d = d.iloc[order].reset_index(drop=True).drop(columns=['index'])
    d['serial'] = d.index+1
    return md.Trajectory(
        t.xyz[:, order],
        md.Topology.from_dataframe(d)
    )

def build_associated_mdtraj_topology(t_bb, atom_mask, aas, nterminal_data):
    # Build topology for most atoms
    aanum_to_resname = np.vectorize(lambda aa: protein_letters_1to3[ConvertIntToOneLetterCode[aa]].upper())
    seq_3 = aanum_to_resname(aas)
    resseqs = np.repeat((np.arange(aas.shape[0])+1), 13)[atom_mask.ravel()]
    resnames = np.repeat(seq_3, 13)[atom_mask.ravel()]
    atomnames = HATOM_ORDER_PADDED[aas][atom_mask]
    # add in special Nterminal atoms topology
    n = nterminal_data['atom_order'].shape[0]
    resseqs, resnames, atomnames = (np.concatenate([x,x2]) for x, x2 in [
        (np.ones(n, dtype=int), resseqs),
        (np.tile(seq_3[0], n), resnames),
        (nterminal_data['atom_order'], atomnames)])

    d = pd.DataFrame({
        'name': atomnames,
        'element': 'H',
        'resSeq': resseqs,
        'resName': resnames,
        'chainID': 0,
        'segmentID': ''
    })
    d = pd.concat([t_bb.top.to_dataframe()[0], d]).reset_index(drop=True)
    d['serial'] = d.index+1
    return d

def reconstruct_from_mdtraj(t):
    _parse_coords, _restrict_to_scnet_atoms, _scnet_to_list = get_scnet_loader_fns(t)
    t_bb = _restrict_to_scnet_atoms(t)
    bbcoords = vmap(_parse_coords)(t.xyz)

    seq = np.array(list(t.top.to_fasta()[0]))
    aas = ConvertOneLetterCodeToInt(seq)
    data = {k: v[aas] for k,v in HATOM_DATA.items()}
    atom_mask = HATOM_MASK[aas]
    atom_mask[0,0] = False # remove the H from first residue (if not already)

    k = 'nt-amide' if (aas[0]!=ConvertOneLetterCodeToInt("P")) else 'nt-pro'
    nterminal_data = HATOM_NTERMINAL_DATA[k]

    def _reconstruct(bbcoords):
        return reconstruct(bbcoords, data, nterminal_data, atom_mask, aas)

    hpos = vmap(_reconstruct)(bbcoords)

    t_reconstructed = reorder_traj(md.Trajectory(
        xyz=np.concatenate([t_bb.xyz, hpos/10.], axis=1), 
        topology=md.Topology.from_dataframe(build_associated_mdtraj_topology(t_bb, atom_mask, aas, nterminal_data))))
    return t_reconstructed