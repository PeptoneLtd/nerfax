import struct
from Bio.Data.IUPACData import protein_letters_3to1
import numpy as np

import foldcomp
import mdtraj as md
import plotly.express as px
import plotly.graph_objects as go
import nerfax
from nerfax.reconstruct import mp_nerf_jax_pretrig_self_normalising, normalise
import jax
from jax import numpy as jnp, vmap, jit

get_dist = lambda x: (((x)**2).sum(-1)**0.5)
reverse = lambda x: np.flip(x, axis=1)

# dict(enumerate(list(map(protein_letters_3to1.__getitem__, sorted(protein_letters_3to1)))+['B','Z','*','X']))
ConvertIntToOneLetterCode = {0: 'A',
 1: 'R',
 2: 'N',
 3: 'D',
 4: 'C',
 5: 'Q',
 6: 'E',
 7: 'G',
 8: 'H',
 9: 'I',
 10: 'L',
 11: 'K',
 12: 'M',
 13: 'F',
 14: 'P',
 15: 'S',
 16: 'T',
 17: 'W',
 18: 'Y',
 19: 'V',
 20: 'B',
 21: 'Z',
 22: '*',
 23: 'X'}
ConvertIntToOneLetterCode = np.array(list(ConvertIntToOneLetterCode.values()))

expand = lambda x: (jnp.sin(x), jnp.cos(x)) # expand x in sin,cos basis
def reconstruct_from_internal_coordinates_pure_sequential(lengths, angles, dihedrals, init):
    '''
    We follow the simplifications in Parsons et al, Practical conversion from torsion space to Cartesian space for in silico protein synthesis, 10.1002/jcc.20237
    detailed as SN-NeRF
    Implementation here computes the sin and cos of angles and dihedrals externally to the scan so vectorised trig instructions can be used
    
    '''
    def _body(ba, cb, c, length, sin_angle, cos_angle, sin_dihedral, cos_dihedral):
        d, dc = mp_nerf_jax_pretrig_self_normalising(ba, cb, c, length, sin_angle, cos_angle, sin_dihedral, cos_dihedral)
        return (cb, dc, d), (d,)

    _, (coords,) = jax.lax.scan(
        lambda carry, x: _body(*carry, *x),
        init=(normalise(init[1]-init[0]), normalise(init[2]-init[1]), init[2]),
        xs=(lengths, *expand(angles), *expand(dihedrals)),
        unroll=4
    )
    return coords

def reconstruct_both_ways(lengths, angles, torsions, anchors):
    f = vmap(reconstruct_from_internal_coordinates_pure_sequential)

    body_args = jax.tree_map(lambda x: x[:,:-3], (lengths, angles, torsions))
    coords = f(*body_args, anchors[:-1])

    # Reverse placements
    l_bwd = reverse(lengths[...,1:-2])
    angles_bwd = reverse(angles[...,2:-1])
    torsions_bwd = reverse(torsions[...,3:])
    anchors_bwd = reverse(anchors[1:]) # flip both in axis 1 and axis 0
    coords_bwd = f(l_bwd, angles_bwd, torsions_bwd, anchors_bwd)

    # # As if fwd and bwd
    # coords_reconstructed_fwd = np.concatenate([
    #     np.concatenate([anchors, coords], axis=1).reshape(-1,3), 
    #     anchorCoords[n]
    # ])
    # coords_reconstructed_bwd = np.concatenate([
    #     anchors[0], 
    #     reverse(np.concatenate([anchors_bwd, coords_bwd], axis=1)).reshape(-1,3),
    # ])

    # Weighted reconstruction
    m = coords.shape[1]
    w = (np.arange(m)/(m-1)).reshape(1,-1,1)
    coords_w_avg = coords*(1-w) + reverse(coords_bwd)*w
    # coords_w_avg  = np.concatenate([
    #     np.concatenate([anchors, coords_w_avg], axis=1).reshape(-1,3),
    #     anchors[-1]
    # ])
    return jnp.concatenate([anchors[:-1], coords_w_avg], axis=1).reshape(-1,3)

def reconstruct_backbone(angles_torsions_cont, anchorIndices, anchorCoords):
    # reorder so we place (anchors)-N->CA->C->... so N first
    # hence we need (CA-C-N, psi, length C-N) then (C-N-CA, omega, length N-CA) then (N-CA-C,phi,CA-C) on repeat
    torsions = angles_torsions_cont[...,[1,2,0]]
    angles = angles_torsions_cont[...,[4,5,3]]
    angles = np.pi - angles # due to historical scnet reasons, the scnet angle is defined as pi-angle. It seems they've used scnet defn here
    # {"N_TO_CA", 1.46}, {"CA_TO_C", 1.52}, {"C_TO_N", 1.33}
    bond_lengths = np.array([1.33,1.46,1.52])

    m = int(anchorIndices[1]-anchorIndices[0])
    n = nAnchor-2
    lengths_body = np.tile(bond_lengths, m*n).reshape((n, m*3))
    angles_body, torsions_body = (x[:n*m].reshape(n, m*3) for x in (angles, torsions))
    coords_body = reconstruct_both_ways(lengths_body, angles_body, torsions_body, anchorCoords[:-1])

    p = int(anchorIndices[-1]-anchorIndices[-2])
    lengths_end = jnp.tile(bond_lengths, p).reshape((1, p*3))
    angles_end, torsions_end = (x[n*m:n*m+p].reshape(1, -1) for x in (angles, torsions))
    coords_end = reconstruct_both_ways(lengths_end, angles_end, torsions_end, anchorCoords[-2:])

    bb_pos = jnp.concatenate([coords_body, coords_end, anchorCoords[-1]])
    return bb_pos

def load_data(path):
    fcz = open(path, "rb")
    # tag, nResidue, nAtom, idxResidue, idxAtom, nAnchor, chain, _, \
    # nSideChainTorsion,firstResidue,lastResidue,_,lenTitle, \
    # phiDisc_min,psiDisc_min,omegaDisc_min,n_ca_c_angleDisc_min,ca_c_n_angleDisc_min,c_n_ca_angleDisc_min, \
    # phiDisc_cont_f,psiDisc_cont_f,omegaDisc_cont_f,n_ca_c_angleDisc_cont_f,ca_c_n_angleDisc_cont_f,c_n_ca_angleDisc_cont_f = struct.unpack('@4s4HBccIcccI12f' ,fcz.read(76))

    tag, nResidue, nAtom, idxResidue, idxAtom, nAnchor, chain, _, \
    nSideChainTorsion,firstResidue,lastResidue,_,lenTitle = struct.unpack('@4s4HBccIcccI' ,fcz.read(28))
    angles_torsions_discretizers = np.frombuffer(fcz.read(48), dtype=np.float32).reshape(2,6)

    anchorIndices = np.frombuffer(fcz.read(nAnchor*4), dtype=np.uint32)
    strTitle = fcz.read(lenTitle)

    anchorCoords = np.frombuffer(fcz.read(36*nAnchor), dtype=np.float32).reshape(-1,3,3)
    prevAnchorCoords, innerAnchorCoords, lastAtomCoords = np.split(anchorCoords, [1,-1])
    hasOXT = struct.unpack('?',fcz.read(1))[0]
    oxtCoords = np.frombuffer(fcz.read(12), dtype=np.float32)
    # place OXT in atoms

    d = np.frombuffer(fcz.read(8*nResidue), dtype=np.uint8).reshape(-1,8).astype(np.uint16)
    aas = (d[:,0] & 0xF8) >>3
    # overwrite first three parts of d
    d[:,0] = ((d[:,0] & 0x0007) << 8) | (d[:,1] & 0x00FF) # omega
    d[:,1] = ((d[:,2] & 0x00FF) << 4) | (d[:,3] & 0x00FF) >> 4 # psi
    d[:,2] = ((d[:,3] & 0x000F) << 8) | (d[:,4] & 0x00FF) # phi
    # ca_c_n_angle, c_n_ca_angle, n_ca_c_angle = d[:,-3:].T
    # 0-omega, 1-psi, 2-phi, 5-ca_c_n_angle, 6-c_n_ca_angle, 7-n_ca_c_angle
    # angles_torsions_discretizers is (phi,psi,omega, n_ca_c_angle, ca_c_n_angle, c_n_ca_angle), we reorder to match this
    angles_torsions = d[:,[2,1,0,7,5,6]] 

    # encodedSideChain = np.ceil(nSideChainTorsion/2).astype(int) # half rounded upwards
    # maybe need to add 1 to sc if odd number?
    sideChainAnglesDiscretized = np.frombuffer(fcz.read(nSideChainTorsion), dtype=np.uint8)
    tempFactorsDisc_min, tempFactorsDisc_cont_f = struct.unpack('@2f',fcz.read(2*4))
    tempFactorsDisc = np.frombuffer(fcz.read(nResidue), dtype=np.uint8)

    print(fcz.read())
    return (tag, nResidue, nAtom, idxResidue, idxAtom, nAnchor, chain, firstResidue,lastResidue, strTitle), (anchorIndices, anchorCoords), (hasOXT, oxtCoords), aas, (angles_torsions_discretizers, angles_torsions), sideChainAnglesDiscretized, (tempFactorsDisc_min, tempFactorsDisc_cont_f, tempFactorsDisc)

def decompress(path):
    (tag, nResidue, nAtom, idxResidue, idxAtom, nAnchor, chain, firstResidue,lastResidue, strTitle), (anchorIndices, anchorCoords), (hasOXT, oxtCoords), aas, (angles_torsions_discretizers, angles_torsions), sideChainAnglesDiscretized, (tempFactorsDisc_min, tempFactorsDisc_cont_f, tempFactorsDisc) = load_data(path)

    # continuize temp factors
    tempFactors = tempFactorsDisc*tempFactorsDisc_cont_f+tempFactorsDisc_min
    # continuize backbone torsions and angles
    mins, conf_fs = angles_torsions_discretizers # swap so ordering matches
    # (phi,psi,omega, n_ca_c_angle, ca_c_n_angle, c_n_ca_angle) ordering in axis=1 [phi,psi,omega, n_ca_c_angle, ca_c_n_angle, c_n_ca_angle = angles_torsions_cont.T]
    angles_torsions_cont = (angles_torsions*conf_fs+mins) * (np.pi/180) # convert to radians

    # reorder so we place (anchors)-N->CA->C->... so N first
    # hence we need (CA-C-N, psi, length C-N) then (C-N-CA, omega, length N-CA) then (N-CA-C,phi,CA-C) on repeat
    torsions = angles_torsions_cont[...,[1,2,0]]
    angles = angles_torsions_cont[...,[4,5,3]]
    angles = np.pi - angles # due to historical scnet reasons, the scnet angle is defined as pi-angle. It seems they've used scnet defn here
    # {"N_TO_CA", 1.46}, {"CA_TO_C", 1.52}, {"C_TO_N", 1.33}
    bond_lengths = np.array([1.33,1.46,1.52])

    m = anchorIndices[1]-anchorIndices[0]
    n = nAnchor-2
    lengths_body = np.tile(bond_lengths, m*n).reshape((n, m*3))
    angles_body, torsions_body = (x[:n*m].reshape(n, m*3) for x in (angles, torsions))
    coords_body = reconstruct_both_ways(lengths_body, angles_body, torsions_body, anchorCoords[:-1])
    
path = '/raid/app/oliver/repos/foldcomp/test/compressed.fcz'
(tag, nResidue, nAtom, idxResidue, idxAtom, nAnchor, chain, firstResidue,lastResidue, strTitle), (anchorIndices, anchorCoords), (hasOXT, oxtCoords), aas, (angles_torsions_discretizers, angles_torsions), sideChainAnglesDiscretized, (tempFactorsDisc_min, tempFactorsDisc_cont_f, tempFactorsDisc) = load_data(path)


def checks():
    import foldcomp
    import mdtraj as md
    t = md.load('/raid/app/oliver/repos/foldcomp/test/test.pdb')

    x = foldcomp.get_data(open("/raid/app/oliver/repos/foldcomp/test/compressed.fcz",'rb').read())
    x.update(dict(
        zip(
            ['ca_c_n_angle', 'c_n_ca_angle','n_ca_c_angle'],
            list(np.array(x['bond_angles']).reshape(-1,3).T)
        )
    ))

    for k in 'phi,psi,omega,n_ca_c_angle,ca_c_n_angle,c_n_ca_angle'.split(','):
        print(k)
        display(px.scatter(x=x[k], y=globals()[k]))
    display(px.scatter(x=d['b_factors'], y=tempFactors))
    assert t.top.to_fasta()[0]==''.join(ConvertIntToOneLetterCode[aas])
    return t,x

