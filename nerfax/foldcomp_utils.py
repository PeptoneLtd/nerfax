import struct
import numpy as np

import jax
from jax import numpy as jnp, vmap

from nerfax.reconstruct import mp_nerf_jax_pretrig_self_normalising, normalise, mp_nerf_jax
from nerfax.foldcomp_constants import AA_REF_BOND_LENGTHS, AA_REF_ANGLES, AA_PLACEMENT_DEPENDENCIES, AA_REF_ATOM_MASK, BACKBONE_BOND_LENGTHS

def load_backbone_data(fcz, n):
    d = np.frombuffer(fcz.read(8*n), dtype=np.uint8).reshape(-1,8).astype(np.uint16)
    aas = (d[:,0] & 0xF8) >>3
    # overwrite first three parts of d
    d[:,0] = ((d[:,0] & 0x0007) << 8) | (d[:,1] & 0x00FF) # omega
    d[:,1] = ((d[:,2] & 0x00FF) << 4) | (d[:,3] & 0x00FF) >> 4 # psi
    d[:,2] = ((d[:,3] & 0x000F) << 8) | (d[:,4] & 0x00FF) # phi
    # ca_c_n_angle, c_n_ca_angle, n_ca_c_angle = d[:,-3:].T
    # 0-omega, 1-psi, 2-phi, 5-ca_c_n_angle, 6-c_n_ca_angle, 7-n_ca_c_angle
    # angles_torsions_discretizers is (phi,psi,omega, n_ca_c_angle, ca_c_n_angle, c_n_ca_angle), we reorder to match this
    angles_torsions = d[:,[2,1,0,7,5,6]] 
    return aas, angles_torsions

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
    # prevAnchorCoords, innerAnchorCoords, lastAtomCoords = np.split(anchorCoords, [1,-1])
    hasOXT = struct.unpack('?',fcz.read(1))[0]
    oxtCoords = np.frombuffer(fcz.read(12), dtype=np.float32)
    # place OXT in atoms
    split_index = anchorIndices[-2]
    aas_body, angles_torsions_body = load_backbone_data(fcz, split_index)
    aas_end, angles_torsions_end = load_backbone_data(fcz, nResidue-split_index)
    aas = np.concatenate([aas_body, aas_end])
    angles_torsions_body = angles_torsions_body.reshape(anchorIndices.shape[0]-2, -1, 6)

    # encodedSideChain = np.ceil(nSideChainTorsion/2).astype(int) # half rounded upwards
    # maybe need to add 1 to sc if odd number?
    sideChainAnglesDiscretized = np.frombuffer(fcz.read(nSideChainTorsion), dtype=np.uint8)
    tempFactorsDisc_min, tempFactorsDisc_cont_f = struct.unpack('@2f',fcz.read(2*4))
    tempFactorsDisc = np.frombuffer(fcz.read(nResidue), dtype=np.uint8)

    # print(fcz.read())
    return (tag, nResidue, nAtom, idxResidue, idxAtom, nAnchor, chain, firstResidue,lastResidue, strTitle), (anchorIndices, anchorCoords), (hasOXT, oxtCoords), aas, (angles_torsions_discretizers, angles_torsions_body, angles_torsions_end), sideChainAnglesDiscretized, (tempFactorsDisc_min, tempFactorsDisc_cont_f, tempFactorsDisc)

reverse = lambda x: jnp.flip(x, axis=1)
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

    # Weighted reconstruction
    m = coords.shape[1]
    w = (jnp.arange(m)/(m-1)).reshape(1,-1,1)
    coords_w_avg = coords*(1-w) + reverse(coords_bwd)*w
    return jnp.concatenate([anchors[:-1], coords_w_avg], axis=1).reshape(-1,3)

# def reconstruct_backbone(angles_torsions_discretizers, angles_torsions, anchorIndices, anchorCoords):
#     # continuize backbone torsions and angles
#     mins, conf_fs = angles_torsions_discretizers # swap so ordering matches
#     # (phi,psi,omega, n_ca_c_angle, ca_c_n_angle, c_n_ca_angle) ordering in axis=1 [phi,psi,omega, n_ca_c_angle, ca_c_n_angle, c_n_ca_angle = angles_torsions_cont.T]
#     angles_torsions_cont = (angles_torsions*conf_fs+mins) * (jnp.pi/180) # convert to radians

#     # reorder so we place (anchors)-N->CA->C->... so N first
#     # hence we need (CA-C-N, psi, length C-N) then (C-N-CA, omega, length N-CA) then (N-CA-C,phi,CA-C) on repeat
#     torsions = angles_torsions_cont[...,[1,2,0]]
#     angles = angles_torsions_cont[...,[4,5,3]]
#     angles = jnp.pi - angles # due to historical scnet reasons, the scnet angle is defined as pi-angle. It seems they've used scnet defn here

#     # there's a body of reconstruction all equally spaced (normally 25 spaced)
#     m = int(anchorIndices[1]-anchorIndices[0])
#     n = anchorIndices.shape[0]-2
#     lengths_body = jnp.tile(BACKBONE_BOND_LENGTHS, m*n).reshape((n, m*3))
#     angles_body, torsions_body = (x[:n*m].reshape(n, m*3) for x in (angles, torsions))
#     coords_body = reconstruct_both_ways(lengths_body, angles_body, torsions_body, anchorCoords[:-1])

#     # final set of reconstruction can be varying length
#     p = int(anchorIndices[-1]-anchorIndices[-2])
#     lengths_end = jnp.tile(BACKBONE_BOND_LENGTHS, p).reshape((1, p*3))
#     angles_end, torsions_end = (x[n*m:n*m+p].reshape(1, -1) for x in (angles, torsions))
#     coords_end = reconstruct_both_ways(lengths_end, angles_end, torsions_end, anchorCoords[-2:])

#     bb_pos = jnp.concatenate([coords_body, coords_end, anchorCoords[-1]])
#     return bb_pos

def reconstruct_backbone(angles_torsions_discretizers, anchorCoords, angles_torsions_body, angles_torsions_end):
    angles_torsions_end = angles_torsions_end[None, :-1]

    # continuize backbone torsions and angles
    mins, conf_fs = angles_torsions_discretizers # swap so ordering matches
    
    def process(angles_torsions):
        # (phi,psi,omega, n_ca_c_angle, ca_c_n_angle, c_n_ca_angle) ordering in axis=1 [phi,psi,omega, n_ca_c_angle, ca_c_n_angle, c_n_ca_angle = angles_torsions_cont.T]
        angles_torsions_cont = (angles_torsions*conf_fs+mins) * (jnp.pi/180) # convert to radians

        # reorder so we place (anchors)-N->CA->C->... so N first
        # hence we need (CA-C-N, psi, length C-N) then (C-N-CA, omega, length N-CA) then (N-CA-C,phi,CA-C) on repeat
        torsions = angles_torsions_cont[...,[1,2,0]]
        angles = angles_torsions_cont[...,[4,5,3]]
        angles = jnp.pi - angles # due to historical scnet reasons, the scnet angle is defined as pi-angle. It seems they've used scnet defn here
        
        angles, torsions = [x.reshape((x.shape[0], -1)) for x in (angles, torsions)]
        return angles, torsions

    # there's a body of reconstruction all equally spaced (normally 25 spaced)    
    angles_body, torsions_body = process(angles_torsions_body)
    lengths_body = jnp.broadcast_to(jnp.tile(BACKBONE_BOND_LENGTHS, angles_body.shape[-1]//3), angles_body.shape)
    coords_body = reconstruct_both_ways(lengths_body, angles_body, torsions_body, anchorCoords[:-1])

    # final set of reconstruction can be varying length
    angles_end, torsions_end = process(angles_torsions_end)
    lengths_end = jnp.broadcast_to(jnp.tile(BACKBONE_BOND_LENGTHS, angles_end.shape[-1]//3), angles_end.shape)
    coords_end = reconstruct_both_ways(lengths_end, angles_end, torsions_end, anchorCoords[-2:])

    bb_pos = jnp.concatenate([coords_body, coords_end, anchorCoords[-1]])
    return bb_pos
    
def place_sidechains(cloud_mask, point_ref_mask, angles_mask, bond_mask, bb_coords):
    n = bb_coords.shape[0]
    coords = jnp.concatenate([bb_coords, jnp.zeros((n,11,3))], axis=-2)
    for i in range(11):
        level_mask = cloud_mask[:, i]
        thetas, dihedrals = angles_mask[:, level_mask, i]
        idx_a, idx_b, idx_c = point_ref_mask[:, level_mask, i]
        placed_coords = vmap(mp_nerf_jax)(
            coords[level_mask, idx_a],
            coords[level_mask, idx_b],
            coords[level_mask, idx_c],
            bond_mask[level_mask, i], 
            thetas, dihedrals
        )
        coords = coords.at[level_mask, i+3].set(placed_coords)
    return coords
    
def continuize_sidechain_angles(sideChainAnglesDiscretized):
    MIN_ANGLE, MAX_ANGLE = -jnp.pi, jnp.pi
    scaling = (MAX_ANGLE-MIN_ANGLE)/255
    continuizeAngle = lambda x: x*scaling+MIN_ANGLE
    sideChainAngles = continuizeAngle(sideChainAnglesDiscretized)
    return sideChainAngles

def reconstruct_sidechains(aas, sideChainAnglesDiscretized, bb_pos):
    lengths, angles, placement_dependencies, atom_mask = jax.tree_map(lambda x: jnp.array(x).at[aas].get(),
        (AA_REF_BOND_LENGTHS, 
        AA_REF_ANGLES, 
        AA_PLACEMENT_DEPENDENCIES, 
        AA_REF_ATOM_MASK))

    sideChainAngles = continuize_sidechain_angles(sideChainAnglesDiscretized)
    sidechain_indexs = jnp.where(atom_mask, size=sideChainAngles.shape[0]) # fixed size for jit
    torsions = jnp.zeros(atom_mask.shape).at[sidechain_indexs].set(sideChainAngles)

    reconstructed_sparse_coords = place_sidechains(
        cloud_mask=atom_mask, 
        point_ref_mask=jnp.transpose(placement_dependencies, (2,0,1)), 
        angles_mask=jnp.stack([angles, torsions]), 
        bond_mask=lengths, 
        bb_coords=bb_pos.reshape(-1,3,3)
    )
    reconstructed_coords = reconstructed_sparse_coords[
        jnp.concatenate([
            jnp.ones((atom_mask.shape[0], 3), dtype=bool), 
            atom_mask
        ], axis=1)
    ]
    return reconstructed_coords

def reconstruct(angles_torsions_discretizers, angles_torsions_body, angles_torsions_end, anchorCoords, aas, sideChainAnglesDiscretized, hasOXT, oxtCoords):
    bb_coords = reconstruct_backbone(angles_torsions_discretizers, anchorCoords, angles_torsions_body, angles_torsions_end)
    coords = reconstruct_sidechains(aas, sideChainAnglesDiscretized, bb_coords)
    if (hasOXT):
        coords = jnp.concatenate([coords, jnp.array(oxtCoords)])
    return coords    

def decompress(path):
    (tag, nResidue, nAtom, idxResidue, idxAtom, nAnchor, chain, firstResidue, lastResidue, strTitle), \
    (anchorIndices, anchorCoords), (hasOXT, oxtCoords), aas, (angles_torsions_discretizers, angles_torsions_body, angles_torsions_end), \
    sideChainAnglesDiscretized, (tempFactorsDisc_min, tempFactorsDisc_cont_f, tempFactorsDisc) = load_data(path)
    coords = reconstruct(angles_torsions_discretizers, angles_torsions_body, angles_torsions_end, anchorCoords, aas, sideChainAnglesDiscretized, hasOXT, oxtCoords)
    # continuize temp factors
    # tempFactors = tempFactorsDisc*tempFactorsDisc_cont_f+tempFactorsDisc_min
    return coords
