from jax import vmap, numpy as jnp
from nerfax.reconstruct import normalise

scale_to = lambda direc, dist: normalise(direc)*dist
lerp = lambda lo, hi, a: lo + a*(hi-lo)
def place_type1(p1,p2,p3,p4, dist):
    direc = -normalise(jnp.stack([p2,p3,p4])-p1).sum(0)
    return p1 + scale_to(direc, dist)

def place_type2(p1,p2,p3, dist, angle, fudge):
    shift = lerp(*normalise(jnp.stack([p2,p3])-p1), 0.5+fudge)
    between = p1+shift
    return place_type3(p1, between, p2, dist, angle, 90.)

def matrix4d(v, theta):
    m = jnp.zeros((4,4))
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    t = 1-c
    x,y,z = v
    m3 = t*v[:,None]*v[None,:] + jnp.array([
        [c   , z*s , -y*s],
        [-z*s, c   ,  x*s],
        [y*s , -x*s,    c]
    ])
    return m.at[:3,:3].set(m3).at[3,3].set(1.)

def apply_matrix4d(m, v):
    # m is 4d matrix (4,4), v is 3d vector (3,)
    x = (v[0] * m[0][0]) + (v[1] * m[1][0]) + (v[2] * m[2][0]) + m[3][0]
    y = (v[0] * m[0][1]) + (v[1] * m[1][1]) + (v[2] * m[2][1]) + m[3][1]
    z = (v[0] * m[0][2]) + (v[1] * m[1][2]) + (v[2] * m[2][2]) + m[3][2]
    w = (v[0] * m[0][3]) + (v[1] * m[1][3]) + (v[2] * m[2][3]) + m[3][3]
    return jnp.array([x,y,z])/w

def makeVec(a,b):
    return normalise(a-b)

def rotate(theta, a, b, v):
    rotmat = matrix4d(makeVec(b,a), theta)
    return apply_matrix4d(rotmat,(v-b))+b
    
def place_type3(p1,p2,p3, dist, theta, phi):
    # assume input in angle
    phi = (jnp.pi/180)*phi
    theta = (jnp.pi/180)*theta     
                      
    direc = jnp.cross(*normalise(jnp.stack([p1,p3])-p2))
    norm = scale_to(direc, dist)
    
    pos4 = rotate(phi-jnp.pi/2, p2, p1, p1+norm)
    pos5 = jnp.cross(*normalise(jnp.stack([pos4, p2])-p1)) + p1
    
    return rotate(jnp.pi/2-theta, p1, pos5, pos4)

def place_type4(p1,p2,p3, dist, fudge): # fudge = ang2
    direc = -lerp(*normalise(jnp.stack([p2,p3])-p1), 0.5+fudge)
    return p1 + scale_to(direc, dist)

def angle(p1,p2,p3):
    ab = jnp.stack([p1,p3])-p2
    mags = jnp.linalg.norm(ab, axis=-1)
    return jnp.arccos(jnp.dot(*ab)/jnp.prod(mags))
    
def place_type5(p1,p2,p3, dist, fract):
    vto2 = scale_to(jnp.stack([p2,p3])-p1, dist)
    pos4 = jnp.cross(*vto2)+p1
    cnca_angle = angle(p2, p1, p3)
    hnca_angle = fract*(2*jnp.pi - cnca_angle) 
    return rotate(hnca_angle, pos4, p1, p1+vto2[0])

# currently just computes everything, then selects.
def place(placement_type, xyz, ancillary_data):
    vals = jnp.stack([
        place_type1(*xyz, ancillary_data[0]),
        place_type2(*xyz[:3], *ancillary_data),
        place_type3(*xyz[:3], *ancillary_data),
        place_type4(*xyz[:3], *ancillary_data[jnp.array([0,2])]),
        place_type5(*xyz[:3], *ancillary_data[jnp.array([0,2])])
    ])
    return vals[placement_type-1]

def reconstruct(bbcoords, data, nterminal_data, atom_mask, aas):
    placement_data = vmap(lambda x,y: x[y])(bbcoords, data['placement_data'])
    # Hot fix for using prev residue to place amide proton
    amide_mask = (aas!=14) & (atom_mask[:,0])
    # placement_data[mask, 0 = first H atom (H), 2 = third placement atom (prev C)]
    # bbcoords[:, 2 = heavy atom order index of C], roll shifts back by 1 
    placement_data = placement_data.at[amide_mask,0,2,:].set(jnp.roll(bbcoords[:, 2], 1, axis=0)[amide_mask]) 
    hcoords_list = vmap(place)(*[x[atom_mask]  for x in (data['placement_type'], placement_data, data['ancillary_data'])])
    # deal with N-terminal
    placement_data = bbcoords[0, nterminal_data['placement_data']]
    nterminal_hcoords_list = vmap(place)(nterminal_data['placement_type'], placement_data, nterminal_data['ancillary_data'])
    return jnp.concatenate([nterminal_hcoords_list, hcoords_list])