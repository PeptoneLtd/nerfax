import numpy as np
import foldcomp
from nerfax.foldcomp_utils import decompress
compute_dists = lambda a,b: (((a-b)**2).sum(-1)**0.5)
compute_rmsd = lambda a,b: ((a-b)**2).sum(-1).mean()**0.5

def check_rmsd_to_foldcomp(path, threshold=0.1):
    foldcomp_coords = np.array(foldcomp.get_data(open(path,'rb').read())['coordinates'])
    coords = decompress(path)
    assert compute_rmsd(foldcomp_coords, coords)<threshold # threshold is in Angstrom
