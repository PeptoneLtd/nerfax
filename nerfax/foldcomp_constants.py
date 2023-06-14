import numpy as np
def load_reference_sidechain_data():
    import re
    import tree
    import ast
    import pandas as pd

    s = '''        output.emplace("ALA", AminoAcid('A', "ALA", "Alanine", // name
                {"N", "CA", "C", "O", "CB",}, // atoms
                {{"O", {"N", "CA", "C"}}, {"CB", {"O", "C", "CA"}}}, // sidechain
                {"N", "CA", "C", "CB", "O"}));
            output["ALA"].bondLengths = {{"CA_CB", 1.52}, {"C_O", 1.23}};
            output["ALA"].bondAngles = {{"CA_C_O", 120.31}, {"C_CA_CB", 110.852}};
            // Arginine (R/ARG)
            output.emplace("ARG", AminoAcid(
                'R', "ARG", "Arginine", // name
                { "N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2" }, // atoms
                {{"O", {"N", "CA", "C"}},{"CB", {"O", "C", "CA"}},
                {"CG", {"N", "CA", "CB"}},{"CD", {"CA", "CB", "CG"}},
                {"NE", {"CB", "CG", "CD"}},{"CZ", {"CG", "CD", "NE"}},
                {"NH1", {"CD", "NE", "CZ"}},{"NH2", {"CD", "NE", "CZ"}}}, // sidechain
                { "N", "CA", "C", "CB", "O", "CG", "CD", "NE", "NH1", "NH2", "CZ"}
            ));
            output["ARG"].bondLengths = {
                {"CA_CB", 1.53}, {"C_O", 1.23}, {"CB_CG", 1.53},
                {"CG_CD", 1.52}, {"CD_NE", 1.46}, {"NE_CZ", 1.32},
                {"CZ_NH1", 1.31}, {"CZ_NH2", 1.31}
            };
            output["ARG"].bondAngles = {
                {"CA_C_O", 119.745}, {"C_CA_CB", 110.579}, {"CA_CB_CG", 113.233},
                {"CB_CG_CD", 110.787}, {"CG_CD_NE", 111.919}, {"CD_NE_CZ", 125.192},
                {"NE_CZ_NH1", 120.077}, {"NE_CZ_NH2", 120.077}
            };
            // Asparagine (N/ASN)
            output.emplace("ASN", AminoAcid(
                'N', "ASN", "Asparagine", // name
                { "N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"}, // atoms
                {{"O", {"N", "CA", "C"}},{"CB", {"O", "C", "CA"}},
                {"CG", {"N", "CA", "CB"}},{"OD1", {"CA", "CB", "CG"}},
                {"ND2", {"CA", "CB", "CG"}}}, // sidechain
                { "N", "CA", "C", "CB", "O", "CG", "ND2", "OD1"}
            ));
            output["ASN"].bondLengths = {
                {"CA_CB", 1.52}, {"C_O", 1.23}, {"CB_CG", 1.52}, {"CG_OD1", 1.23}, {"CG_ND2", 1.325}
            };
            output["ASN"].bondAngles = {
                {"CA_C_O", 120.313}, {"C_CA_CB", 110.852}, {"CA_CB_CG", 113.232},
                {"CB_CG_OD1", 120.85}, {"CB_CG_ND2", 116.48}
            };
            // Aspartic Acid (D/ASP)
            output.emplace("ASP", AminoAcid(
                'D', "ASP", "Aspartic acid", // name
                { "N", "CA", "C", "O", "CB", "CG", "OD1", "OD2" }, // atoms
                {{"O", {"N", "CA", "C"}},{"CB", {"O", "C", "CA"}},
                {"CG", {"N", "CA", "CB"}},{"OD1", {"CA", "CB", "CG"}},
                {"OD2", {"CA", "CB", "CG"}}}, // sidechain
                { "N", "CA", "C", "CB", "O", "CG", "OD1", "OD2"}
            ));
            output["ASP"].bondLengths = {
                {"CA_CB", 1.53}, {"C_O", 1.23}, {"CB_CG", 1.52}, {"CG_OD1", 1.248}, {"CG_OD2", 1.248}
            };
            output["ASP"].bondAngles = {
                {"CA_C_O", 121.051}, {"C_CA_CB", 110.871}, {"CA_CB_CG", 113.232},
                {"CB_CG_OD1", 118.344}, {"CB_CG_OD2", 118.344}
            };
            // Cysteine (C/CYS)
            output.emplace("CYS", AminoAcid(
                'C', "CYS", "Cysteine", // name
                {"N", "CA", "C", "O", "CB", "SG"}, // atoms
                {{"O", {"N", "CA", "C"}},{"CB", {"O", "C", "CA"}},
                {"SG", {"N", "CA", "CB"}}}, // sidechain
                { "N", "CA", "C", "CB", "O", "SG"}
            ));
            output["CYS"].bondLengths = {
                {"CA_CB", 1.53}, {"C_O", 1.23}, {"CB_SG", 1.8}
            };
            output["CYS"].bondAngles = {
                {"CA_C_O", 120.063}, {"C_CA_CB", 111.078}, {"CA_CB_SG", 113.817}
            };
            // Glutamine (Q/GLN)
            output.emplace("GLN", AminoAcid(
                'Q', "GLN", "Glutamine", // name
                {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"}, // atoms
                {{"O", {"N", "CA", "C"}},{"CB", {"O", "C", "CA"}},
                {"CG", {"N", "CA", "CB"}},{"CD", {"CA", "CB", "CG"}},
                {"OE1", {"CB", "CG", "CD"}},{"NE2", {"CB", "CG", "CD"}}}, // sidechain
                { "N", "CA", "C", "CB", "O", "CG", "CD", "NE2", "OE1"}
            ));
            output["GLN"].bondLengths = {
                {"CA_CB", 1.53}, {"C_O", 1.23}, {"CB_CG", 1.52}, {"CG_CD", 1.52},
                {"CD_OE1", 1.23}, {"CD_NE2", 1.32}
            };
            output["GLN"].bondAngles = {
                {"CA_C_O", 120.211}, {"C_CA_CB", 109.5}, {"CA_CB_CG", 113.292},
                {"CB_CG_CD", 112.811}, {"CG_CD_OE1", 121.844}, {"CG_CD_NE2", 116.50}
            };
            // Glutamic Acid (E/GLU)
            output.emplace("GLU", AminoAcid(
                'E', "GLU", "Glutamic acid",
                {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"},
                {{"O", {"N", "CA", "C"}},{"CB", {"O", "C", "CA"}},
                {"CG", {"N", "CA", "CB"}},{"CD", {"CA", "CB", "CG"}},
                {"OE1", {"CB", "CG", "CD"}},{"OE2", {"CB", "CG", "CD"}}},
                { "N", "CA", "C", "CB", "O", "CG", "CD", "OE1", "OE2"}
            ));
            output["GLU"].bondLengths = {
                {"CA_CB", 1.53}, {"C_O", 1.23}, {"CB_CG", 1.52}, {"CG_CD", 1.52},
                {"CD_OE1", 1.25}, {"CD_OE2", 1.25}
            };
            output["GLU"].bondAngles = {
                {"CA_C_O", 120.594}, {"C_CA_CB", 110.538}, {"CA_CB_CG", 113.82},
                {"CB_CG_CD", 112.912}, {"CG_CD_OE1", 118.479}, {"CG_CD_OE2", 118.479}
            };
            // Glycine (G/GLY)
            output.emplace("GLY", AminoAcid(
                'G', "GLY", "Glycine", // name
                {"N", "CA", "C", "O"}, // atoms
                {{"O", {"N", "CA", "C"}}}, // sidechain
                { "N", "CA", "C", "O"}
            ));
            output["GLY"].bondLengths = {{"C_O", 1.23}};
            output["GLY"].bondAngles = {{"CA_C_O", 120.522}};
            // Histidine (H/HIS)
            output.emplace("HIS", AminoAcid(
                'H', "HIS", "Histidine",
                { "N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2" }, // atoms
                {{"O", {"N", "CA", "C"}}, {"CB", {"O", "C", "CA"}},
                {"CG", {"N", "CA", "CB"}}, {"ND1", {"CA", "CB", "CG"}},
                {"CD2", {"CA", "CB", "CG"}}, {"CE1", {"CB", "CG", "ND1"}},
                {"NE2", {"CB", "CG", "CD2"}}},
                { "N", "CA", "C", "CB", "O", "CG", "CD2", "ND1", "CE1", "NE2" }
            ));
            output["HIS"].bondLengths = {
                {"CA_CB", 1.53}, {"C_O", 1.23}, {"CB_CG", 1.5}, {"CG_ND1", 1.38},
                {"CG_CD2", 1.36}, {"ND1_CE1", 1.33}, {"CD2_NE2", 1.38}
            };
            output["HIS"].bondAngles = {
                {"CA_C_O", 120.548}, {"C_CA_CB", 111.329}, {"CA_CB_CG", 113.468},
                {"CB_CG_CD2", 130.61}, {"CB_CG_ND1", 122.85}, {"CG_CD2_NE2", 107.439},
                {"CG_ND1_CE1", 108.589}
            };
            // Isoleucine (I/ILE)
            output.emplace("ILE", AminoAcid(
                'I', "ILE", "Isoleucine", // name
                {"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"}, // atoms
                {{"O", {"N", "CA", "C"}}, {"CB", {"O", "C", "CA"}},
                {"CG1", {"N", "CA", "CB"}}, {"CG2", {"N", "CA", "CB"}},
                {"CD1", {"CA", "CB", "CG1"}}},
                { "N", "CA", "C", "CB", "O", "CG1", "CG2", "CD1" }
            ));
            output["ILE"].bondLengths = {
                {"CA_CB", 1.54}, {"C_O", 1.235}, {"CB_CG1", 1.53}, {"CB_CG2", 1.52},
                {"CG1_CD1", 1.51}
            };
            output["ILE"].bondAngles = {
                {"CA_C_O", 120.393}, {"C_CA_CB", 111.983}, {"CA_CB_CG1", 110.5},
                {"CA_CB_CG2", 110.5}, {"CB_CG1_CD1", 113.97}
            };
            // Leucine (L/LEU)
            output.emplace("LEU", AminoAcid(
                'L', "LEU", "Leucine",
                { "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2" }, // atoms
                {{"O", {"N", "CA", "C"}}, {"CB", {"O", "C", "CA"}},
                {"CG", {"N", "CA", "CB"}}, {"CD1", {"CA", "CB", "CG"}},
                {"CD2", {"CA", "CB", "CG"}} },
                { "N", "CA", "C", "CB", "O", "CG", "CD1", "CD2" }
            ));
            output["LEU"].bondLengths = {
                {"CA_CB", 1.53}, {"C_O", 1.235}, {"CB_CG", 1.53}, {"CG_CD1", 1.52},
                {"CG_CD2", 1.52}
            };
            output["LEU"].bondAngles = {
                {"CA_C_O", 120.211}, {"C_CA_CB", 110.418}, {"CA_CB_CG", 116.10},
                {"CB_CG_CD1", 110.58}, {"CB_CG_CD2", 110.58}
            };
            // Lysine (K/LYS)
            // 2022-06-10 21:56:59 - TODO: RECALCULATE GEOMETRY FOR LYSINE
            output.emplace("LYS", AminoAcid(
                'K', "LYS", "Lysine",
                { "N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ" }, // atoms
                {{"O", {"N", "CA", "C"}}, {"CB", {"O", "C", "CA"}},
                {"CG", {"N", "CA", "CB"}}, {"CD", {"CA", "CB", "CG"}},
                {"CE", {"CB", "CG", "CD"}}, {"NZ", {"CG", "CD", "CE"}}},
                { "N", "CA", "C", "CB", "O", "CG", "CD", "CE", "NZ" }
            ));
            output["LYS"].bondLengths = {
                {"C_O", 1.23}, {"CA_CB", 1.53}, {"CB_CG", 1.52}, {"CG_CD", 1.52},
                {"CD_CE", 1.52}, {"CE_NZ", 1.49} // sidechain
            };
            output["LYS"].bondAngles = {
                {"CA_C_O", 120.54}, {"C_CA_CB", 109.5}, {"CA_CB_CG", 113.83},
                {"CB_CG_CD", 111.79}, {"CG_CD_CE", 111.79}, {"CD_CE_NZ", 112.25}
            };
            // Methionine (M/MET)
            output.emplace("MET", AminoAcid(
                'M', "MET", "Methionine",
                { "N", "CA", "C", "O", "CB", "CG", "SD", "CE"}, // atoms
                {{"O", {"N", "CA", "C"}}, {"CB", {"O", "C", "CA"}},
                {"CG", {"N", "CA", "CB"}}, {"SD", {"CA", "CB", "CG"}},
                {"CE", {"CB", "CG", "SD"}}},
                { "N", "CA", "C", "CB", "O", "CG", "SD", "CE" }
            ));
            output["MET"].bondLengths = {
                {"CA_CB", 1.53}, {"C_O", 1.23}, {"CB_CG", 1.52}, {"CG_SD", 1.8},
                {"SD_CE", 1.79}
            };
            output["MET"].bondAngles = {
                {"CA_C_O", 120.148}, {"C_CA_CB", 110.833}, {"CA_CB_CG", 113.68},
                {"CB_CG_SD", 112.773}, {"CG_SD_CE", 100.61}
            };
            // Phenylalanine (F/PHE)
            output.emplace("PHE", AminoAcid(
                'F', "PHE", "Phenylalanine",
                { "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"}, // atoms
                {{"O", {"N", "CA", "C"}}, {"CB", {"O", "C", "CA"}},
                {"CG", {"N", "CA", "CB"}}, {"CD1", {"CA", "CB", "CG"}},
                {"CD2", {"CA", "CB", "CG"}}, {"CE1", {"CB", "CG", "CD1"}},
                {"CE2", {"CB", "CG", "CD2"}}, {"CZ", {"CG", "CD1", "CE1"}}},
                { "N", "CA", "C", "CB", "O", "CG", "CD1", "CD2", "CE1", "CE2", "CZ" }
            ));
            output["PHE"].bondLengths = {
                {"CA_CB", 1.53}, {"C_O", 1.23}, {"CB_CG", 1.51}, {"CG_CD1", 1.385},
                {"CG_CD2", 1.385}, {"CD1_CE1", 1.385}, {"CD2_CE2", 1.385},
                {"CE1_CZ", 1.385}
            };
            output["PHE"].bondAngles = {
                {"CA_C_O", 120.283}, {"C_CA_CB", 110.846}, {"CA_CB_CG", 114.0},
                {"CB_CG_CD1", 120.0}, {"CB_CG_CD2", 120.0}, {"CG_CD1_CE1", 120.0},
                {"CG_CD2_CE2", 120.0}, {"CD1_CE1_CZ", 120.0}
            };
            // Proline (P/PRO)
            output.emplace("PRO", AminoAcid(
                'P', "PRO", "Proline", // name
                { "N", "CA", "C", "O", "CB", "CG", "CD"}, // atoms
                {{"O", {"N", "CA", "C"}}, {"CB", {"O", "C", "CA"}},
                {"CG", {"N", "CA", "CB"}}, {"CD", {"CA", "CB", "CG"}}},
                { "N", "CA", "C", "CB", "O", "CG", "CD" } // atoms
            ));
            output["PRO"].bondLengths = {
                {"CA_CB", 1.53}, {"C_O", 1.23}, {"CB_CG", 1.49}, {"CG_CD", 1.50}
            };
            output["PRO"].bondAngles = {
                {"CA_C_O", 120.6}, {"C_CA_CB", 111.372}, {"CA_CB_CG", 104.21},
                {"CB_CG_CD", 105.0}
            };
            // Serine (S/SER)
            output.emplace("SER", AminoAcid(
                'S', "SER", "Serine",
                { "N", "CA", "C", "O", "CB", "OG"}, // atoms
                {{"O", {"N", "CA", "C"}}, {"CB", {"O", "C", "CA"}},
                {"OG", {"N", "CA", "CB"}}},
                { "N", "CA", "C", "CB", "O", "OG" }
            ));
            output["SER"].bondLengths = {
                {"CA_CB", 1.53}, {"C_O", 1.23}, {"CB_OG", 1.417}
            };
            output["SER"].bondAngles = {
                {"CA_C_O", 120.475}, {"C_CA_CB", 110.248}, {"CA_CB_OG", 111.132}
            };
            // Threonine (T/THR)
            output.emplace("THR", AminoAcid(
                'T', "THR", "Threonine",
                {"N", "CA", "C", "O", "CB", "OG1", "CG2"}, // atoms
                {{"O", {"N", "CA", "C"}}, {"CB", {"O", "C", "CA"}},
                {"OG1", {"N", "CA", "CB"}}, {"CG2", {"N", "CA", "CB"}}},
                { "N", "CA", "C", "CB", "O", "CG2", "OG1" }
            ));
            output["THR"].bondLengths = {
                {"CA_CB", 1.53}, {"C_O", 1.23}, {"CB_OG1", 1.43}, {"CB_CG2", 1.52}
            };
            output["THR"].bondAngles = {
                {"CA_C_O", 120.252}, {"C_CA_CB", 110.075}, {"CA_CB_OG1", 109.442},
                {"CA_CB_CG2", 111.457}
            };
            // Tryptophan (W/TRP)
            output.emplace("TRP", AminoAcid(
                'W', "TRP", "Tryptophan",
                {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2",
                "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"}, // atoms
                {{"O", {"N", "CA", "C"}}, {"CB", {"O", "C", "CA"}},
                {"CG", {"N", "CA", "CB"}}, {"CD1", {"CA", "CB", "CG"}},
                {"CD2", {"CA", "CB", "CG"}}, {"NE1", {"CB", "CG", "CD1"}},
                {"CE2", {"CB", "CG", "CD2"}}, {"CE3", {"CB", "CG", "CD2"}},
                {"CZ2", {"CG", "CD2", "CE2"}}, {"CZ3", {"CG", "CD2", "CE3"}},
                {"CH2", {"CD2", "CE2", "CZ2"}}},
                { "N", "CA", "C", "CB", "O", "CG", "CD1", "CD2",
                "CE2", "CE3", "NE1", "CH2", "CZ2", "CZ3" }
            ));
            output["TRP"].bondLengths = {
                {"CA_CB", 1.53}, {"C_O", 1.23}, {"CB_CG", 1.50},
                {"CG_CD1", 1.36}, {"CG_CD2", 1.44}, {"CD1_NE1", 1.38},
                {"CD2_CE2", 1.41}, {"CD2_CE3", 1.40}, {"CE2_CZ2", 1.40},
                {"CE3_CZ3", 1.384}, {"CZ2_CH2", 1.367}
            };
            output["TRP"].bondAngles = {
                {"CA_C_O", 120.178}, {"C_CA_CB", 110.852}, {"CA_CB_CG", 114.10},
                {"CB_CG_CD1", 126.712}, {"CB_CG_CD2", 126.712}, {"CG_CD1_NE1", 109.959},
                {"CG_CD2_CE2", 107.842}, {"CG_CD2_CE3", 133.975}, {"CD2_CE2_CZ2", 120.0},
                {"CD2_CE3_CZ3", 120.0}, {"CE2_CZ2_CH2", 120.0}
            };
            // Tyrosine (Y/TYR)
            output.emplace("TYR", AminoAcid(
                'Y', "TYR", "Tyrosine", // name
                {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2",
                "CE1", "CE2", "CZ", "OH"}, // atoms
                {{"O", {"N", "CA", "C"}}, {"CB", {"O", "C", "CA"}},
                {"CG", {"N", "CA", "CB"}}, {"CD1", {"CA", "CB", "CG"}},
                {"CD2", {"CA", "CB", "CG"}}, {"CE1", {"CB", "CG", "CD1"}},
                {"CE2", {"CB", "CG", "CD2"}}, {"CZ", {"CG", "CD1", "CE1"}},
                {"OH", {"CD1", "CE1", "CZ"}}},
                { "N", "CA", "C", "CB", "O", "CG", "CD1", "CD2",
                "CE1", "CE2", "OH", "CZ" }
            ));
            output["TYR"].bondLengths = {
                {"CA_CB", 1.53}, {"C_O", 1.235}, {"CB_CG", 1.51},
                {"CG_CD1", 1.39}, {"CG_CD2", 1.39}, {"CD1_CE1", 1.38},
                {"CD2_CE2", 1.38}, {"CE1_CZ", 1.378}, {"CZ_OH", 1.375}
            };
            output["TYR"].bondAngles = {
                {"CA_C_O", 120.608}, {"C_CA_CB", 110.852}, {"CA_CB_CG", 113.744},
                {"CB_CG_CD1", 120.937}, {"CB_CG_CD2", 120.937}, {"CG_CD1_CE1", 120.0},
                {"CG_CD2_CE2", 120.0}, {"CD1_CE1_CZ", 120.0}, {"CE1_CZ_OH", 120.0}
            };
            // Valine (V/VAL)
            output.emplace("VAL", AminoAcid(
                'V', "VAL", "Valine", // name
                { "N", "CA", "C", "O", "CB", "CG1", "CG2"}, // atoms
                {{"O", {"N", "CA", "C"}}, {"CB", {"O", "C", "CA"}},
                {"CG1", {"N", "CA", "CB"}}, {"CG2", {"N", "CA", "CB"}}},
                { "N", "CA", "C", "CB", "O", "CG1", "CG2" }
            ));
            output["VAL"].bondLengths = {
                {"CA_CB", 1.54}, {"C_O", 1.235}, {"CB_CG1", 1.52}, {"CB_CG2", 1.52}
            };
            output["VAL"].bondAngles = {
                {"CA_C_O", 120.472}, {"C_CA_CB", 111.381},
                {"CA_CB_CG1", 110.7}, {"CA_CB_CG2", 110.4}
            };'''

    s = ''.join([re.search('(.*?)(/|$)', l).groups()[0] for l in s.splitlines()])

    regex_expression = '.*?'.join([
        'output.emplace\("([A-Z]{3})', 
        '(\{', ')\)\);', 
        'bondLengths',
        '(\{', ');',
        'bondAngles',
        '(\{', ');',               
    ])
    parts = re.findall(regex_expression, s)
    d = tree.map_structure(lambda s: ast.literal_eval(s.replace('{','[').replace('}',']')) if '}' in s else s , parts)
    d = pd.DataFrame(d, columns=['aa', 'data', 'lengths', 'angles'])
    d['atoms'] = d['data'].apply(lambda s: s[0])
    d['torsions'] = d['data'].apply(lambda s: s[1:-1][0])
    d['alternative_atom_order'] = d['data'].apply(lambda s: s[-1])

    convert_to_dict = lambda x: {k.split('_')[-1]: v for k,v in dict(x).items()}
    def process_for_properties(r):
        atoms = r['atoms']
        # all ordered by atoms[3:]
        length_dict, angle_dict = map(convert_to_dict, (r.lengths, r.angles))
        torsions_dict = dict(r.torsions)
        lengths, angles = [np.array([d[atom] for atom in atoms[3:]]) for d in [length_dict, angle_dict]]
        angles = np.pi - (angles*np.pi/180)
        
        atom_dependency = np.vectorize(r.atoms.index)(np.array([torsions_dict[atom] for atom in atoms[3:]]))
        return lengths, angles, atom_dependency

    d[['lengths_arr', 'angles_arr', 'placement_dependency']] = d.apply(process_for_properties, axis=1).apply(pd.Series)

    lengths_arr = np.array(list(d.lengths_arr.apply(lambda x: np.concatenate([x, np.zeros(11-x.shape[0])]))))
    angles_arr = np.array(list(d.angles_arr.apply(lambda x: np.concatenate([x, np.zeros(11-x.shape[0])]))))
    placement_dependency = np.array(list(d.placement_dependency.apply(lambda x: np.concatenate([x, np.zeros((11-x.shape[0], 3), dtype=int)]))))
    atom_mask = (~(placement_dependency==0).all(-1))
    return lengths_arr, angles_arr, placement_dependency, atom_mask, d

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

# load_reference_sidechain_data pre-processed here to:
AA_REF_BOND_LENGTHS = np.array([
    [1.23 , 1.52 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [1.23 , 1.53 , 1.53 , 1.52 , 1.46 , 1.32 , 1.31 , 1.31 , 0.   , 0.   , 0.   ],
    [1.23 , 1.52 , 1.52 , 1.23 , 1.325, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [1.23 , 1.53 , 1.52 , 1.248, 1.248, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [1.23 , 1.53 , 1.8  , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [1.23 , 1.53 , 1.52 , 1.52 , 1.23 , 1.32 , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [1.23 , 1.53 , 1.52 , 1.52 , 1.25 , 1.25 , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [1.23 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [1.23 , 1.53 , 1.5  , 1.38 , 1.36 , 1.33 , 1.38 , 0.   , 0.   , 0.   , 0.   ],
    [1.235, 1.54 , 1.53 , 1.52 , 1.51 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [1.235, 1.53 , 1.53 , 1.52 , 1.52 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [1.23 , 1.53 , 1.52 , 1.52 , 1.52 , 1.49 , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [1.23 , 1.53 , 1.52 , 1.8  , 1.79 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [1.23 , 1.53 , 1.51 , 1.385, 1.385, 1.385, 1.385, 1.385, 0.   , 0.   , 0.   ],
    [1.23 , 1.53 , 1.49 , 1.5  , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [1.23 , 1.53 , 1.417, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [1.23 , 1.53 , 1.43 , 1.52 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [1.23 , 1.53 , 1.5  , 1.36 , 1.44 , 1.38 , 1.41 , 1.4  , 1.4  , 1.384, 1.367],
    [1.235, 1.53 , 1.51 , 1.39 , 1.39 , 1.38 , 1.38 , 1.378, 1.375, 0.   , 0.   ],
    [1.235, 1.54 , 1.52 , 1.52 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ]])

AA_REF_ANGLES = np.array([
    [1.04178703, 1.20686027, 0.        , 0.        , 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
    [1.05164814, 1.21162502, 1.16530398, 1.20799474, 1.18823761, 0.95658006, 1.04585365,
    1.04585365, 0.        , 0.        , 0.        ],
    [1.04173467, 1.20686027, 1.16532143, 1.03236225, 1.10863314, 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
    [1.02885414, 1.20652866, 1.16532143, 1.0761002 , 1.0761002 , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
    [1.04609799, 1.20291583, 1.15511126, 0.        , 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
    [1.04351491, 1.23045712, 1.16427424, 1.17266927, 1.01501368, 1.10828408, 0.        ,
    0.        , 0.        , 0.        , 0.        ],
    [1.0368303 , 1.21234061, 1.1550589 , 1.17090649, 1.07374401, 1.07374401, 0.        ,
    0.        , 0.        , 0.        , 0.        ],
    [1.03808693, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
    [1.03763315, 1.19853505, 1.16120246, 0.99745567, 0.86201812, 1.24635707, 1.26642836,
    0.        , 0.        , 0.        , 0.        ],
    [1.04033841, 1.1871206 , 1.21300383, 1.21300383, 1.15244091, 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
    [1.04351491, 1.214435  , 1.11526539, 1.21160757, 1.21160757, 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
    [1.03777277, 1.23045712, 1.15488437, 1.19048908, 1.19048908, 1.18246057, 0.        ,
    0.        , 0.        , 0.        , 0.        ],
    [1.04461446, 1.20719188, 1.15750236, 1.1733325 , 1.38561689, 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
    [1.04225827, 1.20696499, 1.15191731, 1.04719755, 1.04719755, 1.04719755, 1.04719755,
    1.04719755, 0.        , 0.        , 0.        ],
    [1.03672558, 1.19778456, 1.32278504, 1.30899694, 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
    [1.03890724, 1.21740206, 1.20197335, 0.        , 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
    [1.04279932, 1.22042148, 1.23146941, 1.19630103, 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
    [1.04409087, 1.20686027, 1.15017198, 0.93005105, 0.93005105, 1.22244606, 1.25939468,
    0.80328779, 1.04719755, 1.04719755, 1.04719755],
    [1.03658595, 1.20686027, 1.15638535, 1.03084382, 1.03084382, 1.04719755, 1.04719755,
    1.04719755, 1.04719755, 0.        , 0.        ],
    [1.0389596 , 1.19762748, 1.20951317, 1.21474916, 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ]])

AA_PLACEMENT_DEPENDENCIES = np.array([[
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 3,  3,  3,  3,  3,  3,  3,  0,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  0,  1,  1,  0,  1,  0,  1,  1,  1,  1,  1,  0,  0,  1,  1,  0],
    [ 0,  4,  1,  1,  0,  4,  4,  0,  1,  1,  1,  4,  4,  1,  0,  0,  0,  1,  1,  0],
    [ 0,  5,  0,  0,  0,  4,  4,  0,  4,  0,  0,  5,  0,  4,  0,  0,  0,  4,  4,  0],
    [ 0,  6,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  4,  0,  0,  0,  4,  4,  0],
    [ 0,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  5,  0,  0,  0,  4,  5,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  5,  6,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  5,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  7,  0,  0]],

    [[ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [ 2,  2,  2,  2,  2,  2,  2,  0,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
    [ 0,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [ 0,  4,  4,  4,  0,  4,  4,  0,  4,  1,  4,  4,  4,  4,  4,  0,  1,  4,  4,  1],
    [ 0,  5,  4,  4,  0,  5,  5,  0,  4,  4,  4,  5,  5,  4,  0,  0,  0,  4,  4,  0],
    [ 0,  6,  0,  0,  0,  5,  5,  0,  5,  0,  0,  6,  0,  5,  0,  0,  0,  5,  5,  0],
    [ 0,  7,  0,  0,  0,  0,  0,  0,  5,  0,  0,  0,  0,  5,  0,  0,  0,  5,  5,  0],
    [ 0,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  0,  0,  0,  5,  6,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  7,  8,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  7,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9,  0,  0]],

    [[ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
    [ 1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [ 0,  4,  4,  4,  4,  4,  4,  0,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4],
    [ 0,  5,  5,  5,  0,  5,  5,  0,  5,  4,  5,  5,  5,  5,  5,  0,  4,  5,  5,  4],
    [ 0,  6,  5,  5,  0,  6,  6,  0,  5,  5,  5,  6,  6,  5,  0,  0,  0,  5,  5,  0],
    [ 0,  7,  0,  0,  0,  6,  6,  0,  6,  0,  0,  7,  0,  6,  0,  0,  0,  6,  6,  0],
    [ 0,  8,  0,  0,  0,  0,  0,  0,  7,  0,  0,  0,  0,  7,  0,  0,  0,  7,  7,  0],
    [ 0,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8,  0,  0,  0,  7,  8,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9, 10,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 11,  0,  0]]]).T

AA_REF_ATOM_MASK = ~(AA_PLACEMENT_DEPENDENCIES==0).all(-1)
AA_N_SC_ATOMS = np.concatenate([
    np.array(AA_REF_ATOM_MASK.sum(-1), dtype=int),
    np.zeros(4, dtype=int) # for B,Z,*,X
])

# {"N_TO_CA", 1.46}, {"CA_TO_C", 1.52}, {"C_TO_N", 1.33}
BACKBONE_BOND_LENGTHS = np.array([1.33,1.46,1.52])

ATOM_ORDER = {
 'ALA': ['N', 'CA', 'C', 'O', 'CB'],
 'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
 'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
 'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
 'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
 'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
 'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
 'GLY': ['N', 'CA', 'C', 'O'],
 'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
 'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
 'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
 'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
 'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
 'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
 'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
 'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
 'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
 'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
 'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
 'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2']}