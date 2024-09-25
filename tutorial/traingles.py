import sys
import gemmi
import json
import pprint


if len(sys.argv) < 2:
    sys.stderr.write("ERROR: Input PDB/mmCIF was not provided.\n")
    sys.exit(1)
inputName = sys.argv[1]
prefix = sys.argv[1].split(".")[0]

factor = 1.3
max_dist = 3
st = gemmi.read_structure(inputName)
st.setup_entities()

metals = []
triangles = []
atom_pairs = []
atom_cra_pairs = []
ns = gemmi.NeighborSearch(st[0], st.cell, 4.2).populate(include_h=False)
for n_ch, chain in enumerate(st[0]):
    for n_res, res in enumerate(chain):
        for n_atom, atom in enumerate(res):
            if atom.element.is_metal:
                metals.append(atom)
print(len(metals), " metals")
for metal in metals:
    print(metal)
    marks1 = ns.find_neighbors(metal, min_dist=0.1, max_dist=max_dist)
    # print("  ", len(marks1), " connections")
    for mark1 in marks1:
        cra1 = mark1.to_cra(st[0])
        # print("  ", cra1.atom)
        marks2 = ns.find_neighbors(cra1.atom, min_dist=0.1, max_dist=max_dist)
        # print("    ", len(marks2), " connections")
        marks2_filtered = []
        for mark2 in marks2:
            cra2 = mark2.to_cra(st[0])
            dist_cra1_cra2 = cra1.atom.pos.dist(cra2.atom.pos)
            dist_max = 1.3 * (cra1.atom.element.covalent_r + cra2.atom.element.covalent_r)
            if cra1.chain.name == cra2.chain.name and \
                    cra1.residue.seqid == cra2.residue.seqid and \
                    cra1.atom.element == cra2.atom.element and \
                    not cra1.atom.element.is_metal and \
                    not cra2.atom.element.is_metal and \
                    dist_cra1_cra2 < dist_max:
                marks2_filtered.append(mark2)
        for mark2 in marks2_filtered:
            cra2 = mark2.to_cra(st[0])
            # print("    ", cra2.atom)
            marks3 = ns.find_neighbors(cra2.atom, min_dist=0.1, max_dist=max_dist)
            # print("      ", len(marks3), " connections")
            for mark3 in marks3:
                cra3 = mark3.to_cra(st[0])
                dist_metal_cra2 = metal.pos.dist(cra2.atom.pos)
                dist_metal_cra2_max = factor * (metal.element.covalent_r + cra2.atom.element.covalent_r)
                dist_metal_cra1 = metal.pos.dist(cra1.atom.pos)
                dist_metal_cra1_max = factor * (metal.element.covalent_r + cra1.atom.element.covalent_r)
                if cra3.atom == metal and \
                        dist_metal_cra2 < dist_metal_cra2_max and \
                        dist_metal_cra1 < dist_metal_cra1_max:
                    triangle = {metal, cra1.atom, cra2.atom}
                    atom_pair = {cra1.atom, cra2.atom}
                    atom_cra_pair = {cra1, cra2}
                    dist_cra1_cra2 = cra1.atom.pos.dist(cra2.atom.pos)
                    if triangle not in triangles:
                        print("TRIANGLE!", metal, cra1.atom, cra2.atom, cra3.atom)
                        triangles.append(triangle)
                    if atom_pair not in atom_pairs:
                        print("ATOM PAIR!", cra1, cra2, "distance:", round(dist_cra1_cra2, 2))
                        atom_pairs.append(atom_pair)
                        atom_cra_pairs.append(atom_cra_pair)

pprint.pprint(triangles)                   
print("")
pprint.pprint(atom_pairs)

atoms_equivalent_all = []
for atom_cra_pair in atom_cra_pairs:
    atoms_equivalent = [{}, {}]
    atom_cra_pair = list(atom_cra_pair)
    pos_average = (atom_cra_pair[0].atom.pos + atom_cra_pair[1].atom.pos) / 2
    print("Shifted:", atom_cra_pair[0])
    print("Deleted:", atom_cra_pair[1])
    for i in range(2):
        if atom_cra_pair[i].atom.altloc == '\x00':
            altloc = ''
        else:
            altloc = atom_cra_pair[i].atom.altloc
        if atom_cra_pair[i].residue.seqid.icode.strip() == "":
            icode = "."
        else:
            icode = atom_cra_pair[i].residue.seqid.icode
        atoms_equivalent[i] = {
            "name": atom_cra_pair[i].atom.name,
            "element": atom_cra_pair[i].atom.element.name,
            "chain": atom_cra_pair[i].chain.name,
            "residue": atom_cra_pair[i].residue.name,
            "sequence": atom_cra_pair[i].residue.seqid.num,
            "icode": icode,
            "altloc": altloc,
            "symmetry": 0
        }
    atoms_equivalent_all.append(atoms_equivalent)
    atom_cra_pair[0].atom.pos = pos_average
    atom_cra_pair[1].residue.remove_atom(atom_cra_pair[1].atom.name, atom_cra_pair[1].atom.altloc, atom_cra_pair[1].atom.element)
print("")
pprint.pprint(atoms_equivalent_all)

with open(prefix + '_nopairs_equivalents.json', 'w') as f:
    json.dump(atoms_equivalent_all, f, indent=4)
outputPdbName = prefix + "_nopairs.pdb"
outputMmcifName = prefix + "_nopairs.mmcif"
st.make_mmcif_document().write_file(outputMmcifName)
try:
    st.write_pdb(outputPdbName)
except Exception as e:
    print("WARNING: Output PDB file could not be written: ")
