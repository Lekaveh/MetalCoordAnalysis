MetalCoord tutorial
===================

In this tutorial, we will show how to use the *MetalCoord* program to analyse the coordination of metal atoms in macromolecular structures. We will use the results to create the external restraints describing ideal distances and angles for interactions of metal atoms that can be used in structure refinement.

*MetalCoord* is a program developed in Python and can be installed according to `these instructions <https://github.com/Lekaveh/MetalCoordAnalysis?tab=readme-ov-file#installation>`_. For this tutorial, you will need a command line; in MS Windows, you can use PowerShell, Command Prompt, CCP4Console or Windows Subsystem for Linux (WSL).


Aluminium trifluoride - unusual coordination
--------------------------------------------

Aluminium trifluoride (monomer code `AF3 <https://www.rcsb.org/ligand/AF3>`_) is typically a planar molecule with the aluminium atom in the centre and possessing 3-fold symmetry. However, in the structure of dUTPase (PDB entry `4dl8 <https://www3.rcsb.org/structure/4dl8>`_), the molecule shape is different owing to an extra interaction of the aluminium with an oxygen atom of phosphate.

Thus, the standard monomer dictionary for aluminium trifluoride is not suitable for this particular case. Let’s download the structure model and diffraction data in one directory:

 - https://files.rcsb.org/download/4dl8.cif
 - https://files.rcsb.org/download/4dl8-sf.cif

Let's carry out the analysis of metal coordination using MetalCoord in the stats mode. Moreover, there are also other metals in the 4dl8 structure: sodium and magnesium ions. We should analyse all the metal atoms and ions in the structure::

   metalCoord stats -l AF3 -p 4dl8.cif -o 4dl8_AF3_mc.json
   metalCoord stats -l MG -p 4dl8.cif -o 4dl8_MG_mc.json
   metalCoord stats -l NA -p 4dl8.cif -o 4dl8_NA_mc.json

The results are now available in output JSON files. A JSON file is a text file with a particular syntax, they can be opened in any text editor (e.g. notepad, gedit, vim) or a web browser. Let’s inspect the file `4dl8_AF3_mc.json`::

   [
       {
           "chain": "A",
           "residue": "AF3",
           "sequence": 302,
           "metal": "AL",
           "metalElement": "Al",
           "icode": ".",
           "altloc": "",
           "ligands": [
               {
                   "class": "octahedral",
                   "procrustes": 0.067,
                   "coordination": 6,
                   "count": 33,
                   "description": "Based on coordination and all atoms availability only",
                   "base": [
                       {
                           "ligand": {
                               "name": "F2",
                               "element": "F",
                               "chain": "A",
                               "residue": "AF3",
                               "sequence": 302,
                               "icode": ".",
                               "altloc": ""
                           },
                           "distance": [
                               1.82
                           ],
                           "std": [
                               0.05
                           ]
                       },
                       ### etc. other atoms close to the aluminium within the monomer AF3 A302
                   ],
                   "angles": [
                       {
                           "ligand1": {
                               "name": "F2",
                               "element": "F",
                               "chain": "A",
                               "residue": "AF3",
                               "sequence": 302,
                               "icode": ".",
                               "altloc": ""
                           },
                           "ligand2": {
                               "name": "F1",
                               "element": "F",
                               "chain": "A",
                               "residue": "AF3",
                               "sequence": 302,
                               "icode": ".",
                               "altloc": ""
                           },
                           "angle": 90.0,
                           "std": 5.0
                       },
                       ### etc. other angles relevant for the aluminium in AF3 A302
                   ],
                   "pdb": [
                       {
                           "ligand": {
                               "name": "O1",
                               "element": "O",
                               "chain": "A",
                               "residue": "PO4",
                               "sequence": 303,
                               "icode": ".",
                               "altloc": ""
                           },
                           "distance": [
                               1.88
                           ],
                           "std": [
                               0.04
                           ]
                       },
                       ### etc. other atoms from other monomers/residues close to the aluminium
                   ]
               }
           ]
       }
   ]

For every metal atom, possible coordinations are reported. In this case, there is only one metal atom reported: AF3 A302/AL. For this atom, only one coordination is described: octahedral class with a coordination of six. The procrustes score is 0.067 which means a good agreement of the coordination geometry of aluminium in the 4dl8 structure in comparison with the reference structures which were used by *MetalCoord*. (A higher procrustes score denotes a worse agreement.)

Further, for a particular metal atom considering a particular coordination, the report consists of three sections: *base*, *angles* and *pdb*.  All the stated ideal values have also their standard deviations denoted as *std*.

 - The section *base* refers to ideal distances within the same monomer.
 - The section *angles* refers to ideal angles.
 - The section *pdb* refers to intermolecular ideal distances. This information is to be used for the definition of connections in a mmCIF file (formerly also known as LINK records in the PDB file format).

We can now generate external bond length and angle restraints for *Refmacat* or *Servalcat* using the script *json2restraints.py*. We have to specify all the JSON files from *MetalCoord*. We also need to specify the structure model PDB/mmCIF file to apply the connections reported in the *pdb* section::

   python json2restraints.py -i 4dl8_AF3_mc.json 4dl8_MG_mc.json 4dl8_NA_mc.json -p 4dl8.cif -o 4dl8_mc_restraints

The script creates five files:

 - 4dl8_mc_restraints.txt is a keyword file for *Servalcat* or *Refmacat* containing the external bond length and angle restraints based on a coordination class with the lowest procrustes score.
 - 4dl8_mc_restraints_coot.txt is the same keyword file in a simplified format which is compatible with *Coot*. It can be loaded in Coot using Calculate -> Modules -> Restraints and then Restraints -> Read Refmac Extra Restraints. The file does not include restraints for atoms with alternative conformations and atoms from symmetry-related molecules.
 - 4dl8_mc_restraints.def is a keyword file compatible with *Phenix.refine*. The file does not include restraints for atoms from symmetry-related molecules.
 - 4dl8_mc_restraints.mmcif and 4dl8_mc_restraints.pdb are structure models with updated connection/link records related to the interactions with metals. Note that the script deletes all the connection/link records specified in the input PDB/mmCIF file - this behaviour can be turned off using an extra option ``--keep-links`` but a user should be very careful and check the result to avoid inconsistencies in the connection/link records.

Now we can refine the structure model in *Servalcat* using the external restraints::

   servalcat refine_xtal_norefmac --hklin 4dl8-sf.cif --model 4dl8_mc_restraints.mmcif -s xray --keyword_file 4dl8_mc_restraints.txt -o 4dl8_servalcat

After the refinement, the bond length and angle outliers reported in the refinement log file should be checked.


Excluding irrelevant atoms from the analysis
--------------------------------------------

It is generally recommended to check results reported in a JSON file. The current algorithm in *MetalCoord* analyses metal coordination in terms of geometry but not in terms of chemistry. Although some atoms can be close enough to a metal atom to be taken into consideration, a reported interaction would not have a chemical sense. For instance, consider the crystal structure of laccase (PDB entry `3kw8 <https://www3.rcsb.org/structure/3kw8>`_) which includes four copper ions. We can download the structure https://files.rcsb.org/download/3kw8.cif and run *MetalCoord* for all the metal atoms and ions using the default options::

   metalCoord stats -l NA -p 3kw8.cif -o 3kw8_NA_mc.json
   metalCoord stats -l CU -p 3kw8.cif -o 3kw8_CU_mc.json

However, for the copper ion A401/CU, there is a reported coordination of 4 and an ideal distance to the sulphur atom of a methionine residue A298/SD of 2.40 angstrom.

::

   [
    {
        "chain": "A",
        "residue": "CU",
        "sequence": 401,
        "metal": "CU",
        "metalElement": "Cu",
        "icode": ".",
        "altloc": "",
        "ligands": [
            {
                "class": "tetrahedral",
                "procrustes": 0.235,
                "coordination": 4,
                "count": 328,
                "description": "Strict correspondence",
                "base": [],
                "angles": [
                    {
                    ### angles
                "pdb": [
                    ### other atoms:
                    ### CYS A288/SG
                    ### HIS A293/ND1
                    ### HIS A231/ND1
                    {
                        "ligand": {
                            "name": "SD",
                            "element": "S",
                            "chain": "A",
                            "residue": "MET",
                            "sequence": 298,
                            "icode": ".",
                            "altloc": "",
                            "symmetry": 0
                        },
                        "distance": [
                            2.4
                        ],
                        "std": [
                            0.12
                        ]
                    }
                ]
            },
            ### ...
            }
        ]
    }
   ]

When the structure is inspected *e.g.* in *Coot*, it is obvious that this ideal distance is wrong as both atoms A401/CU and A298/SD are clearly located in the observed electron density and their distance is much higher: 3.49 angstrom. Therefore, we need to exclude the atom A298/SD from the analysis. To do so, we can specify the optional argument ``-d DISTANCE_THRESHOLD_D`` to modify the maximum distance of atoms from a metal atom which are taken into account. A threshold to select atoms is (*r1* + *r2*)*(1 + *d*) where *r1* and *r2* are covalent radii. The default value of *d* is 0.5. Let's lower the value of *d*::

   metalCoord stats -l CU -p 3kw8.cif -o 3kw8_CU_mc-d0p4.json -d 0.4

In the output file 3kw8_FE_mc-d0p4.json, we can now see that the reported coordination for A401/CU is three, the procrustes score is improved and the atom A298/SD is not taken into account.

An alternative way how to achieve the same is to set the optional argument ``-c MAXIMUM_COORDINATION`` to specify a lower maximum coordination number::

   metalCoord stats -l CU -p 3kw8.cif -o 3kw8_CU_mc-c3.json -c 3

However, this setting affects also the other copper ions in the structure. And some of them have actually a coordination of four. Thus, this approach would lead to suboptimal results for them.

Another example of the described issue with reporting irrelevant atoms can be found *e.g.* in the structure of photosystem II (PDB entry `1s5l <https://www3.rcsb.org/structure/1s5l>`_). Ideal distances between a magnesium centre of chlorophyll (monomer code CLA) and a carbon atom of adjacent chlorophyll molecule (C20, CMB, ...) are reported (*e.g.* C481/MG - C480/C20 or B513/MG - B523/CMB). Such interactions between metal and carbon atoms are not relevant in terms of chemistry. To solve this case, the options ``-c 5 -d 0.35`` can be used to exclude all the irrelevant carbon atoms.


Multiple coordination - Manual modification of output JSON file
---------------------------------------------------------------

It is also worth checking whether *MetalCoord* found multiple possible coordination classes - they are sorted by their procrustes score. If a structure consists of more copies of the same chain, it is recommended to check the consistency in results for individual chains.

It can happen that two coordination classes with similar procrustes scores are reported in a JSON file. If a refinement using the restraints corresponding to the first coordination class was problematic (*e.g.* several bond length or angles outliers relating to a metal reported in a refinement log file), then it is recommended to try also restraints based on the other coordination class reported in the JSON file from *MetalCoord*. It is also possible to set a procrustes score threshold using the optional argument ``-t PROCRUSTES_THRESHOLD``, the default procrustes score threshold value is 0.3.

Generally, the JSON files can be manually modified to remove parts of a report which could be considered irrelevant for any reason. It is just necessary to keep the files in the `JSON format syntax <https://www.w3schools.com/js/js_json_syntax.asp>`_, *i.e.* do not break the structure of brackets. The syntax of a modified JSON file should be checked using a JSON format validator.


Metal-atom-atom-Metal triangles
-------------------------------

A special category of the metal containing ligands are molecules which include a *triangle* of interactions: metal-atom-atom-metal where the last and the first metal is the same. An example of a such case is the Cu2O2 cluster (monomer code `CUO <https://www.rcsb.org/ligand/CUO>`_) in the crystal structure of hemocyanin functional unit CCHB-g (PDB entry `8tnv <https://www3.rcsb.org/structure/8tnv>`_). In the cluster, there are triangles CU1-O1-O2-CU1 and CU2-O1-O2-CU2.

In these cases, the non-metal atom pairs (oxygen atoms O1 and O2 in the example case) cause issues in the *MetalCoord* analysis as it is difficult to characterise an interaction between them. Thus, the following procedure is currently recommended:

1) Identification of the metal-atom-atom-metal interactions in an input PDB/mmCIF file. Extracting the relevant non-metal atom pairs.

2) Declaration of a new helper atom which is placed in between the atoms in the identified pairs. The original pairs of atoms are temporarily deleted from the structure. The information about the atom pairs is kept in a new separate file with the suffix ``_nopairs_equivalents.json``. Let’s download the structure model and diffraction data in one directory:

   https://files.rcsb.org/download/8tnv.cif

   https://files.rcsb.org/download/8tnv-sf.cif

   The first two steps are automated in an extra script ``traingles.py``::

    python traingles.py 8tnv.cif

3) Then *MetalCoord* can be run using the temporary structure as input::

    metalCoord stats -l CUO -p 8tnv_nopairs.mmcif -o 8tnv_nopairs_mc.json

4) Generate the external restraints while applying the bond length for the helper atom to the whole pair of atoms. The bond angle restraints for these atoms are not applied::

    python3 json2restraints.py -i 8tnv_nopairs_mc.json -o 8tnv_mc_restraints -p 8tnv.cif -e 8tnv_nopairs_equivalents.json

At the end of the file ``8tnv_mc_restraints.txt``, these lines describing the distance of the pairs of atoms can be added manually::

   exte dist first chain A resi 402 inse . atom O1 second chain A resi 402 inse . atom O2 value 1.6 sigma 0.2 type 0
   exte dist first chain B resi 406 inse . atom O1 second chain B resi 406 inse . atom O2 value 1.6 sigma 0.2 type 0

Now it is possible to refine the structure in *Servalcat* while taking the external restrains into account::

   servalcat refine_xtal_norefmac --hklin 8tnv-sf.cif --model 8tnv_mc_restraints.mmcif -s xray --ncsr --ligand CUO_out_final.cif --keyword_file 8tnv_mc_restraints.txt -o 8tnv_servalcat_restraints --adp aniso

Other examples of monomers with such traingles are `C4R <https://www.rcsb.org/ligand/C4R>`_, `DVW <https://www.rcsb.org/ligand/DVW>`_, `J9H <https://www.rcsb.org/ligand/J9H>`_, `RMD <https://www.rcsb.org/ligand/RMD>`_. A special case is `PLL <https://www.rcsb.org/ligand/PLL>`_ where three atoms would need to be replaced with one.

Note: From these considerations, we exclude *sandwich*-like metal containing ligands or cases where both non-metal atoms involved above are members of the same ring, *e.g.* monomer codes `JSC <https://www.rcsb.org/ligand/JSC>`_ or `4IR <https://www.rcsb.org/ligand/4IR>`_.
