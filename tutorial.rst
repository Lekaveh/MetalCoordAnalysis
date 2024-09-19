MetalCoord tutorial
===================

In this tutorial, we will show how to use the *MetalCoord* program to analyse coordination of metal atoms in macromolecular structures. We will use the results for creating the external restraints describing ideal distances and angles for interactions of metal atoms that can be used in structure refinement.

*MetalCoord* is a program developed in Python and can be installed according to `these instructions <https://github.com/Lekaveh/MetalCoordAnalysis?tab=readme-ov-file#installation>`_. For this tutorial, you will need a command line; in MS Windows, you can use PowerShell, Command Prompt or CCP4Console.

Aluminium trifluoride
---------------------

Aluminium trifluoride (monomer code `AF3 <https://www.rcsb.org/ligand/AF3>`_) is typically a planar molecule with the aluminium atom in the centre and possessing 3-fold symmetry. However, in the structure of dUTPase (PDB entry `4dl8 <https://www3.rcsb.org/structure/4dl8>`_), the molecule shape is different owing to an extra interaction of the aluminium with an oxygen atom of phosphate.

Thus, the standard monomer dictionary for aluminium trifluoride is not suitable for this particular case. Let’s download the structure and carry out the analysis of metal coordination using MetalCoord in the stats mode. Moreover, there are also other metals in the 4dl8 structure: sodium and magnesium ions. So we analyse them as well::

   wget https://files.rcsb.org/download/4dl8.cif
   wget https://files.rcsb.org/ligands/download/AF3.cif
   metalCoord stats -l AF3 -p 4dl8.cif -o 4dl8_AF3_mc.json
   metalCoord stats -l MG -p 4dl8.cif -o 4dl8_MG_mc.json
   metalCoord stats -l NA -p 4dl8.cif -o 4dl8_NA_mc.json

The results are now available in output JSON files. A JSON file is a text file with a particular syntax, they can be opened in any text editor (e.g. notepad, gedit, vim) or a web browser. Let’s inspect the file `4dl8_AF3_mc.json`. For every metal atom, possible coordinations are reported. In this case, there is only one metal atom reported: AF3 A302/AL. For this atom, only one coordination is described: octahedral class with a coordination of six. The procrustes score is 0.067 which means a good agreement of the coordination geometry of aluminium in the 4dl8 structure in comparison with the reference structures which were used by *MetalCoord*. (A higher procrustes score denotes a worse agreement.)

Further, for an particular atom considering a particular coordination, the report consists of three sections: *base*, *angles* and *pdb*.  All the stated ideal values have also their standard deviations denoted as *std*.
 - The section *base* refers to ideal distances within the same monomer.
 - The section *pdb* refers to ideal distances which are considered intermolecular interactions,
 - The section *angles* refers to intermolecular ideal distances. This information is to be used for the definition of connections in a mmCIF file (formerly also known as LINK records in the PDB file format).

We can now generate external bond length and angle restraints for *Refmacat* or *Servalcat* using the script *json2restraints.py*. We have to specify all the JSON files from *MetalCoord*. We also need to specify the structure model mmCIF file to apply the connections reported in the *pdb* section::

   python json2restraints.py -i 4dl8_AF3_mc.json 4dl8_MG_mc.json 4dl8_NA_mc.json -p 4dl8.cif -o 4dl8_mc_restraints

The script creates four files:
 - 4dl8_mc_restraints.txt is a keyword file for *Servalcat* or *Refmacat* containing the external bond length and angle restraints based on a coordination class with the lowest procrustes score.
 - 4dl8_mc_restraints_coot.txt is the same keyword file in a simplified format which is compatible with *Coot*. It can be loaded in Coot using Calculate -> Modules -> Restraints and then Restraints -> Read Refmac Extra Restraints.
 - 4dl8_mc_restraints.mmcif and 4dl8_mc_restraints.pdb are structure models with updated connection/link records related to the interactions with metals.

Now we can refine the structure model in *Servalcat* using the external restraints::

   wget https://files.rcsb.org/download/4dl8-sf.cif
   servalcat refine_xtal_norefmac --hklin 4dl8-sf.cif --model 4dl8_mc_restraints.mmcif -s xray --keyword_file 4dl8_mc_restraints.txt -o 4dl8_servalcat

Report details and common issues
--------------------------------

It is generally recommended to check results reported in a JSON file. The current algorithm in *MetalCoord* analyses metal coordination in terms of geometry but not in terms of chemistry. Although some atoms can be close enough to a metal atom to be taken into consideration, a reported interaction would not have a chemical sense. For instance consider the following case: an interaction of an iron (possessing a coordination of 4) with a carbon atom is reported in a JSON file. That is not correct in terms of chemistry. In such a case, MetalCoord can be rerun with the optional argument ``-c MAXIMUM_COORDINATION`` to specify a lower maximum coordination number::

   metalCoord stats -l FE -p structure.cif -o mc_output.json -c 3

Another way is to specify the optional argument ``-d DISTANCE_THRESHOLD`` to set a maximum distance of atoms from a metal atom (in angstrom) which are taken into account::

   metalCoord stats -l FE -p structure.cif -o mc_output.json -d 2.1

It is also worth checking whether *MetalCoord* found multiple possible coordination classes - they are sorted by their procrustes score. It can happen that two coordination classes with similar procrustes scores are reported in a JSON file. If a refinement using the restraints corresponding to the first coordination class was problematic (*e.g.* several bond length or angles outliers relating to a metal reported in a refinement log file), then it is recommended to try also restraints based on the other coordination class reported in the JSON file from *MetalCoord*. It is also possible to set a procrustes score threshold using the optional argument ``-t PROCRUSTES_THRESHOLD``, the default procrustes score threshold value is 0.3.
