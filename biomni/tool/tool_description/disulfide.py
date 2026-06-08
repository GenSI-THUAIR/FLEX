description = [
    {
        "description": "Find disulfide bonds within a specific distance range around target atoms using ChimeraX. This is the core function for disulfide bond analysis that accepts any ChimeraX atom specification. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**Usage Examples:**\n- Find disulfide bonds around residue 150: target_atom_spec=':150', distance_range=8.0\n- Find disulfide bonds around cysteine residues 1-50: target_atom_spec=':1-50', distance_range=8.0\n- Find disulfide bonds around chain A: target_atom_spec='/A', distance_range=8.0\n- Find disulfide bonds around specific cysteine sulfur atoms: target_atom_spec=':150@SG', distance_range=6.0\n- Analyze whole protein disulfide bonds: target_atom_spec='protein', distance_range=0 (distance_range=0 means no distance restriction)\n- Typical disulfide bond distance range: 1.8-3.0 Angstroms",
        "name": "find_disulfide_bonds_around_atoms",
        "optional_parameters": [
            {
                "default": 8.0,
                "description": "Search distance range in Angstroms around the target atoms (default: 8.0 for disulfide bond search area)",
                "name": "distance_range",
                "type": "float",
            },
            {
                "default": None,
                "description": "Output file path to save disulfide bond results (optional)",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB file to analyze",
                "name": "pdb_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Target atom specification in ChimeraX format (e.g., ':150@SG' for sulfur atom in residue 150, ':150-160' for residue range)",
                "name": "target_atom_spec",
                "type": "str",
            },
        ],
    },
    {
        "description": "Find disulfide bonds around a specific cysteine residue within a specified distance. Simplified function for single residue disulfide bond analysis. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**Usage Examples:**\n- Find disulfide bonds around cysteine 25: residue_num=25, distance=8.0\n- Find disulfide bonds around cysteine 100 within 6Å: residue_num=100, distance=6.0\n- Standard disulfide analysis for residue 50: residue_num=50 (uses default distance=8.0)\n- This function automatically targets the sulfur atom (@SG) of the specified cysteine residue",
        "name": "find_residue_disulfide_bonds_in_range",
        "optional_parameters": [
            {
                "default": 8.0,
                "description": "Search distance in Angstroms around the cysteine residue (default: 8.0 for disulfide bonds)",
                "name": "distance",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB file to analyze",
                "name": "pdb_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Cysteine residue number to analyze",
                "name": "residue_num",
                "type": "int",
            },
        ],
    },
    {
        "description": "Find disulfide bonds around a specific cysteine residue in a specific chain. Useful for multi-chain protein complexes where chain identification is important. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**Usage Examples:**\n- Find disulfide bonds around cysteine 25 in chain A: chain_id='A', residue_num=25, distance=8.0\n- Find disulfide bonds around cysteine 100 in chain B: chain_id='B', residue_num=100, distance=6.0\n- Analyze inter-chain disulfide potential: chain_id='C', residue_num=75, distance=10.0\n- This function automatically targets the sulfur atom (@SG) of the specified cysteine residue in the given chain",
        "name": "find_chain_residue_disulfide_bonds",
        "optional_parameters": [
            {
                "default": 8.0,
                "description": "Search distance in Angstroms around the cysteine residue (default: 8.0 for disulfide bonds)",
                "name": "distance",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB file to analyze",
                "name": "pdb_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Chain identifier (e.g., 'A', 'B', 'C')",
                "name": "chain_id",
                "type": "str",
            },
            {
                "default": None,
                "description": "Cysteine residue number within the specified chain",
                "name": "residue_num",
                "type": "int",
            },
        ],
    },
    {
        "description": "Analyze all disulfide bonds in a complete protein structure without distance restrictions. Provides a comprehensive disulfide bond network analysis. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**Usage Examples:**\n- Complete protein disulfide bond analysis: analyze_protein_disulfide_bonds(pdb_file='protein.pdb')\n- Save results to file: analyze_protein_disulfide_bonds(pdb_file='protein.pdb', output_file='all_disulfides.txt')\n- This function is ideal for getting an overview of ALL disulfide bonds in the entire protein structure\n- Identifies both intra-chain and inter-chain disulfide bonds automatically",
        "name": "analyze_protein_disulfide_bonds",
        "optional_parameters": [
            {
                "default": None,
                "description": "Output file path to save complete disulfide bond analysis (optional)",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB file containing the protein structure",
                "name": "pdb_file",
                "type": "str",
            },
        ],
    },
    {
        "description": "Find disulfide bonds between two different protein chains. Specialized function for analyzing inter-chain disulfide bonds in protein complexes. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**Usage Examples:**\n- Find disulfide bonds between chain A and B: find_inter_chain_disulfide_bonds(pdb_file='complex.pdb', chain1='A', chain2='B')\n- Analyze antibody heavy-light chain disulfides: chain1='H', chain2='L'\n- Study insulin A-B chain connections: chain1='A', chain2='B'\n- Save inter-chain analysis: find_inter_chain_disulfide_bonds('complex.pdb', 'A', 'B', output_file='inter_chain.txt')",
        "name": "find_inter_chain_disulfide_bonds",
        "optional_parameters": [
            {
                "default": None,
                "description": "Output file path to save inter-chain disulfide bond results (optional)",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB file containing the protein complex",
                "name": "pdb_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "First chain identifier (e.g., 'A', 'H', '1')",
                "name": "chain1",
                "type": "str",
            },
            {
                "default": None,
                "description": "Second chain identifier (e.g., 'B', 'L', '2')",
                "name": "chain2",
                "type": "str",
            },
        ],
    },
    {
        "description": "Find disulfide bonds within a single protein chain. Specialized function for analyzing intra-chain disulfide bonds that contribute to protein folding stability. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**Usage Examples:**\n- Find intra-chain disulfides in chain A: find_intra_chain_disulfide_bonds(pdb_file='protein.pdb', chain_id='A')\n- Analyze single domain stability: chain_id='B'\n- Study individual chain folding: chain_id='C'\n- Save intra-chain analysis: find_intra_chain_disulfide_bonds('protein.pdb', 'A', output_file='intra_chain.txt')",
        "name": "find_intra_chain_disulfide_bonds",
        "optional_parameters": [
            {
                "default": None,
                "description": "Output file path to save intra-chain disulfide bond results (optional)",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB file to analyze",
                "name": "pdb_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Chain identifier to analyze (e.g., 'A', 'B', 'C')",
                "name": "chain_id",
                "type": "str",
            },
        ],
    },
    
]
