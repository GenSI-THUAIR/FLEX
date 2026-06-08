description = [
    {
        "description": "Find π-π stacking interactions within a specific distance range around target atoms using ChimeraX. This is the core function for π-π stacking analysis that accepts any ChimeraX atom specification and focuses on aromatic residues (Phe, Tyr, Trp, His). **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**Usage Examples:**\n- Find π-π stacking around residue 150: target_atom_spec=':150', distance_range=6.0\n- Find π-π stacking around aromatic residues 1-50: target_atom_spec=':1-50', distance_range=8.0\n- Find π-π stacking around chain A: target_atom_spec='/A', distance_range=6.0\n- Find π-π stacking around specific aromatic residues: target_atom_spec=':phe,tyr,trp', distance_range=7.0\n- Analyze whole protein π-π stacking: target_atom_spec='protein', distance_range=0 (distance_range=0 means no distance restriction)\n- Typical π-π stacking distance range: 3.0-8.0 Angstroms",
        "name": "find_pi_pi_stacking_around_atoms",
        "optional_parameters": [
            {
                "default": 8.0,
                "description": "Search distance range in Angstroms around the target atoms (default: 8.0 for π-π stacking search area)",
                "name": "distance_range",
                "type": "float",
            },
            {
                "default": 30.0,
                "description": "Maximum angle deviation from parallel/perpendicular orientation (default: 30.0 degrees)",
                "name": "angle_cutoff",
                "type": "float",
            },
            {
                "default": None,
                "description": "Output file path to save π-π stacking results (optional)",
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
                "description": "Target atom specification in ChimeraX format (e.g., ':150' for residue 150, ':150-160' for residue range)",
                "name": "target_atom_spec",
                "type": "str",
            },
        ],
    },
    {
        "description": "Find π-π stacking interactions around a specific aromatic residue within a specified distance. Simplified function for single residue π-π stacking analysis. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**Usage Examples:**\n- Find π-π stacking around phenylalanine 25: residue_num=25, distance=6.0\n- Find π-π stacking around tryptophan 100 within 8Å: residue_num=100, distance=8.0\n- Standard π-π analysis for residue 50: residue_num=50 (uses default distance=6.0)\n- This function works best with aromatic residues (Phe, Tyr, Trp, His)",
        "name": "find_residue_pi_stacking_in_range",
        "optional_parameters": [
            {
                "default": 6.0,
                "description": "Search distance in Angstroms around the aromatic residue (default: 6.0 for π-π stacking)",
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
                "description": "Aromatic residue number to analyze",
                "name": "residue_num",
                "type": "int",
            },
        ],
    },
    {
        "description": "Find π-π stacking interactions around a specific aromatic residue in a specific chain. Useful for multi-chain protein complexes where chain identification is important. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**Usage Examples:**\n- Find π-π stacking around phenylalanine 25 in chain A: chain_id='A', residue_num=25, distance=6.0\n- Find π-π stacking around tryptophan 100 in chain B: chain_id='B', residue_num=100, distance=8.0\n- Analyze aromatic interactions in specific chain: chain_id='C', residue_num=75, distance=7.0\n- This function works best with aromatic residues (Phe, Tyr, Trp, His)",
        "name": "find_chain_residue_pi_stacking",
        "optional_parameters": [
            {
                "default": 6.0,
                "description": "Search distance in Angstroms around the aromatic residue (default: 6.0 for π-π stacking)",
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
                "description": "Aromatic residue number within the specified chain",
                "name": "residue_num",
                "type": "int",
            },
        ],
    },
    {
        "description": "Analyze all π-π stacking interactions in a complete protein structure without distance restrictions. Provides a comprehensive π-π stacking network analysis focusing on aromatic residues. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**Usage Examples:**\n- Complete protein π-π stacking analysis: analyze_protein_pi_stacking(pdb_file='protein.pdb')\n- Save results to file: analyze_protein_pi_stacking(pdb_file='protein.pdb', output_file='all_pi_stacking.txt')\n- This function is ideal for getting an overview of ALL π-π stacking interactions in the entire protein structure\n- Automatically identifies interactions between Phe, Tyr, Trp, and His residues",
        "name": "analyze_protein_pi_stacking",
        "optional_parameters": [
            {
                "default": None,
                "description": "Output file path to save complete π-π stacking analysis (optional)",
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
        "description": "Find π-π stacking interactions at the interface between two molecular entities (e.g., protein-protein, domain-domain interfaces). Specialized for studying interface stability and aromatic interactions. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**Usage Examples:**\n- Find π-π stacking between chain A and B: interface_spec1='/A', interface_spec2='/B', distance=6.0\n- Analyze protein-ligand π-π interactions: interface_spec1='protein', interface_spec2='ligand', distance=6.0\n- Study domain-domain aromatic interactions: interface_spec1=':1-100', interface_spec2=':200-300', distance=7.0\n- Save interface analysis: find_interface_pi_stacking('complex.pdb', '/A', '/B', output_file='interface_pi.txt')",
        "name": "find_interface_pi_stacking",
        "optional_parameters": [
            {
                "default": 6.0,
                "description": "Search distance in Angstroms at the interface (default: 6.0 for π-π stacking)",
                "name": "distance",
                "type": "float",
            },
            {
                "default": None,
                "description": "Output file path to save interface π-π stacking results (optional)",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB file containing the molecular complex",
                "name": "pdb_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "First interface specification in ChimeraX format (e.g., '/A' for chain A)",
                "name": "interface_spec1",
                "type": "str",
            },
            {
                "default": None,
                "description": "Second interface specification in ChimeraX format (e.g., '/B' for chain B)",
                "name": "interface_spec2",
                "type": "str",
            },
        ],
    },
    {
        "description": "Find π-π stacking interactions between ligand and protein aromatic residues. Specialized function for drug design and ligand binding analysis focusing on aromatic interactions. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**Usage Examples:**\n- Find ligand-protein π-π stacking: find_ligand_protein_pi_stacking(pdb_file='complex.pdb', ligand_spec='ligand')\n- Analyze specific ligand: find_ligand_protein_pi_stacking(pdb_file='complex.pdb', ligand_spec='het', distance=5.0)\n- Chain-specific ligand: find_ligand_protein_pi_stacking(pdb_file='complex.pdb', ligand_spec='/A:401', distance=7.0)\n- Study drug-target aromatic interactions: ligand_spec='drug', distance=6.0",
        "name": "find_ligand_protein_pi_stacking",
        "optional_parameters": [
            {
                "default": "ligand",
                "description": "Ligand specification in ChimeraX format (default: 'ligand'). Can be specific like 'het', chain/residue specs",
                "name": "ligand_spec",
                "type": "str",
            },
            {
                "default": 6.0,
                "description": "Search distance in Angstroms around the ligand (default: 6.0 for π-π stacking)",
                "name": "distance",
                "type": "float",
            },
            {
                "default": None,
                "description": "Output file path to save ligand-protein π-π stacking results (optional)",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB file containing protein-ligand complex",
                "name": "pdb_file",
                "type": "str",
            },
        ],
    },
    {
        "description": "Find clusters of aromatic residues that may form π-π stacking networks. Identifies groups of aromatic residues (Phe, Tyr, Trp, His) that are spatially close and may participate in cooperative π-π interactions. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**Usage Examples:**\n- Find aromatic clusters: find_aromatic_clusters(pdb_file='protein.pdb', cluster_distance=8.0, min_cluster_size=3)\n- Identify π-π networks: cluster_distance=6.0, min_cluster_size=2\n- Study aromatic core regions: cluster_distance=10.0, min_cluster_size=4\n- Save cluster analysis: find_aromatic_clusters('protein.pdb', output_file='aromatic_clusters.txt')",
        "name": "find_aromatic_clusters",
        "optional_parameters": [
            {
                "default": 8.0,
                "description": "Maximum distance for clustering aromatic residues in Angstroms (default: 8.0)",
                "name": "cluster_distance",
                "type": "float",
            },
            {
                "default": 3,
                "description": "Minimum number of aromatic residues required to form a cluster (default: 3)",
                "name": "min_cluster_size",
                "type": "int",
            },
            {
                "default": None,
                "description": "Output file path to save aromatic cluster results (optional)",
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
        ],
    },
]
