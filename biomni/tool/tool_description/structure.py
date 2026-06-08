description = [
    {
        "description": "Find hydrogen bonds within a specific distance range around target atoms using ChimeraX. This is the core function that accepts any ChimeraX atom specification and can handle complex selections. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **Before counting or processing results:** Examine whether the output is a list, dictionary, or other data type\n- **If the count result is 0:** This may indicate an error OR genuinely no hydrogen bonds in the specified region. Verify the atom selection and distance parameters, and re-run the analysis if needed\n\n**Usage Examples:**\n- Find H-bonds around residue 150: target_atom_spec=':150', distance_range=5.0\n- Find H-bonds around residues 1-50: target_atom_spec=':1-50', distance_range=5.0\n- Find H-bonds around chain A: target_atom_spec='/A', distance_range=5.0\n- Find H-bonds around specific atoms: target_atom_spec=':150@N*', distance_range=4.0\n- Analyze whole protein: target_atom_spec='protein', distance_range=0 (distance_range=0 means no distance restriction)",
        "name": "find_hydrogen_bonds_around_atoms",
        "optional_parameters": [
            {
                "default": 5.0,
                "description": "Search distance range in Angstroms around the target atoms",
                "name": "distance_range",
                "type": "float"
            },
            {
                "default": None,
                "description": "Output file path to save hydrogen bond results (optional)",
                "name": "output_file",
                "type": "str"
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB or CIF file to analyze",
                "name": "pdb_file",
                "type": "str"
            },
            {
                "default": None,
                "description": "Target atom specification in ChimeraX format (e.g., ':150@N*' for N atoms in residue 150, ':150-160' for residue range)",
                "name": "target_atom_spec",
                "type": "str"
            }
        ]
    },
    {
        "description": "Find hydrogen bonds around specific residue atoms within a specified distance. Simplified function for single residue analysis without chain specification. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **Before counting or processing results:** Examine whether the output is a list, dictionary, or other data type\n- **If the count result is 0:** This may indicate an error OR genuinely no hydrogen bonds in the specified region. Verify the atom selection and distance parameters, and re-run the analysis if needed\n\n**Usage Examples:**\n- Find H-bonds around residue 25 (all atoms): residue_num=25, atom_name='*', distance=5.0\n- Find H-bonds around backbone nitrogen of residue 25: residue_num=25, atom_name='N', distance=4.0\n- Find H-bonds around side chain atoms of residue 25: residue_num=25, atom_name='C*', distance=5.0",
        "name": "find_residue_hbonds_in_range",
        "optional_parameters": [
            {
                "default": "*",
                "description": "Atom name to analyze (default: all atoms '*'). Can be specific like 'CA', 'N', etc.",
                "name": "atom_name",
                "type": "str"
            },
            {
                "default": 5.0,
                "description": "Search distance in Angstroms around the residue",
                "name": "distance",
                "type": "float"
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB or CIF file to analyze",
                "name": "pdb_file",
                "type": "str"
            },
            {
                "default": None,
                "description": "Residue number to analyze",
                "name": "residue_num",
                "type": "int"
            }
        ]
    },
    {
        "description": "Find hydrogen bonds around a specific atom in a specific chain and residue. Useful for multi-chain protein complexes where chain identification is important. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **Before counting or processing results:** Examine whether the output is a list, dictionary, or other data type\n- **If the count result is 0:** This may indicate an error OR genuinely no hydrogen bonds in the specified region. Verify the atom selection and distance parameters, and re-run the analysis if needed\n\n**Usage Examples:**\n- Find H-bonds around N atom of residue 25 in chain A: chain_id='A', residue_num=25, atom_name='N', distance=5.0\n- Find H-bonds around CA atom of residue 100 in chain B: chain_id='B', residue_num=100, atom_name='CA', distance=4.0\n- Find H-bonds around side chain oxygen of serine 50 in chain A: chain_id='A', residue_num=50, atom_name='OG', distance=3.5",
        "name": "find_chain_residue_atom_hbonds",
        "optional_parameters": [
            {
                "default": 5.0,
                "description": "Search distance in Angstroms around the target atom",
                "name": "distance",
                "type": "float"
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB or CIF file to analyze",
                "name": "pdb_file",
                "type": "str"
            },
            {
                "default": None,
                "description": "Chain identifier (e.g., 'A', 'B', 'C')",
                "name": "chain_id",
                "type": "str"
            },
            {
                "default": None,
                "description": "Residue number within the specified chain",
                "name": "residue_num",
                "type": "int"
            },
            {
                "default": None,
                "description": "Specific atom name to analyze (e.g., 'CA', 'N', 'O')",
                "name": "atom_name",
                "type": "str"
            }
        ]
    },
    {
        "description": "Analyze all hydrogen bonds in a complete protein structure without distance restrictions. Provides a comprehensive hydrogen bond network analysis. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **Before counting or processing results:** Examine whether the output is a list, dictionary, or other data type\n- **If the count result is 0:** This indicates an error - complete proteins should have many hydrogen bonds. Recheck the function call and re-run the analysis\n\n**Usage Examples:**\n- Complete protein H-bond analysis: analyze_protein_hbonds(pdb_file='protein.pdb')\n- Save results to file: analyze_protein_hbonds(pdb_file='protein.pdb', output_file='all_hbonds.txt')\n- This function is ideal for getting an overview of ALL hydrogen bonds in the entire protein structure",
        "name": "analyze_protein_hbonds",
        "optional_parameters": [
            {
                "default": None,
                "description": "Output file path to save complete hydrogen bond analysis (optional)",
                "name": "output_file",
                "type": "str"
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB or CIF file containing the protein structure",
                "name": "pdb_file",
                "type": "str"
            }
        ]
    },
    {
        "description": "Find hydrogen bonds between ligand and protein within specified distance. Specialized function for drug design and ligand binding analysis. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **Before counting or processing results:** Examine whether the output is a list, dictionary, or other data type\n- **If the count result is 0:** This may indicate an error OR genuinely no ligand-protein H-bonds. Verify ligand presence and recheck if needed\n\n**Usage Examples:**\n- Find ligand-protein H-bonds: find_ligand_protein_hbonds(pdb_file='complex.pdb', ligand_spec='ligand')\n- Analyze specific ligand: find_ligand_protein_hbonds(pdb_file='complex.pdb', ligand_spec='het', distance=4.0)\n- Chain-specific ligand: find_ligand_protein_hbonds(pdb_file='complex.pdb', ligand_spec='/A:401', distance=5.0)",
        "name": "find_ligand_protein_hbonds",
        "optional_parameters": [
            {
                "default": "ligand",
                "description": "Ligand specification in ChimeraX format (default: 'ligand'). Can be specific like 'het', chain/residue specs",
                "name": "ligand_spec",
                "type": "str"
            },
            {
                "default": 5.0,
                "description": "Search distance in Angstroms around the ligand",
                "name": "distance",
                "type": "float"
            },
            {
                "default": None,
                "description": "Output file path to save ligand-protein hydrogen bond results (optional)",
                "name": "output_file",
                "type": "str"
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB or CIF file containing protein-ligand complex",
                "name": "pdb_file",
                "type": "str"
            }
        ]
    },
    {
        "description": "Find salt bridges within a specific distance range around target atoms using ChimeraX. Core function for salt bridge analysis that accepts any ChimeraX atom specification. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **Before counting or processing results:** Examine whether the output is a list, dictionary, or other data type\n- **If the count result is 0:** This may indicate an error OR genuinely no salt bridges in the region. Verify charged residue presence and recheck if needed\n\n**Salt Bridge Definition:** Salt bridges are electrostatic interactions between positively charged residues (Arg, Lys, His) and negatively charged residues (Asp, Glu) within a distance typically \u22644.0\u00c5 between charged atoms.\n\n**Technical Implementation:** Uses ChimeraX 'contacts' command with 'restrict' and 'distanceOnly' parameters to accurately detect charged residue interactions. The function automatically selects and separates positive and negative charged residues before analyzing their interactions.\n\n**Usage Examples:**\n- Find salt bridges around residue 150: target_atom_spec=':150', distance_range=4.0\n- Find salt bridges around residues 1-50: target_atom_spec=':1-50', distance_range=4.0\n- Find salt bridges around chain A: target_atom_spec='/A', distance_range=4.0\n- Analyze whole protein salt bridges: target_atom_spec='protein', distance_range=0 (distance_range=0 means no distance restriction)\n- Find salt bridges around charged residues only: target_atom_spec=':arg,lys,his,asp,glu', distance_range=5.0\n\n**Output Format:** Returns detailed information including residue pairs, atom names, and precise distances for all detected salt bridges.",
        "name": "find_salt_bridges_around_atoms",
        "optional_parameters": [
            {
                "default": 4.0,
                "description": "Search distance range in Angstroms around the target atoms (default: 4.0 for salt bridges)",
                "name": "distance_range",
                "type": "float"
            },
            {
                "default": None,
                "description": "Output file path to save salt bridge results (optional)",
                "name": "output_file",
                "type": "str"
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB or CIF file to analyze",
                "name": "pdb_file",
                "type": "str"
            },
            {
                "default": None,
                "description": "Target atom specification in ChimeraX format (e.g., ':150' for residue 150, ':150-160' for residue range)",
                "name": "target_atom_spec",
                "type": "str"
            }
        ]
    },
    {
        "description": "Find salt bridges around a specific residue within a specified distance. Simplified function for single residue salt bridge analysis. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **Before counting or processing results:** Examine whether the output is a list, dictionary, or other data type\n- **If the count result is 0:** This may indicate an error OR genuinely no salt bridges around this residue. Verify if the residue is charged and recheck if needed\n\n**Salt Bridge Analysis:** Detects electrostatic interactions between the target residue and oppositely charged residues within the specified distance. Uses advanced ChimeraX 'contacts' command with proper restriction parameters for accurate salt bridge identification.\n\n**Usage Examples:**\n- Find salt bridges around lysine 25: residue_num=25, distance=4.0\n- Find salt bridges around arginine 100 within 5\u00c5: residue_num=100, distance=5.0\n- Standard salt bridge analysis for residue 50: residue_num=50 (uses default distance=4.0)\n\n**Expected Results:** Returns information about charged residue pairs, specific atoms involved (e.g., LYS NZ \u2194 ASP OD2), and precise interaction distances.",
        "name": "find_residue_salt_bridges_in_range",
        "optional_parameters": [
            {
                "default": 4.0,
                "description": "Search distance in Angstroms around the residue (default: 4.0 for salt bridges)",
                "name": "distance",
                "type": "float"
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB or CIF file to analyze",
                "name": "pdb_file",
                "type": "str"
            },
            {
                "default": None,
                "description": "Residue number to analyze",
                "name": "residue_num",
                "type": "int"
            }
        ]
    },
    {
        "description": "Find salt bridges around a specific residue in a specific chain. Useful for multi-chain protein complexes where chain identification is important. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **Before counting or processing results:** Examine whether the output is a list, dictionary, or other data type\n- **If the count result is 0:** This may indicate an error OR genuinely no salt bridges around this chain/residue. Verify charged residue presence and recheck if needed\n\n**Multi-Chain Analysis:** Particularly useful for protein complexes, homodimers, and heterodimers where salt bridges may form both within and between chains. Uses precise ChimeraX 'contacts' command for accurate detection.\n\n**Usage Examples:**\n- Find salt bridges around lysine 25 in chain A: chain_id='A', residue_num=25, distance=4.0\n- Find salt bridges around arginine 100 in chain B: chain_id='B', residue_num=100, distance=4.0\n- Analyze charged residue interactions in specific chain: chain_id='C', residue_num=75\n\n**Chain-Specific Benefits:** Enables analysis of inter-chain salt bridges in protein complexes and helps identify stabilizing interactions at protein-protein interfaces.",
        "name": "find_chain_residue_salt_bridges",
        "optional_parameters": [
            {
                "default": 4.0,
                "description": "Search distance in Angstroms around the residue (default: 4.0 for salt bridges)",
                "name": "distance",
                "type": "float"
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB or CIF file to analyze",
                "name": "pdb_file",
                "type": "str"
            },
            {
                "default": None,
                "description": "Chain identifier (e.g., 'A', 'B', 'C')",
                "name": "chain_id",
                "type": "str"
            },
            {
                "default": None,
                "description": "Residue number within the specified chain",
                "name": "residue_num",
                "type": "int"
            }
        ]
    },
    {
        "description": "Analyze all salt bridges in a complete protein structure without distance restrictions. Provides a comprehensive salt bridge network analysis. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **Before counting or processing results:** Examine whether the output is a list, dictionary, or other data type\n- **If the count result is 0:** This may indicate an error OR genuinely no salt bridges in the protein. Verify charged residue presence and recheck if needed\n\n**Comprehensive Analysis:** Identifies ALL salt bridges in the entire protein structure using optimized ChimeraX 'contacts' command. Provides complete electrostatic interaction network for understanding protein stability and function.\n\n**Usage Examples:**\n- Complete protein salt bridge analysis: analyze_protein_salt_bridges(pdb_file='protein.pdb')\n- Save results to file: analyze_protein_salt_bridges(pdb_file='protein.pdb', output_file='all_saltbridges.txt')\n- This function is ideal for getting an overview of ALL salt bridges in the entire protein structure\n\n**Typical Output:** Returns comprehensive list of all charged residue pairs with distances, enabling analysis of protein stability, domain interactions, and electrostatic networks.",
        "name": "analyze_protein_salt_bridges",
        "optional_parameters": [
            {
                "default": None,
                "description": "Output file path to save complete salt bridge analysis (optional)",
                "name": "output_file",
                "type": "str"
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB or CIF file containing the protein structure",
                "name": "pdb_file",
                "type": "str"
            }
        ]
    },
    {
        "description": "Find salt bridges at the interface between two molecular entities (e.g., protein-protein, protein-ligand interfaces). Specialized for studying interface stability and interactions. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **Before counting or processing results:** Examine whether the output is a list, dictionary, or other data type\n- **If the count result is 0:** This may indicate an error OR genuinely no interface salt bridges. Verify interface specification and charged residue presence\n\n**Interface Analysis:** Uses advanced ChimeraX 'contacts' command with proper restriction to specifically detect salt bridges that cross the interface between different molecular entities. Critical for understanding binding affinity and complex stability.\n\n**Usage Examples:**\n- Find salt bridges between chain A and B: interface_spec1='/A', interface_spec2='/B', distance=4.0\n- Analyze protein-ligand interface: interface_spec1='protein', interface_spec2='ligand', distance=4.0\n- Study domain-domain interactions: interface_spec1=':1-100', interface_spec2=':200-300', distance=5.0\n- Save interface analysis: find_interface_salt_bridges('complex.pdb', '/A', '/B', output_file='interface.txt')\n\n**Interface Specificity:** Only reports salt bridges that span between the two specified entities, filtering out intra-entity interactions to focus on interface-stabilizing contacts.",
        "name": "find_interface_salt_bridges",
        "optional_parameters": [
            {
                "default": 4.0,
                "description": "Search distance in Angstroms at the interface (default: 4.0 for salt bridges)",
                "name": "distance",
                "type": "float"
            },
            {
                "default": None,
                "description": "Output file path to save interface salt bridge results (optional)",
                "name": "output_file",
                "type": "str"
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB or CIF file containing the molecular complex",
                "name": "pdb_file",
                "type": "str"
            },
            {
                "default": None,
                "description": "First interface specification in ChimeraX format (e.g., '/A' for chain A)",
                "name": "interface_spec1",
                "type": "str"
            },
            {
                "default": None,
                "description": "Second interface specification in ChimeraX format (e.g., '/B' for chain B)",
                "name": "interface_spec2",
                "type": "str"
            }
        ]
    },
    {
        "description": "Find interactions between charged residues (Arg, Lys, His, Asp, Glu) including salt bridges and electrostatic interactions. Considers pH-dependent protonation states. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **Before counting or processing results:** Examine whether the output is a list, dictionary, or other data type\n- **If the count result is 0:** This may indicate an error OR genuinely no charged interactions. Verify charged residue presence and recheck if needed\n\n**pH-Dependent Analysis:** Accounts for histidine protonation states based on pH value. Uses sophisticated ChimeraX 'contacts' command with 'restrict' and 'distanceOnly' parameters for precise electrostatic interaction detection.\n\n**Protonation Logic:**\n- pH < 6.0: Histidine treated as positively charged (:his)\n- pH \u2265 6.0: Histidine variants considered (:hie,hid,hip)\n\n**Usage Examples:**\n- Standard charged residue analysis: find_charged_residue_interactions(pdb_file='protein.pdb')\n- Analysis at physiological pH: find_charged_residue_interactions(pdb_file='protein.pdb', ph_value=7.4)\n- Analysis at acidic conditions: find_charged_residue_interactions(pdb_file='protein.pdb', ph_value=5.0, distance=5.0)\n- Save comprehensive analysis: find_charged_residue_interactions('protein.pdb', output_file='charged_interactions.txt')\n\n**Advanced Features:** Provides comprehensive electrostatic analysis considering environmental pH, essential for understanding protein behavior in different physiological conditions.",
        "name": "find_charged_residue_interactions",
        "optional_parameters": [
            {
                "default": 4.0,
                "description": "Search distance in Angstroms for electrostatic interactions",
                "name": "distance",
                "type": "float"
            },
            {
                "default": 7.0,
                "description": "pH value for determining protonation states, especially for histidine residues",
                "name": "ph_value",
                "type": "float"
            },
            {
                "default": None,
                "description": "Output file path to save charged residue interaction results (optional)",
                "name": "output_file",
                "type": "str"
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB or CIF file to analyze",
                "name": "pdb_file",
                "type": "str"
            }
        ]
    },
    {
        "description": "Validate parameters for complex structure prediction including proteins, DNA sequences, ligands, and prediction settings. This utility function ensures all input parameters are properly formatted before structure prediction.\n\n**Parameter Validation Features:**\n- **Protein Count:** Validates positive integer for protein copies\n- **DNA Sequences:** Validates nucleotide sequences (ATCG only) and corresponding counts\n- **Ligands:** Validates SMILES strings and corresponding counts\n- **Model Names:** Validates against supported Protenix model names\n- **Seeds:** Validates comma-separated integer format\n\n**Supported Protenix Models:**\n- protenix_base_default_v0.5.0 (default, full MSA capability)\n- protenix_mini_esm_v0.5.0 (ESM-based, no MSA required)\n- protenix_mini_default_v0.5.0 (mini version with MSA)\n- protenix_tiny_default_v0.5.0 (tiny version)\n\n**Validation Rules:**\n- DNA sequences must contain only A, T, C, G characters (case insensitive)\n- Ligand SMILES strings must be non-empty\n- All counts must be positive integers\n- If counts arrays are not provided, defaults to 1 for each sequence/ligand\n- Seeds must be comma-separated non-negative integers\n\n**Usage Examples:**\n```python\n# Validate simple protein prediction\nprotein_count, dna_data, ligand_data, model, seeds = validate_complex_parameters(\n    protein_count=2,\n    model_name='protenix_base_default_v0.5.0',\n    seeds='42,43'\n)\n\n# Validate complex with DNA and ligands\nvalidated_params = validate_complex_parameters(\n    protein_count=1,\n    dna_sequences=['ATCGATCG', 'GCTAGCTA'],\n    dna_counts=[1, 2],\n    ligands=['CC(C)O', 'SMILES_STRING'],\n    ligand_counts=[1, 1]\n)\n```\n\n**Return Value:** Tuple containing (protein_count, validated_dna_data, validated_ligand_data, model_name, seeds)\n\n**Error Handling:** Raises ValueError with specific error messages for invalid parameters",
        "name": "validate_complex_parameters",
        "optional_parameters": [
            {
                "default": 1,
                "description": "Number of protein copies (must be >= 1)",
                "name": "protein_count",
                "type": "int"
            },
            {
                "default": None,
                "description": "List of DNA sequences as strings containing only A, T, C, G characters",
                "name": "dna_sequences",
                "type": "list"
            },
            {
                "default": None,
                "description": "List of integers specifying count for each DNA sequence (must match dna_sequences length)",
                "name": "dna_counts",
                "type": "list"
            },
            {
                "default": None,
                "description": "List of ligand SMILES strings",
                "name": "ligands",
                "type": "list"
            },
            {
                "default": None,
                "description": "List of integers specifying count for each ligand (must match ligands length)",
                "name": "ligand_counts",
                "type": "list"
            },
            {
                "default": "protenix_base_default_v0.5.0",
                "description": "Protenix model name for structure prediction",
                "name": "model_name",
                "type": "str"
            },
            {
                "default": "42",
                "description": "Random seeds as comma-separated string (e.g., '42' or '42,43,44')",
                "name": "seeds",
                "type": "str"
            }
        ],
        "required_parameters": []
    },
    {
        "description": "Convert TTS JSON format to Protenix-compatible JSON format with comprehensive support for protein-DNA-ligand complexes. This function bridges the gap between TTS sequence generation and Protenix structure prediction.\n\n**TTS JSON Input Format:**\nExpects JSON array with structure:\n```json\n[\n  {\n    \"sample_id\": \"seq_1\",\n    \"sequences\": [\n      {\n        \"seq\": \"MGSSHHHHHHSSGLVPRGSHMSGKIQHKAVVPAPSRIPLTI...\",\n        \"score\": {\"progen_nll\": 2.348, \"TM_score\": 0.865},\n        \"weighted_score\": 300.0\n      }\n    ]\n  }\n]\n```\n\n**Complex Structure Support:**\n- **Proteins:** Extracted from TTS JSON with configurable copy numbers\n- **DNA Sequences:** User-provided nucleotide sequences with individual counts\n- **Ligands:** User-provided SMILES strings with individual counts\n- **MSA Integration:** Automatic MSA path references when precomputed MSA directory provided\n\n**Generated Protenix JSON Format:**\n```json\n[\n  {\n    \"name\": \"complex_prediction_0\",\n    \"covalent_bonds\": [],\n    \"sequences\": [\n      {\n        \"proteinChain\": {\n          \"sequence\": \"PROTEIN_SEQUENCE\",\n          \"count\": 2,\n          \"modifications\": [],\n          \"msa\": {\"precomputed_msa_dir\": \"/path/to/msa/0\"}\n        }\n      },\n      {\n        \"dnaSequence\": {\n          \"sequence\": \"ATCGATCG\",\n          \"count\": 1,\n          \"modifications\": []\n        }\n      },\n      {\n        \"ligand\": {\n          \"ligand\": \"CC(C)O\",\n          \"count\": 1\n        }\n      }\n    ]\n  }\n]\n```\n\n**Usage Examples:**\n```python\n# Simple protein prediction\nconvert_tts_to_protenix_format('input.json', 'output.json')\n\n# Protein-DNA complex\nconvert_tts_to_protenix_format(\n    'input.json', 'output.json',\n    protein_count=1,\n    dna_sequences=['ATCGATCG'],\n    dna_counts=[1]\n)\n\n# Multi-component complex with MSA\nconvert_tts_to_protenix_format(\n    'input.json', 'output.json',\n    msa_dir='./msa_output',\n    protein_count=2,\n    dna_sequences=['GCTAGCTA'],\n    dna_counts=[1],\n    ligands=['SMILES1'],\n    ligand_counts=[1]\n)\n```\n\n**Error Handling:** Raises ValueError for invalid parameters and FileNotFoundError for missing TTS JSON file.",
        "name": "convert_tts_to_protenix_format",
        "optional_parameters": [
            {
                "default": None,
                "description": "Directory containing precomputed MSA files (optional, enables MSA-based prediction)",
                "name": "msa_dir",
                "type": "str"
            },
            {
                "default": 1,
                "description": "Number of protein copies in the complex",
                "name": "protein_count",
                "type": "int"
            },
            {
                "default": None,
                "description": "List of DNA sequences as strings (ATCG nucleotides)",
                "name": "dna_sequences",
                "type": "list"
            },
            {
                "default": None,
                "description": "List of integers specifying count for each DNA sequence",
                "name": "dna_counts",
                "type": "list"
            },
            {
                "default": None,
                "description": "List of ligand SMILES strings",
                "name": "ligands",
                "type": "list"
            },
            {
                "default": None,
                "description": "List of integers specifying count for each ligand",
                "name": "ligand_counts",
                "type": "list"
            },
            {
                "default": "complex_prediction",
                "description": "Base name for the prediction job (will be suffixed with sequence counter)",
                "name": "job_name",
                "type": "str"
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to TTS JSON file containing protein sequences and scores",
                "name": "tts_json_file",
                "type": "str"
            },
            {
                "default": None,
                "description": "Path for output Protenix-compatible JSON file",
                "name": "output_json_file",
                "type": "str"
            }
        ]
    },
    {
        "description": "Create FASTA file from TTS JSON for MSA generation. This is a utility function to prepare for MSA-based structure prediction.",
        "name": "create_fasta_from_tts",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to TTS JSON file",
                "name": "tts_json_file",
                "type": "str"
            },
            {
                "default": None,
                "description": "Path for output FASTA file",
                "name": "output_fasta_file",
                "type": "str"
            }
        ]
    },
    {
        "description": "Build Protenix predict command based on MSA preferences and model settings. This utility function constructs the appropriate command line arguments for different prediction modes.\n\n**Prediction Modes:**\n1. **Precomputed MSA Mode:** Uses MSA files referenced in JSON, fastest prediction with evolutionary info\n2. **Real-time MSA Search:** Protenix searches MSA databases automatically during prediction\n3. **ESM Mode:** No MSA used, relies on ESM embeddings, fastest but potentially less accurate\n\n**Command Construction Logic:**\n- **ESM Mode (use_msa=False):** Forces protenix_mini_esm_v0.5.0 model with --use_msa False\n- **Precomputed MSA:** Uses specified model with JSON containing MSA paths\n- **Real-time MSA:** Uses --use_msa True flag for automatic MSA search\n\n**Usage Examples:**\n```python\n# Precomputed MSA mode\ncmd = build_protenix_predict_command(\n    'input.json', './output',\n    use_msa=True, precompute_msa=True,\n    model_name='protenix_base_default_v0.5.0'\n)\n\n# Real-time MSA search\ncmd = build_protenix_predict_command(\n    'input.json', './output',\n    use_msa=True, precompute_msa=False\n)\n\n# ESM mode (no MSA)\ncmd = build_protenix_predict_command(\n    'input.json', './output',\n    use_msa=False\n)\n```\n\n**Generated Commands Examples:**\n- Precomputed: ['protenix', 'predict', '--input', 'input.json', '--out_dir', './output', '--seeds', '42', '--model_name', 'protenix_base_default_v0.5.0']\n- Real-time: ['protenix', 'predict', '--input', 'input.json', '--out_dir', './output', '--seeds', '42', '--use_msa', 'True']\n- ESM: ['protenix', 'predict', '--input', 'input.json', '--out_dir', './output', '--seeds', '42', '--model_name', 'protenix_mini_esm_v0.5.0', '--use_msa', 'False']",
        "name": "build_protenix_predict_command",
        "optional_parameters": [
            {
                "default": True,
                "description": "Whether to use MSA features (False = ESM mode)",
                "name": "use_msa",
                "type": "bool"
            },
            {
                "default": True,
                "description": "Whether MSA is precomputed (True = use JSON MSA paths, False = real-time search)",
                "name": "precompute_msa",
                "type": "bool"
            },
            {
                "default": "protenix_base_default_v0.5.0",
                "description": "Protenix model name (ignored in ESM mode)",
                "name": "model_name",
                "type": "str"
            },
            {
                "default": "42",
                "description": "Random seeds as comma-separated string",
                "name": "seeds",
                "type": "str"
            },
            {
                "default": "protenix",
                "description": "Path to protenix executable",
                "name": "protenix_executable",
                "type": "str"
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to Protenix-compatible JSON input file",
                "name": "json_file",
                "type": "str"
            },
            {
                "default": None,
                "description": "Output directory for prediction results",
                "name": "output_dir",
                "type": "str"
            }
        ]
    },
    {
        "description": "Perform comprehensive protein-DNA-ligand complex structure prediction using Protenix with advanced MSA and model options. This is the main function that orchestrates the complete prediction workflow from TTS sequences to 3D structures.\n\n**14 COMPLEX STRUCTURE PREDICTION CAPABILITIES:**\n- **Multi-Component Complexes:** Supports any combination of proteins, DNA, and ligands\n- **Flexible Stoichiometry:** Individual copy numbers for each component\n- **Three Prediction Modes:** Precomputed MSA, Real-time MSA search, ESM-only\n- **Advanced Model Selection:** Support for all Protenix model variants\n\n**14 PREDICTION MODES:**\n\n**1. Precomputed MSA Mode (Recommended for Accuracy):**\n- Generates MSA files using 'protenix msa' command\n- Uses evolutionary information for better prediction\n- Slower initial MSA generation, faster prediction\n- Best accuracy for most proteins\n\n**2. Real-time MSA Search Mode:**\n- Protenix searches MSA databases during prediction\n- No pre-computation required\n- Moderate speed and accuracy\n- Good for rapid prototyping\n\n**3. ESM Mode (Fastest):**\n- Uses ESM language model embeddings only\n- No MSA generation or search\n- Fastest prediction mode\n- Good for large-scale screening\n\n**14 COMPLEX COMPOSITION EXAMPLES:**\n\n```python\n# Protein-DNA complex (transcription factor)\nresult = structure_prediction(\n    'tf_sequences.json',\n    protein_count=1,\n    dna_sequences=['GATAGTGACAAACTTGACAACTCATCACTTCCTAGGTATAATGCTAGCTACTAGAG'],\n    dna_counts=[1],\n    job_name='transcription_factor_complex'\n)\n\n# Protein-ligand complex (enzyme inhibitor)\nresult = structure_prediction(\n    'enzyme_sequences.json',\n    protein_count=2,  # homodimer\n    ligands=['CC(C)(COP([O-])(=O)OP([O-])(=O)OC[C@H]1O[C@H]([C@H](O)[C@@H]1OP([O-])([O-])=O)n1cnc2c(N)ncnc12)[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)CCC([O-])=O'],\n    ligand_counts=[2],  # one ligand per protein chain\n    use_msa=True,\n    precompute_msa=True,\n    model_name='protenix_base_default_v0.5.0'\n)\n```\n\n**Return Value:** Dictionary containing prediction results, including paths to generated files and stdout/stderr logs.\n\n**Error Handling:** Raises exceptions for invalid parameters and returns detailed error dictionary for failed subprocess calls.",
        "name": "structure_prediction",
        "optional_parameters": [
            {
                "default": "./output",
                "description": "Output directory for all generated files including FASTA, JSON, MSA files, and prediction results",
                "name": "output_dir",
                "type": "str"
            },
            {
                "default": 1,
                "description": "Number of protein copies in the complex (stoichiometry)",
                "name": "protein_count",
                "type": "int"
            },
            {
                "default": None,
                "description": "List of DNA sequences as strings containing ATCG nucleotides (e.g., ['ATCGATCG', 'GCTAGCTA'])",
                "name": "dna_sequences",
                "type": "list"
            },
            {
                "default": None,
                "description": "List of integers specifying copy number for each DNA sequence (must match dna_sequences length)",
                "name": "dna_counts",
                "type": "list"
            },
            {
                "default": None,
                "description": "List of ligand SMILES strings (e.g., ['CC(C)O', 'SMILES_STRING'])",
                "name": "ligands",
                "type": "list"
            },
            {
                "default": None,
                "description": "List of integers specifying copy number for each ligand (must match ligands length)",
                "name": "ligand_counts",
                "type": "list"
            },
            {
                "default": True,
                "description": "Whether to use MSA features (True=MSA modes, False=ESM mode for fastest prediction)",
                "name": "use_msa",
                "type": "bool"
            },
            {
                "default": True,
                "description": "Whether to precompute MSA using 'protenix msa' (True=precompute, False=real-time search)",
                "name": "precompute_msa",
                "type": "bool"
            },
            {
                "default": "protenix_base_default_v0.5.0",
                "description": "Protenix model name (protenix_base_default_v0.5.0, protenix_mini_default_v0.5.0, etc.). Automatically set to protenix_mini_esm_v0.5.0 when use_msa=False",
                "name": "model_name",
                "type": "str"
            },
            {
                "default": "42",
                "description": "Random seeds for prediction sampling, comma-separated for multiple predictions (e.g., '42,43,44')",
                "name": "seeds",
                "type": "str"
            },
            {
                "default": "complex_prediction",
                "description": "Base name for the prediction job (used in output filenames and directories)",
                "name": "job_name",
                "type": "str"
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to TTS JSON file. The essential format is a list of objects, each containing a 'sequences' key, which in turn contains a list of objects with a 'seq' key (e.g., [{'sequences': [{'seq': '...'}]}]).",
                "name": "tts_json_file",
                "type": "str"
            }
        ]
    }
]