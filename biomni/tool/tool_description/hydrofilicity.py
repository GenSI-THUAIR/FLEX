description = [
    {
        "description": "Get residues within a specific distance range around a target residue using ChimeraX. This is the core function for identifying neighboring residues in protein structures. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **Before counting or processing results:** Examine whether the output is a list, dictionary, or other data type\n- **If the count result is 0:** This indicates an error - residues should exist around most target positions. Recheck the function call and re-run the analysis\n\n**Usage Examples:**\n- Find residues around position 3 in chain A: target_residue='/A:3', distance_range=5.0\n- Find residues around position 150: target_residue=':150', distance_range=8.0\n- Find residues in larger neighborhood: target_residue='/B:25', distance_range=10.0\n- Standard protein analysis: target_residue=':100', distance_range=5.0 (default distance)",
        "name": "get_residues",
        "optional_parameters": [
            {
                "default": 5.0,
                "description": "Search distance range in Angstroms around the target residue",
                "name": "distance_range",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB or CIF file to analyze",
                "name": "pdb_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Target residue specification in ChimeraX format (e.g., '/A:3' for residue 3 in chain A, ':150' for residue 150)",
                "name": "target_residue",
                "type": "str",
            },
        ],
    },
    {
        "description": "Analyze hydrophilicity of residues around a target residue using the Kyte-Doolittle hydropathy scale. This function provides comprehensive hydrophobic/hydrophilic environment analysis. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **Before counting or processing results:** Examine whether the output is a dictionary with specific keys\n- **If the count result is 0:** This may indicate an error OR genuinely no residues in the specified range. Verify target residue specification and recheck if needed\n\n**Kyte-Doolittle Scale Interpretation:**\n- **Positive values (hydrophobic):** ALA(1.8), CYS(2.5), ILE(4.5), LEU(3.8), MET(1.9), PHE(2.8), VAL(4.2)\n- **Negative values (hydrophilic):** ARG(-4.5), ASN(-3.5), ASP(-3.5), GLN(-3.5), GLU(-3.5), HIS(-3.2), LYS(-3.9)\n- **Near neutral:** GLY(-0.4), PRO(-1.6), SER(-0.8), THR(-0.7), TRP(-0.9), TYR(-1.3)\n\n**Analysis Results:**\n- **target_residue_kd:** Hydropathy score of the target residue itself\n- **residues_count:** Number of neighboring residues analyzed\n- **total_kd_score:** Sum of all neighboring residues' KD scores\n- **average_kd_score:** Average hydropathy of the local environment\n\n**Usage Examples:**\n- Analyze hydrophilic environment around residue 3: target_residue='/A:3', distance_range=8.0\n- Study local hydropathy around active site: target_residue=':150', distance_range=10.0\n- Quick hydrophilicity check: target_residue='/B:25' (uses default distance=8.0)\n- Membrane protein analysis: target_residue=':200', distance_range=12.0",
        "name": "analyze_hydrofilicity",
        "optional_parameters": [
            {
                "default": 8.0,
                "description": "Search distance range in Angstroms around the target residue (default: 8.0 for hydropathy analysis)",
                "name": "distance_range",
                "type": "float",
            },
            {
                "default": None,
                "description": "Output file path (optional, currently not implemented)",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB or CIF file to analyze",
                "name": "pdb_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Target residue specification in ChimeraX format (e.g., '/A:3' for residue 3 in chain A)",
                "name": "target_residue",
                "type": "str",
            },
        ],
    },
    {
        "description": "Calculate solvent accessible surface area (SASA) for a target residue using ChimeraX with standard probe radius. This function provides raw SASA values in Ų. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **The output is a string containing ChimeraX results** - you need to parse it to extract numerical SASA values\n- **If no SASA information is found:** This indicates an error - verify target residue specification and recheck\n\n**SASA Calculation Details:**\n- **Probe radius:** 1.4 Å (standard water molecule radius)\n- **Algorithm:** ChimeraX uses advanced surface calculation methods\n- **Units:** Square Angstroms (Ų)\n- **Accuracy:** High precision suitable for quantitative analysis\n\n**Technical Implementation:** Uses ChimeraX 'measure sasa' command with precise probe radius specification. The function automatically handles temporary script creation and cleanup.\n\n**Usage Examples:**\n- Calculate SASA for residue 3: target_residue='/A:3'\n- Analyze surface exposure: target_residue=':150'\n- Save SASA results to file: target_residue='/B:25', output_file='sasa_results.txt'\n- Membrane protein surface analysis: target_residue=':200'\n\n**Output Parsing:** The returned string contains residue information and SASA values that need to be extracted using regular expressions or string parsing methods.",
        "name": "calculate_sasa",
        "optional_parameters": [
            {
                "default": None,
                "description": "Output file path for saving SASA calculation results (optional)",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB or CIF file to analyze",
                "name": "pdb_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Target residue specification in ChimeraX format (e.g., '/A:3' for residue 3 in chain A)",
                "name": "target_residue",
                "type": "str",
            },
        ],
    },
    {
        "description": "Analyze if a residue is on the protein surface by calculating normalized SASA (nSASA). This function provides quantitative surface exposure analysis with values between 0.0-1.0. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **The output is a float value between 0.0-1.0** representing surface exposure\n- **If ValueError is raised:** This indicates missing residue type or SASA information - verify target residue specification\n\n**Normalized SASA (nSASA) Interpretation:**\n- **0.0-0.2:** Buried residue (core/interior)\n- **0.2-0.5:** Partially exposed (intermediate)\n- **0.5-0.8:** Surface exposed\n- **0.8-1.0:** Highly exposed (fully accessible)\n- **>1.0:** Capped at 1.0 (may indicate unusual conformations)\n\n**Maximum SASA Reference Values (Ų):**\n- Small residues: ALA(129), GLY(104), SER(155)\n- Medium residues: CYS(167), THR(172), VAL(174)\n- Large residues: ARG(274), TRP(285), PHE(240)\n- Charged residues: LYS(236), ASP(193), GLU(223)\n\n**Technical Implementation:** \n1. Calculates actual SASA using ChimeraX with 1.4Å probe radius\n2. Extracts residue type from ChimeraX output\n3. Normalizes against maximum theoretical SASA for that residue type\n4. Caps result at 1.0 to handle edge cases\n\n**Usage Examples:**\n- Check if residue is surface-exposed: analyze_if_on_surface(pdb_file='protein.pdb', target_residue='/A:3')\n- Analyze active site accessibility: analyze_if_on_surface(pdb_file='enzyme.pdb', target_residue=':150')\n- Membrane protein surface analysis: analyze_if_on_surface(pdb_file='membrane.pdb', target_residue='/B:25')\n- Save detailed SASA data: analyze_if_on_surface('protein.pdb', ':100', output_file='surface_analysis.txt')\n\n**Applications:** Essential for understanding protein folding, binding site accessibility, mutation effects, and protein-protein interactions.",
        "name": "analyze_if_on_surface",
        "optional_parameters": [
            {
                "default": None,
                "description": "Output file path for saving detailed SASA calculation results (optional)",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB or CIF file to analyze",
                "name": "pdb_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Target residue specification in ChimeraX format (e.g., '/A:3' for residue 3 in chain A)",
                "name": "target_residue",
                "type": "str",
            },
        ],
    },
]
