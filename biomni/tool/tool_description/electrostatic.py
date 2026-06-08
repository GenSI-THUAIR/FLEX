description = [
    {
        "description": "Analyze electrostatic potential within a specific distance range around target residue using ChimeraX. This is the core function for electrostatic surface analysis that accepts any ChimeraX residue specification. **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **Before processing results:** Examine the Coulombic potential values (minimum, mean, maximum) and center of mass coordinates\n- **If the analysis fails:** Check the target residue specification format and ensure the PDB file contains the specified residue\n\n**Electrostatic Analysis Workflow:**\n1. **Structure Preparation:** Loads PDB structure and adds hydrogen atoms using ChimeraX 'addh' command\n2. **Charge Assignment:** Computes atomic charges using Gasteiger method via 'addcharge' command\n3. **Surface Generation:** Creates molecular surface and calculates Coulombic electrostatic potential\n4. **Target Analysis:** Computes electrostatic potential specifically for the target residue\n5. **Environment Selection:** Selects protein atoms within specified distance range around target\n6. **Regional Analysis:** Calculates electrostatic potential for the selected environmental region\n\n**Technical Implementation:** Uses ChimeraX 'coulombic' command for accurate electrostatic potential calculation on molecular surfaces. The analysis provides quantitative electrostatic values including surface potential distribution statistics.\n\n**Usage Examples:**\n- Analyze residue 3 in chain A: target_residue='/A:3', distance_range=5.0\n- Analyze multiple residues: target_residue='/A:10-15', distance_range=6.0\n- Analyze specific chain region: target_residue='/B:25', distance_range=4.0\n- Save electrostatic data: analyze_electrostatic('protein.pdb', '/A:3', output_file='elec_data.txt')\n- Standard 5Å analysis: analyze_electrostatic('protein.pdb', '/A:3') (uses default distance_range=5.0)\n\n**Output Information:**\n- **Coulombic Potential Values:** Minimum, mean, and maximum electrostatic potential on molecular surface\n- **Center of Mass:** 3D coordinates of the target residue's geometric center\n- **Environmental Selection:** Number of atoms selected within the specified distance range\n- **Surface Coloring:** Information about electrostatic potential surface visualization\n- **Charge Warnings:** Details about non-integral total charges and missing atoms\n\n**Electrostatic Interpretation:**\n- **Negative Values:** Regions of negative electrostatic potential (electron-rich areas)\n- **Positive Values:** Regions of positive electrostatic potential (electron-poor areas)\n- **Magnitude:** Larger absolute values indicate stronger electrostatic fields\n- **Mean Value:** Overall electrostatic character of the analyzed region",
        "name": "analyze_electrostatic",
        "optional_parameters": [
            {
                "default": 5.0,
                "description": "Search distance range in Angstroms around the target residue for environmental analysis",
                "name": "distance_range",
                "type": "float",
            },
            {
                "default": None,
                "description": "Output file path to save electrostatic potential data (optional)",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the PDB file containing the protein structure to analyze",
                "name": "pdb_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Target residue specification in ChimeraX format (e.g., '/A:3' for chain A residue 3, '/B:25-30' for residue range)",
                "name": "target_residue",
                "type": "str",
            },
        ],
    },
]
