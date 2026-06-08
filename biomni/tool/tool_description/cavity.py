description = [
    {
        "description": "Analyze cavity volume around target residue using ChimeraX molecular surface calculation. This function creates a molecular surface around selected atoms and measures the enclosed cavity volume using Solvent Excluded Surface (SES). **IMPORTANT: This function handles all ChimeraX scripting internally - simply import and call it directly. No need to create additional scripts.**\n\n**⚠️ CRITICAL OUTPUT HANDLING:** \n- **ALWAYS print out the full output first** to understand the exact data structure and format\n- **Before processing results:** Examine the output for volume measurement information in cubic Angstroms (Å³)\n- **If no volume is reported:** This indicates an error - check the target residue specification and distance range\n\n**Cavity Analysis:** Uses ChimeraX molecular surface calculation with Solvent Excluded Surface (SES) to determine cavity volumes around target residues. The probe radius simulates solvent accessibility, typically using water molecule radius (1.4Å). Automatically adds hydrogens for accurate surface calculation.\n\n**Technical Implementation:** Selects atoms within specified distance using 'zone' selection, creates molecular surface with specified probe radius, and measures enclosed volume using ChimeraX 'measure volume' command. Surface model ID is typically #1.2 in ChimeraX model hierarchy.\n\n**Usage Examples:**\n- Analyze cavity around residue 3 in chain A: target_residue='/A:3', distance_range=5.0\n- Analyze cavity around residue range: target_residue=':100-105', distance_range=6.0\n- Custom probe radius for small molecules: target_residue=':150', probe_radius=1.2\n- Save results to file: analyze_cavity('protein.pdb', '/A:25', output_file='cavity_volume.txt')\n- Larger search radius for deep pockets: target_residue=':200', distance_range=8.0\n\n**Expected Results:** Returns volume measurements in cubic Angstroms (Å³). Typical cavity volumes for binding sites range from 100-1000 Å³. Output includes structure loading information, hydrogen addition summary, atom selection details, surface creation confirmation, and precise volume measurement.",
        "name": "analyze_cavity",
        "optional_parameters": [
            {
                "default": 5.0,
                "description": "Search distance range in Angstroms around the target residue. Atoms within this distance will be included in surface calculation",
                "name": "distance_range",
                "type": "float",
            },
            {
                "default": None,
                "description": "Output file path to save volume measurement results (optional)",
                "name": "output_file",
                "type": "str",
            },
            {
                "default": 1.4,
                "description": "Probe radius in Angstroms for molecular surface calculation. Typically 1.4 Å (water molecule radius) for solvent-accessible surfaces",
                "name": "probe_radius",
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
                "description": "Target residue specification in ChimeraX format (e.g., '/A:3' for chain A residue 3, ':150' for residue 150, ':100-105' for residue range)",
                "name": "target_residue",
                "type": "str",
            },
        ],
    },
]
