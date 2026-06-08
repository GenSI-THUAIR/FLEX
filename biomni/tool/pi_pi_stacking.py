import os
import subprocess

def _run_chimerax_script(script_file):
    """
    Run ChimeraX script using nogui mode.
    """
    result = subprocess.run([
        'chimerax-daily', '--nogui', '--script', script_file
    ], capture_output=True, text=True)
    return result

def find_pi_pi_stacking_around_atoms(pdb_file: str, target_atom_spec: str, distance_range: float = 8.0, 
                                    angle_cutoff: float = 30.0, output_file: str = None):
    """
    Find π-π stacking interactions within a specific distance range around target atoms using ChimeraX
    
    Args:
        pdb_file: Path to the PDB file
        target_atom_spec: Target atom specification (e.g., ":150" or ":150-160")
        distance_range: Search distance range in Angstroms (default: 8.0 for π-π stacking)
        angle_cutoff: Maximum angle deviation from parallel/perpendicular (default: 30.0 degrees)
        output_file: Output file path (optional)
    
    Returns:
        π-π stacking information as string
    """
    
    # Build ChimeraX commands for π-π stacking detection
    commands = [
        f"open {pdb_file}",
    ]
    
    # Handle different selection strategies based on target_atom_spec and distance_range
    if target_atom_spec.lower() == "protein" and distance_range == 0:
        # For whole protein analysis without distance restriction
        commands.append("select protein")
    elif distance_range == 0:
        # For specific selection without distance restriction
        commands.append(f"select {target_atom_spec}")
    else:
        # For selections with distance restriction
        commands.extend([
            f"select {target_atom_spec}",
            f"select zone sel {distance_range}"
        ])
    
    # Select aromatic residues for π-π stacking analysis
    commands.extend([
        "select sel & :phe,tyr,trp,his",  # Aromatic residues
        f"contacts sel restrict cross distance {distance_range} log true",
        f"# π-π stacking interactions between aromatic residues",
        f"# Distance cutoff: {distance_range} Å, Angle cutoff: {angle_cutoff}°",
    ])
    
    if output_file:
        commands.append(f"contacts sel saveFile {output_file}")
    
    # Create temporary script file
    script_file = "temp_pi_stacking.cxc"
    with open(script_file, 'w') as f:
        for cmd in commands:
            f.write(cmd + '\n')
        f.write('exit\n')
    
    try:
        # Run ChimeraX script
        result = _run_chimerax_script(script_file)
        
        return result.stdout
        
    finally:
        # Clean up temporary file
        if os.path.exists(script_file):
            os.remove(script_file)

def find_residue_pi_stacking_in_range(pdb_file: str, residue_num: int, distance: float = 6.0) -> str:
    """
    Find π-π stacking interactions around a specific aromatic residue within a specified distance
    
    Args:
        pdb_file: Path to the PDB file
        residue_num: Residue number
        distance: Search distance in Angstroms (default: 6.0 for π-π stacking)
    
    Returns:
        π-π stacking information as string
    """
    
    atom_spec = f":{residue_num}"
    return find_pi_pi_stacking_around_atoms(pdb_file, atom_spec, distance)

def find_chain_residue_pi_stacking(pdb_file: str, chain_id: str, residue_num: int, distance: float = 6.0) -> str:
    """
    Find π-π stacking interactions around a specific aromatic residue in a specific chain
    
    Args:
        pdb_file: Path to the PDB file
        chain_id: Chain identifier
        residue_num: Residue number
        distance: Search distance in Angstroms (default: 6.0 for π-π stacking)
    
    Returns:
        π-π stacking information as string
    """
    atom_spec = f"/{chain_id}:{residue_num}"
    return find_pi_pi_stacking_around_atoms(pdb_file, atom_spec, distance)

def analyze_protein_pi_stacking(pdb_file: str, output_file: str = None) -> str:
    """
    Analyze all π-π stacking interactions in a protein structure
    
    Args:
        pdb_file: Path to the PDB file
        output_file: Output file path (optional)
    
    Returns:
        Complete π-π stacking analysis as string
    """
    return find_pi_pi_stacking_around_atoms(pdb_file, "protein", 0, output_file=output_file)

def find_interface_pi_stacking(pdb_file: str, interface_spec1: str, interface_spec2: str, 
                              distance: float = 6.0, output_file: str = None) -> str:
    """
    Find π-π stacking interactions at the interface between two molecular entities
    
    Args:
        pdb_file: Path to the PDB file
        interface_spec1: First interface specification (e.g., "/A" for chain A)
        interface_spec2: Second interface specification (e.g., "/B" for chain B)
        distance: Search distance in Angstroms (default: 6.0 for π-π stacking)
        output_file: Output file path (optional)
    
    Returns:
        Interface π-π stacking information as string
    """
    
    # Build ChimeraX commands for interface π-π stacking detection
    commands = [
        f"open {pdb_file}",
        f"select ({interface_spec1} & :phe,tyr,trp,his) | ({interface_spec2} & :phe,tyr,trp,his)",
        f"contacts sel restrict cross distance {distance} log true",
        f"# Interface π-π stacking between {interface_spec1} and {interface_spec2}",
    ]
    
    if output_file:
        commands.append(f"contacts sel saveFile {output_file}")
    
    # Create temporary script file
    script_file = "temp_interface_pi_stacking.cxc"
    with open(script_file, 'w') as f:
        for cmd in commands:
            f.write(cmd + '\n')
        f.write('exit\n')
    
    try:
        # Run ChimeraX script
        result = _run_chimerax_script(script_file)
        
        return result.stdout
        
    finally:
        # Clean up temporary file
        if os.path.exists(script_file):
            os.remove(script_file)

def find_ligand_protein_pi_stacking(pdb_file: str, ligand_spec: str = "ligand", distance: float = 6.0, 
                                   output_file: str = None) -> str:
    """
    Find π-π stacking interactions between ligand and protein aromatic residues
    
    Args:
        pdb_file: Path to the PDB file
        ligand_spec: Ligand specification (default: "ligand")
        distance: Search distance in Angstroms (default: 6.0 for π-π stacking)
        output_file: Output file path (optional)
    
    Returns:
        Ligand-protein π-π stacking information as string
    """
    
    # Build ChimeraX commands for ligand-protein π-π stacking
    commands = [
        f"open {pdb_file}",
        f"select ({ligand_spec}) | (protein & :phe,tyr,trp,his)",
        f"contacts sel restrict cross distance {distance} log true",
        f"# π-π stacking between ligand and protein aromatic residues",
    ]
    
    if output_file:
        commands.append(f"contacts sel saveFile {output_file}")
    
    # Create temporary script file
    script_file = "temp_ligand_pi_stacking.cxc"
    with open(script_file, 'w') as f:
        for cmd in commands:
            f.write(cmd + '\n')
        f.write('exit\n')
    
    try:
        # Run ChimeraX script
        result = _run_chimerax_script(script_file)
        
        return result.stdout
        
    finally:
        # Clean up temporary file
        if os.path.exists(script_file):
            os.remove(script_file)

def find_aromatic_clusters(pdb_file: str, cluster_distance: float = 8.0, min_cluster_size: int = 3, 
                          output_file: str = None) -> str:
    """
    Find clusters of aromatic residues that may form π-π stacking networks
    
    Args:
        pdb_file: Path to the PDB file
        cluster_distance: Maximum distance for clustering aromatic residues (default: 8.0 Å)
        min_cluster_size: Minimum number of residues in a cluster (default: 3)
        output_file: Output file path (optional)
    
    Returns:
        Aromatic cluster information as string
    """
    
    # Build ChimeraX commands for aromatic cluster detection
    commands = [
        f"open {pdb_file}",
        "select :phe,tyr,trp,his",
        f"contacts sel restrict cross distance {cluster_distance} log true",
        f"# Aromatic residue clusters with distance cutoff: {cluster_distance} Å",
        f"# Minimum cluster size: {min_cluster_size} residues",
    ]
    
    if output_file:
        commands.append(f"contacts sel saveFile {output_file}")
    
    # Create temporary script file
    script_file = "temp_aromatic_clusters.cxc"
    with open(script_file, 'w') as f:
        for cmd in commands:
            f.write(cmd + '\n')
        f.write('exit\n')
    
    try:
        # Run ChimeraX script
        result = _run_chimerax_script(script_file)
        
        return result.stdout
        
    finally:
        # Clean up temporary file
        if os.path.exists(script_file):
            os.remove(script_file)
