import os
import subprocess
import json
import tempfile

def _run_chimerax_script(script_file):
    """
    Run ChimeraX script using nogui mode.
    """
    result = subprocess.run([
        'chimerax-daily', '--nogui', '--script', script_file
    ], capture_output=True, text=True)
    return result

def find_disulfide_bonds_around_atoms(pdb_file: str, target_atom_spec: str, distance_range: float = 8.0, output_file: str = None):
    """
    Find disulfide bonds within a specific distance range around target atoms using ChimeraX
    
    Args:
        pdb_file: Path to the PDB file
        target_atom_spec: Target atom specification (e.g., ":150@SG" or ":150-160")
        distance_range: Search distance range in Angstroms (default: 8.0 for disulfide bonds)
        output_file: Output file path (optional)
    
    Returns:
        Disulfide bond information as string
    """
    
    # Build ChimeraX commands for disulfide bond detection
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
    
    # Select cysteine sulfur atoms for disulfide bond analysis
    commands.extend([
        "select sel & :cys@SG",  # Select cysteine sulfur atoms
        "distance sel sel 1.8 3.0",  # Typical disulfide bond distance range
        "# Disulfide bonds typically range from 1.8-3.0 Angstroms",
    ])
    
    if output_file:
        commands.append(f"save {output_file} sel format text")
    
    # Create temporary script file
    script_file = "temp_disulfides.cxc"
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

def find_residue_disulfide_bonds_in_range(pdb_file: str, residue_num: int, distance: float = 8.0) -> str:
    """
    Find disulfide bonds around a specific cysteine residue within a specified distance
    
    Args:
        pdb_file: Path to the PDB file
        residue_num: Residue number
        distance: Search distance in Angstroms (default: 8.0 for disulfide bonds)
    
    Returns:
        Disulfide bond information as string
    """
    
    atom_spec = f":{residue_num}@SG"
    return find_disulfide_bonds_around_atoms(pdb_file, atom_spec, distance)

def find_chain_residue_disulfide_bonds(pdb_file: str, chain_id: str, residue_num: int, distance: float = 8.0) -> str:
    """
    Find disulfide bonds around a specific cysteine residue in a specific chain
    
    Args:
        pdb_file: Path to the PDB file
        chain_id: Chain identifier
        residue_num: Residue number
        distance: Search distance in Angstroms (default: 8.0 for disulfide bonds)
    
    Returns:
        Disulfide bond information as string
    """
    atom_spec = f"/{chain_id}:{residue_num}@SG"
    return find_disulfide_bonds_around_atoms(pdb_file, atom_spec, distance)

def analyze_protein_disulfide_bonds(pdb_file: str, output_file: str = None) -> str:
    """
    Analyze all disulfide bonds in a protein structure
    
    Args:
        pdb_file: Path to the PDB file
        output_file: Output file path (optional)
    
    Returns:
        Complete disulfide bond analysis as string
    """
    return find_disulfide_bonds_around_atoms(pdb_file, "protein", 0, output_file)

def find_inter_chain_disulfide_bonds(pdb_file: str, chain1: str, chain2: str, output_file: str = None) -> str:
    """
    Find disulfide bonds between two different chains
    
    Args:
        pdb_file: Path to the PDB file
        chain1: First chain identifier
        chain2: Second chain identifier
        output_file: Output file path (optional)
    
    Returns:
        Inter-chain disulfide bond information as string
    """
    
    # Build ChimeraX commands for inter-chain disulfide detection
    commands = [
        f"open {pdb_file}",
        f"select (/{chain1}:cys@SG) | (/{chain2}:cys@SG)",
        "distance sel sel 1.8 3.0",
        f"# Inter-chain disulfide bonds between chain {chain1} and {chain2}",
    ]
    
    if output_file:
        commands.append(f"save {output_file} sel format text")
    
    # Create temporary script file
    script_file = "temp_interchain_disulfides.cxc"
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

def find_intra_chain_disulfide_bonds(pdb_file: str, chain_id: str, output_file: str = None) -> str:
    """
    Find disulfide bonds within a single chain
    
    Args:
        pdb_file: Path to the PDB file
        chain_id: Chain identifier
        output_file: Output file path (optional)
    
    Returns:
        Intra-chain disulfide bond information as string
    """
    
    # Build ChimeraX commands for intra-chain disulfide detection
    commands = [
        f"open {pdb_file}",
        f"select /{chain_id}:cys@SG",
        "distance sel sel 1.8 3.0",
        f"# Intra-chain disulfide bonds within chain {chain_id}",
    ]
    
    if output_file:
        commands.append(f"save {output_file} sel format text")
    
    # Create temporary script file
    script_file = "temp_intrachain_disulfides.cxc"
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