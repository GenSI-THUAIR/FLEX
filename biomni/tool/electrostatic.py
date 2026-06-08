import os
import subprocess
import tempfile
import re

def _run_chimerax_script(script_file):
    """
    Run ChimeraX script using nogui mode.
    """
    result = subprocess.run([
        'chimerax-daily', '--nogui', '--script', script_file
    ], capture_output=True, text=True)
    return result

def analyze_electrostatic(pdb_file: str, target_residue: str, distance_range: float=5.0, output_file: str=None):
    """
    Analyze electrostatic potential within a specific distance range around target residue using ChimeraX
    
    This function performs electrostatic analysis by:
    1. Loading the PDB structure and adding hydrogens
    2. Computing atomic charges using Gasteiger method
    3. Calculating Coulombic electrostatic potential for the target residue
    4. Selecting atoms within specified distance range around the target
    5. Computing electrostatic potential for the selected region
    
    Args:
        pdb_file: Path to the PDB file
        target_residue: Target residue specification (e.g., "/A:3" for chain A residue 3)
        distance_range: Search distance range in Angstroms (default: 5.0)
        output_file: Output file path for saving electrostatic data (optional)
    
    Returns:
        Electrostatic analysis information as string containing:
        - Coulombic potential values (minimum, mean, maximum)
        - Center of mass coordinates for target residue
        - Number of atoms selected in the distance range
        - Surface electrostatic potential coloring information
    """
    
    # Build ChimeraX commands
    commands = [
        f"open {pdb_file}",
        "addh",
        "addcharge #1 method gasteiger",
        f"coulombic {target_residue}",
        f"measure center {target_residue}",
        f"select zone {target_residue} {distance_range} protein res t",
    ]
    
    if output_file:
        commands.append(f"coulombic sel saveFile {output_file}")
    else:
        commands.append(f"coulombic sel")
    
    # Create temporary script file
    script_file = "temp_elec.cxc"
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
