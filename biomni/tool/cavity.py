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

def analyze_cavity(pdb_file: str, target_residue: str, distance_range: float=5.0, output_file: str=None, probe_radius: float = 1.4):
    """
    Analyze cavity volume around target residue using ChimeraX molecular surface calculation
    
    Args:
        pdb_file: Path to the PDB file
        target_residue: Target residue specification (e.g., "/A:3" or ":150")
        distance_range: Search distance range in Angstroms (default: 5.0)
        output_file: Output file path (optional)
        probe_radius: Probe radius for surface calculation in Angstroms (default: 1.4)
    
    Returns:
        Cavity volume information as string
    """
    
    # Build ChimeraX commands
    commands = [
        f"open {pdb_file}",
        "addh",
        f"select zone {target_residue} {distance_range}",
        f"surface sel probeRadius {probe_radius}",
    ]
    
    if output_file:
        commands.append(f"measure volume #1.1 saveFile {output_file}")
    else:
        commands.append(f"measure volume #1.1")
    
    # Create temporary script file
    script_file = "temp_cavity.cxc"
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