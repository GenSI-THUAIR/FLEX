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

def get_residues(pdb_file: str, target_residue: str, distance_range: float=5.0):
    """
    Get residues within a specific distance range around a target residue using ChimeraX
    
    Args:
        pdb_file: Path to the PDB file
        target_residue: Target residue specification (e.g., "/A:3" or ":150")
        distance_range: Search distance range in Angstroms (default: 5.0)
    
    Returns:
        List of dictionaries containing residue information with keys:
        - 'residue_pos': Residue position (e.g., "/A:3")
        - 'residue_type': Residue type (e.g., "CYS", "ALA")
    """

    # Build ChimeraX commands
    commands = [
        f"open {pdb_file}",
        f"addh",
        f"info residues {target_residue}",
        f"select {target_residue}",
        f"select zone sel {distance_range}",
        "info residues sel"
    ]

    script_file = "temp_hydro.cxc"
    with open(script_file, 'w') as f:
        for cmd in commands:
            f.write(cmd + '\n')
        f.write('exit\n')
    
    try:
        # Run ChimeraX script
        result = _run_chimerax_script(script_file)

        if result.stderr:
            print("STDERR:", result.stderr)
        
        residues_info = []

        residue_pattern = r'residue id (/[A-Z]:\d+) name (\w+)'
        matches = re.findall(residue_pattern, result.stdout)

        for match in matches:
            res_pos, res_type = match
            
            residues_info.append({
                'residue_pos': res_pos,
                'residue_type': res_type
            })
    
        return residues_info
        
    finally:
        # Clean up temporary file
        if os.path.exists(script_file):
            os.remove(script_file)

def analyze_hydrofilicity(pdb_file: str, target_residue: str, distance_range: float=8.0, output_file: str=None):
    """
    Analyze hydrophilicity of residues around a target residue using Kyte-Doolittle scale
    
    Args:
        pdb_file: Path to the PDB file
        target_residue: Target residue specification (e.g., "/A:3" or ":150")
        distance_range: Search distance range in Angstroms (default: 8.0)
        output_file: Output file path (optional, currently not used)
    
    Returns:
        Dictionary containing hydrophilicity analysis results:
        - 'target_residue_kd': Kyte-Doolittle score of the target residue
        - 'residues_count': Number of surrounding residues analyzed
        - 'total_kd_score': Sum of all surrounding residues' KD scores
        - 'average_kd_score': Average KD score of surrounding residues
    """
    KD_SCALE = {
        'ALA': 1.8,   'ARG': -4.5,  'ASN': -3.5,  'ASP': -3.5,
        'CYS': 2.5,   'GLN': -3.5,  'GLU': -3.5,  'GLY': -0.4,
        'HIS': -3.2,  'ILE': 4.5,   'LEU': 3.8,   'LYS': -3.9,
        'MET': 1.9,   'PHE': 2.8,   'PRO': -1.6,  'SER': -0.8,
        'THR': -0.7,  'TRP': -0.9,  'TYR': -1.3,  'VAL': 4.2
    }

    residues_info = get_residues(pdb_file, target_residue, distance_range)
    target_residue_kd = 0.0

    if not residues_info:
        return {
            'target_residue_kd': target_residue_kd,
            'residues_count': 0,
            'total_kd_score': 0.0,
            'average_kd_score': 0.0
        }

    kd_scores = []
    for residue in residues_info:
        if residue['residue_pos'] != target_residue:
            kd_scores.append(KD_SCALE[residue['residue_type'].upper()])
        else:
            target_residue_kd = KD_SCALE[residue['residue_type'].upper()]
    
    result = {
        'target_residue_kd': target_residue_kd,
        'residues_count': len(kd_scores),
        'total_kd_score': sum(kd_scores),
        'average_kd_score': sum(kd_scores) / len(kd_scores)
    }

    return result

def calculate_sasa(pdb_file: str, target_residue: str, output_file: str=None):
    """
    Calculate solvent accessible surface area (SASA) for a target residue using ChimeraX
    
    Args:
        pdb_file: Path to the PDB file
        target_residue: Target residue specification (e.g., "/A:3" or ":150")
        output_file: Output file path for saving SASA results (optional)
    
    Returns:
        ChimeraX output string containing SASA calculation results
    """
    
    # Build ChimeraX commands
    commands = [
        f"open {pdb_file}",
        f"info residues {target_residue}"
    ]
    
    if output_file:
        commands.append(f"measure sasa {target_residue} probeRadius 1.4 saveFile {output_file}")
    else:
        commands.append(f"measure sasa {target_residue} probeRadius 1.4")

    script_file = "temp_hydro.cxc"
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


def analyze_if_on_surface(pdb_file: str, target_residue: str, output_file: str=None):
    """
    Analyze if a residue is on the protein surface by calculating normalized SASA (nSASA)
    
    Args:
        pdb_file: Path to the PDB file
        target_residue: Target residue specification (e.g., "/A:3" or ":150")
        output_file: Output file path for saving SASA results (optional)
    
    Returns:
        Normalized SASA value (0.0-1.0) indicating surface exposure:
        - 0.0: Completely buried
        - 1.0: Fully exposed on surface
        - Values > 0.2 typically indicate surface residues
    
    Raises:
        ValueError: If residue type or SASA information cannot be found
    """
    MAX_SASA = {
        'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0,
        'CYS': 167.0, 'GLN': 225.0, 'GLU': 223.0, 'GLY': 104.0,
        'HIS': 224.0, 'ILE': 197.0, 'LEU': 201.0, 'LYS': 236.0,
        'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0, 'SER': 155.0,
        'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0
    }
    
    result = calculate_sasa(pdb_file, target_residue, output_file)
    
    residue_pattern = r'residue id /[A-Z]:\d+ name (\w+) index \d+'
    residue_match = re.search(residue_pattern, result)
    residue_type = None
    if residue_match:
        residue_type = residue_match.group(1)
    else:
        raise ValueError("未找到残基类型信息")

    sasa_pattern = r'Solvent accessible area for \d+ atoms = ([\d.]+)'
    sasa_match = re.search(sasa_pattern, result)
    sasa = 0.0
    if sasa_match:
        sasa = float(sasa_match.group(1))
    else:
        raise ValueError("未找到sasa信息")

    max_sasa = MAX_SASA.get(residue_type.upper())
    if max_sasa is None:
        raise ValueError(f"未知残基类型: {residue_type}")
    
    nsa = sasa / max_sasa
    nsa = min(nsa, 1.0)
    return nsa
