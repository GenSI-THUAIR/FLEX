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

def find_hydrogen_bonds_around_atoms(pdb_file: str, target_atom_spec: str, distance_range: float=5.0, output_file: str=None):
    """
    Find hydrogen bonds within a specific distance range around target atoms using ChimeraX
    
    Args:
        pdb_file: Path to the PDB file
        target_atom_spec: Target atom specification (e.g., ":150@N*" or ":150-160")
        distance_range: Search distance range in Angstroms
        output_file: Output file path (optional)
    
    Returns:
        Hydrogen bond information as string
    """
    
    # Build ChimeraX commands
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
    
    commands.append("hbonds sel log true")
    
    if output_file:
        commands.append(f"hbonds sel saveFile {output_file}")
    
    # Create temporary script file
    script_file = "temp_hbonds.cxc"
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

# Usage examples
def find_residue_hbonds_in_range(pdb_file: str, residue_num: int, atom_name: str = "*", distance: float = 5.0) -> str:
    """
    Find hydrogen bonds around specific residue atoms within a specified distance
    
    Args:
        pdb_file: Path to the PDB file
        residue_num: Residue number
        atom_name: Atom name (default: all atoms "*")
        distance: Search distance in Angstroms
    
    Returns:
        Hydrogen bond information as string
    """
    
    atom_spec = f":{residue_num}@{atom_name}"
    return find_hydrogen_bonds_around_atoms(pdb_file, atom_spec, distance)

def find_chain_residue_atom_hbonds(pdb_file: str, chain_id: str, residue_num: int, atom_name: str, distance: float = 5.0) -> str:
    """
    Find hydrogen bonds around a specific atom in a specific chain and residue
    
    Args:
        pdb_file: Path to the PDB file
        chain_id: Chain identifier
        residue_num: Residue number
        atom_name: Atom name
        distance: Search distance in Angstroms
    
    Returns:
        Hydrogen bond information as string
    """
    atom_spec = f"/{chain_id}:{residue_num}@{atom_name}"
    return find_hydrogen_bonds_around_atoms(pdb_file, atom_spec, distance)

# Additional utility functions
def analyze_protein_hbonds(pdb_file: str, output_file: str = None) -> str:
    """
    Analyze all hydrogen bonds in a protein structure
    
    Args:
        pdb_file: Path to the PDB file
        output_file: Output file path (optional)
    
    Returns:
        Complete hydrogen bond analysis as string
    """
    # Use all protein atoms without distance restriction
    return find_hydrogen_bonds_around_atoms(pdb_file, "protein", 0, output_file)

def find_ligand_protein_hbonds(pdb_file: str, ligand_spec: str = "ligand", distance: float = 5.0, output_file: str = None) -> str:
    """
    Find hydrogen bonds between ligand and protein within specified distance
    
    Args:
        pdb_file: Path to the PDB file
        ligand_spec: Ligand specification (default: "ligand")
        distance: Search distance in Angstroms
        output_file: Output file path (optional)
    
    Returns:
        Ligand-protein hydrogen bond information as string
    """
    return find_hydrogen_bonds_around_atoms(pdb_file, ligand_spec, distance, output_file)

# Salt bridge analysis functions
def find_salt_bridges_around_atoms(pdb_file: str, target_atom_spec: str, distance_range: float = 4.0, output_file: str = None) -> str:
    """
    Find salt bridges within a specific distance range around target atoms using ChimeraX
    
    Args:
        pdb_file: Path to the PDB file
        target_atom_spec: Target atom specification (e.g., ":150@N*" or ":150-160")
        distance_range: Search distance range in Angstroms (default: 4.0 for salt bridges)
        output_file: Output file path (optional)
    
    Returns:
        Salt bridge information as string
    """
    
    # Build ChimeraX commands for salt bridge detection
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
    
    # Select charged residues for salt bridge analysis
    commands.extend([
        "select :arg,lys,his",
        "name frozen positive sel",
        "select :asp,glu", 
        "name frozen negative sel",
        f"contacts positive restrict negative distanceOnly {distance_range} log true",
    ])
    
    if output_file:
        commands.append(f"contacts positive restrict negative distanceOnly {distance_range} saveFile {output_file}")
    
    # Create temporary script file
    script_file = "temp_saltbridges.cxc"
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

def find_residue_salt_bridges_in_range(pdb_file: str, residue_num: int, distance: float = 4.0) -> str:
    """
    Find salt bridges around a specific residue within a specified distance
    
    Args:
        pdb_file: Path to the PDB file
        residue_num: Residue number
        distance: Search distance in Angstroms (default: 4.0 for salt bridges)
    
    Returns:
        Salt bridge information as string
    """
    
    atom_spec = f":{residue_num}"
    return find_salt_bridges_around_atoms(pdb_file, atom_spec, distance)

def find_chain_residue_salt_bridges(pdb_file: str, chain_id: str, residue_num: int, distance: float = 4.0) -> str:
    """
    Find salt bridges around a specific residue in a specific chain
    
    Args:
        pdb_file: Path to the PDB file
        chain_id: Chain identifier
        residue_num: Residue number
        distance: Search distance in Angstroms (default: 4.0 for salt bridges)
    
    Returns:
        Salt bridge information as string
    """
    atom_spec = f"/{chain_id}:{residue_num}"
    return find_salt_bridges_around_atoms(pdb_file, atom_spec, distance)

def analyze_protein_salt_bridges(pdb_file: str, output_file: str = None) -> str:
    """
    Analyze all salt bridges in a protein structure
    
    Args:
        pdb_file: Path to the PDB file
        output_file: Output file path (optional)
    
    Returns:
        Complete salt bridge analysis as string
    """
    return find_salt_bridges_around_atoms(pdb_file, "protein", 0, output_file)

def find_interface_salt_bridges(pdb_file: str, interface_spec1: str, interface_spec2: str, distance: float = 4.0, output_file: str = None) -> str:
    """
    Find salt bridges at the interface between two molecular entities
    
    Args:
        pdb_file: Path to the PDB file
        interface_spec1: First interface specification (e.g., "/A" for chain A)
        interface_spec2: Second interface specification (e.g., "/B" for chain B)
        distance: Search distance in Angstroms (default: 4.0 for salt bridges)
        output_file: Output file path (optional)
    
    Returns:
        Interface salt bridge information as string
    """
    
    # Build ChimeraX commands for interface salt bridge detection
    commands = [
        f"open {pdb_file}",
        f"select ({interface_spec1} & :arg,lys,his) | ({interface_spec2} & :arg,lys,his)",
        "name frozen positive sel",
        f"select ({interface_spec1} & :asp,glu) | ({interface_spec2} & :asp,glu)",
        "name frozen negative sel",
        f"contacts positive restrict negative distanceOnly {distance} log true",
    ]
    
    if output_file:
        commands.append(f"contacts positive restrict negative distanceOnly {distance} saveFile {output_file}")
    
    # Create temporary script file
    script_file = "temp_interface_saltbridges.cxc"
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

def find_charged_residue_interactions(pdb_file: str, distance: float = 4.0, ph_value: float = 7.0, output_file: str = None) -> str:
    """
    Find interactions between charged residues (salt bridges and electrostatic interactions)
    
    Args:
        pdb_file: Path to the PDB file
        distance: Search distance in Angstroms (default: 4.0)
        ph_value: pH value for protonation state consideration (default: 7.0)
        output_file: Output file path (optional)
    
    Returns:
        Charged residue interaction information as string
    """
    
    # Determine His protonation based on pH
    his_spec = ":his" if ph_value < 6.0 else ":hie,hid,hip"
    
    # Build ChimeraX commands
    commands = [
        f"open {pdb_file}",
        f"select :arg,lys | {his_spec}",
        "name frozen positive sel", 
        "select :asp,glu",
        "name frozen negative sel",
        f"contacts positive restrict negative distanceOnly {distance} log true",
        "# Show electrostatic interactions between charged residues",
    ]
    
    if output_file:
        commands.append(f"contacts positive restrict negative distanceOnly {distance} saveFile {output_file}")
    
    # Create temporary script file
    script_file = "temp_charged_interactions.cxc"
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


def convert_tts_to_protenix_format(tts_json_file: str, output_json_file: str, msa_dir: str = None, job_name: str = "protein_prediction"):
    """
    Convert TTS JSON format to Protenix-compatible JSON format
    
    Args:
        tts_json_file: Path to TTS JSON file containing protein sequences and scores
        output_json_file: Path for output Protenix-compatible JSON file
        msa_dir: Directory containing MSA files (optional, for precomputed MSAs)
        job_name: Name for the prediction job
    
    Returns:
        Path to the converted JSON file
    """
    
    with open(tts_json_file, 'r') as f:
        tts_data = json.load(f)
    
    protenix_format = []
    sequence_counter = 0  # Counter for MSA directory numbering and sequence naming (starts from 0)
    
    for sample in tts_data:
        sequences = sample.get("sequences", [])
        
        if not sequences:
            continue
        
        # Process all sequences in the sample, not just the best one
        for sequence_dict in sequences:
            protein_seq = sequence_dict["seq"]
            
            # Create Protenix format entry
            protenix_entry = {
                "name": f"mutant_{sequence_counter}_structure_prediction",
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": protein_seq,
                            "count": 1
                        }
                    }
                ]
            }
            
            # Add MSA information if msa_dir is provided
            if msa_dir:
                # MSA directory should point to the numbered subdirectory containing pairing.a3m and non_pairing.a3m
                # Use absolute path to avoid relative path issues
                msa_sequence_dir = os.path.abspath(os.path.join(msa_dir, str(sequence_counter)))
                protenix_entry["sequences"][0]["proteinChain"]["msa"] = {
                    "precomputed_msa_dir": msa_sequence_dir,
                    "pairing_db": "uniref100"
                }
            
            protenix_format.append(protenix_entry)
            sequence_counter += 1  # Increment counter for next sequence
    
    # Write the converted format
    with open(output_json_file, 'w') as f:
        json.dump(protenix_format, f, indent=2)
    
    return output_json_file


def create_fasta_from_tts(tts_json_file: str, output_fasta_file: str):
    """
    Create FASTA file from TTS JSON for MSA generation
    
    Args:
        tts_json_file: Path to TTS JSON file
        output_fasta_file: Path for output FASTA file
    
    Returns:
        Path to the created FASTA file
    """
    
    with open(tts_json_file, 'r') as f:
        tts_data = json.load(f)
    
    with open(output_fasta_file, 'w') as f:
        sequence_counter = 0  # Counter for naming sequences
        for sample in tts_data:
            sequences = sample.get("sequences", [])
            
            if not sequences:
                continue
            
            # Process all sequences in the sample, not just the best one
            for sequence_dict in sequences:
                protein_seq = sequence_dict["seq"]
                
                # Use a counter-based naming scheme
                f.write(f">mutant_{sequence_counter}\n")
                f.write(f"{protein_seq}\n")
                sequence_counter += 1
    
    return output_fasta_file


def structure_prediction(tts_json_file: str, output_dir: str = "./output", model_name: str = "protenix_base_default_v0.5.0", seeds: str = "101"):
    """
    Perform structure prediction using Protenix from TTS JSON file
    
    This function performs a complete structure prediction workflow:
    1. Converts TTS JSON to FASTA format
    2. Generates MSA files using protenix msa
    3. Converts TTS JSON to Protenix-compatible JSON format
    4. Performs structure prediction using protenix predict
    
    Args:
        tts_json_file: Path to TTS JSON file containing protein sequences and scores
        output_dir: Output directory for all generated files (default: "./output")
        model_name: Protenix model name (default: "protenix_base_default_v0.5.0")
        seeds: Random seeds for prediction (default: "101")
    
    Returns:
        Dictionary containing paths to generated files and prediction results
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary FASTA file
    fasta_file = os.path.join(output_dir, "sequences.fasta")
    create_fasta_from_tts(tts_json_file, fasta_file)
    
    # Step 1: Generate MSA files using protenix msa
    print("Step 1: Generating MSA files...")
    # Use direct path to conda environment instead of conda activate
    conda_base = os.path.expanduser("~/miniconda3")  # or ~/anaconda3 depending on your installation
    protenix_env_path = os.path.join(conda_base, "envs", "protenix", "bin")
    protenix_executable = os.path.join(protenix_env_path, "protenix")
    
    # # change by yupei
    # protenix_executable = "protenix"
    
    msa_command = [protenix_executable, "msa", "--input", fasta_file, "--out_dir", output_dir]
    
    try:
        # Set environment variables for the protenix environment
        env = os.environ.copy()
        env["PATH"] = f"{protenix_env_path}:{env.get('PATH', '')}"
        
        msa_result = subprocess.run(msa_command, capture_output=True, text=True, check=True, env=env)
        print("MSA generation completed successfully")
        print(f"MSA stdout: {msa_result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"MSA generation failed: {e}")
        print(f"MSA stderr: {e.stderr}")
        return {
            "success": False,
            "error": f"MSA generation failed: {e}",
            "msa_stderr": e.stderr
        }
    
    # Step 2: Convert TTS JSON to Protenix format with MSA information
    print("Step 2: Converting TTS JSON to Protenix format...")
    protenix_json_file = os.path.join(output_dir, "protenix_input.json")
    convert_tts_to_protenix_format(tts_json_file, protenix_json_file, output_dir)
    
    # Step 3: Perform structure prediction
    print("Step 3: Performing structure prediction...")
    # Use the same protenix executable path
    predict_command = [protenix_executable, "predict", "--input", protenix_json_file, "--out_dir", output_dir, "--seeds", seeds, "--model_name", model_name]
    
    try:
        # Use the same environment setup
        predict_result = subprocess.run(predict_command, capture_output=True, text=True, check=True, env=env)
        print("Structure prediction completed successfully")
        print(f"Prediction stdout: {predict_result.stdout}")
        
        # Return success information
        return {
            "success": True,
            "output_dir": output_dir,
            "fasta_file": fasta_file,
            "protenix_json_file": protenix_json_file,
            "msa_stdout": msa_result.stdout,
            "prediction_stdout": predict_result.stdout,
            "message": "Structure prediction completed successfully"
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Structure prediction failed: {e}")
        print(f"Prediction stderr: {e.stderr}")
        return {
            "success": False,
            "error": f"Structure prediction failed: {e}",
            "prediction_stderr": e.stderr,
            "output_dir": output_dir,
            "fasta_file": fasta_file,
            "protenix_json_file": protenix_json_file
        }

