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

def validate_complex_parameters(protein_count=1, dna_sequences=None, dna_counts=None, ligands=None, ligand_counts=None, model_name="protenix_base_default_v0.5.0", seeds="42"):
    """
    Validate parameters for complex structure prediction

    Args:
        protein_count (int): Number of protein copies (must be >= 1)
        dna_sequences (list): List of DNA sequences as strings ["ATCG...", "GCTA..."]
        dna_counts (list): List of integers specifying count for each DNA sequence [1, 2, ...]
        ligands (list): List of ligand SMILES strings ["SMILES1", "SMILES2", ...]
        ligand_counts (list): List of integers specifying count for each ligand [1, 2, ...]
        model_name (str): Protenix model name (default: "protenix_base_default_v0.5.0")
        seeds (str): Random seeds as comma-separated string (default: "42")

    Returns:
        tuple: (validated_protein_count, validated_dna_data, validated_ligand_data, validated_model_name, validated_seeds)

    Raises:
        ValueError: If any parameter is invalid
    """

    # Validate protein_count
    if not isinstance(protein_count, int) or protein_count < 1:
        raise ValueError("protein_count must be a positive integer")

    # Validate DNA sequences and counts
    validated_dna_data = []
    if dna_sequences is not None:
        if not isinstance(dna_sequences, list):
            raise ValueError("dna_sequences must be a list of strings")

        if dna_counts is not None:
            if not isinstance(dna_counts, list):
                raise ValueError("dna_counts must be a list of integers")
            if len(dna_sequences) != len(dna_counts):
                raise ValueError("dna_sequences and dna_counts must have the same length")
        else:
            dna_counts = [1] * len(dna_sequences)  # Default to 1 copy each

        for i, (dna_seq, count) in enumerate(zip(dna_sequences, dna_counts)):
            if not isinstance(dna_seq, str) or len(dna_seq) == 0:
                raise ValueError(f"DNA sequence {i} must be a non-empty string")
            if not isinstance(count, int) or count < 1:
                raise ValueError(f"DNA count {i} must be a positive integer")

            # Basic DNA sequence validation (only ATCG characters)
            valid_bases = set('ATCGATCG')
            if not all(base.upper() in valid_bases for base in dna_seq):
                raise ValueError(f"DNA sequence {i} contains invalid characters (only A, T, C, G allowed)")

            validated_dna_data.append({
                "sequence": dna_seq.upper(),
                "count": count
            })

    # Validate ligands and counts
    validated_ligand_data = []
    if ligands is not None:
        if not isinstance(ligands, list):
            raise ValueError("ligands must be a list of SMILES strings")

        if ligand_counts is not None:
            if not isinstance(ligand_counts, list):
                raise ValueError("ligand_counts must be a list of integers")
            if len(ligands) != len(ligand_counts):
                raise ValueError("ligands and ligand_counts must have the same length")
        else:
            ligand_counts = [1] * len(ligands)  # Default to 1 copy each

        for i, (ligand_smiles, count) in enumerate(zip(ligands, ligand_counts)):
            if not isinstance(ligand_smiles, str) or len(ligand_smiles) == 0:
                raise ValueError(f"Ligand {i} must be a non-empty SMILES string")
            if not isinstance(count, int) or count < 1:
                raise ValueError(f"Ligand count {i} must be a positive integer")

            validated_ligand_data.append({
                "ligand": ligand_smiles.strip(),
                "count": count
            })

    # Validate model name
    valid_models = [
        "protenix_base_default_v0.5.0",
        "protenix_mini_esm_v0.5.0",
        "protenix_mini_default_v0.5.0",
        "protenix_tiny_default_v0.5.0"
    ]
    if not isinstance(model_name, str) or model_name not in valid_models:
        raise ValueError(f"model_name must be one of: {valid_models}")

    # Validate seeds format
    if not isinstance(seeds, str):
        raise ValueError("seeds must be a string (e.g., '42' or '42,43,44')")

    # Check seeds format - should be comma-separated integers
    try:
        seed_list = [int(s.strip()) for s in seeds.split(',')]
        if any(seed < 0 for seed in seed_list):
            raise ValueError("All seeds must be non-negative integers")
    except ValueError:
        raise ValueError("seeds must be comma-separated integers (e.g., '42' or '42,43,44')")

    return protein_count, validated_dna_data, validated_ligand_data, model_name, seeds

def convert_tts_to_protenix_format(tts_json_file: str, output_json_file: str, msa_dir: str = None,
                                  protein_count: int = 1, dna_sequences: list = None, dna_counts: list = None,
                                  ligands: list = None, ligand_counts: list = None, job_name: str = "complex_prediction"):
    """
    Convert TTS JSON format to Protenix-compatible JSON format with support for protein-DNA-ligand complexes

    Args:
        tts_json_file (str): Path to TTS JSON file containing protein sequences and scores
        output_json_file (str): Path for output Protenix-compatible JSON file
        msa_dir (str, optional): Directory containing MSA files for precomputed MSAs
        protein_count (int): Number of protein copies (default: 1)
        dna_sequences (list, optional): List of DNA sequences as strings ["ATCG...", "GCTA..."]
        dna_counts (list, optional): List of integers specifying count for each DNA sequence [1, 2, ...]
        ligands (list, optional): List of ligand SMILES strings ["SMILES1", "SMILES2", ...]
        ligand_counts (list, optional): List of integers specifying count for each ligand [1, 2, ...]
        job_name (str): Name for the prediction job (default: "complex_prediction")

    Returns:
        str: Path to the converted JSON file

    Raises:
        ValueError: If input parameters are invalid
        FileNotFoundError: If TTS JSON file doesn't exist

    Examples:
        # Simple protein prediction
        convert_tts_to_protenix_format("input.json", "output.json")

        # Protein-DNA complex
        convert_tts_to_protenix_format("input.json", "output.json",
                                     protein_count=1,
                                     dna_sequences=["ATCGATCG"],
                                     dna_counts=[1])

        # Protein-ligand complex
        convert_tts_to_protenix_format("input.json", "output.json",
                                     protein_count=2,
                                     ligands=["CC(C)O"],
                                     ligand_counts=[1])
    """

    # Validate complex parameters
    protein_count, validated_dna_data, validated_ligand_data, _, _ = validate_complex_parameters(
        protein_count=protein_count,
        dna_sequences=dna_sequences,
        dna_counts=dna_counts,
        ligands=ligands,
        ligand_counts=ligand_counts
    )

    # Load TTS JSON data
    if not os.path.exists(tts_json_file):
        raise FileNotFoundError(f"TTS JSON file not found: {tts_json_file}")

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

            # Create base Protenix format entry with required fields
            protenix_entry = {
                "name": f"{job_name}_{sequence_counter}",
                "covalent_bonds": [],  # Required field based on JSON examples
                "sequences": []
            }

            # Add protein chain
            protein_chain = {
                "proteinChain": {
                    "sequence": protein_seq,
                    "count": protein_count,
                    "modifications": []  # Required field based on JSON examples
                }
            }

            # Add MSA information if msa_dir is provided
            if msa_dir:
                # MSA directory should point to the numbered subdirectory containing pairing.a3m and non_pairing.a3m
                # Use absolute path to avoid relative path issues
                msa_sequence_dir = os.path.abspath(os.path.join(msa_dir, str(sequence_counter)))
                protein_chain["proteinChain"]["msa"] = {
                    "precomputed_msa_dir": msa_sequence_dir,
                    "pairing_db": "uniref100",
                    "pairing_db_fpath": None,
                    "non_pairing_db_fpath": None,
                    "search_too": None,
                    "msa_save_dir": None
                }

            protenix_entry["sequences"].append(protein_chain)

            # Add DNA sequences if provided
            for dna_data in validated_dna_data:
                dna_entry = {
                    "dnaSequence": {
                        "sequence": dna_data["sequence"],
                        "count": dna_data["count"],
                        "modifications": []  # Required field based on JSON examples
                    }
                }
                protenix_entry["sequences"].append(dna_entry)

            # Add ligands if provided
            for ligand_data in validated_ligand_data:
                ligand_entry = {
                    "ligand": {
                        "ligand": ligand_data["ligand"],
                        "count": ligand_data["count"]
                    }
                }
                protenix_entry["sequences"].append(ligand_entry)

            protenix_format.append(protenix_entry)
            sequence_counter += 1  # Increment counter for next sequence

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_json_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Write the converted format
    with open(output_json_file, 'w') as f:
        json.dump(protenix_format, f, indent=2)

    print(f"Successfully converted TTS format to Protenix format: {output_json_file}")
    print(f"Generated {len(protenix_format)} complex structure(s)")
    if validated_dna_data:
        print(f"Included {len(validated_dna_data)} DNA sequence(s)")
    if validated_ligand_data:
        print(f"Included {len(validated_ligand_data)} ligand(s)")

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


def build_protenix_predict_command(json_file: str, output_dir: str, use_msa: bool = True,
                                   precompute_msa: bool = True, model_name: str = "protenix_base_default_v0.5.0",
                                   seeds: str = "42", protenix_executable: str = "protenix"):
    """
    Build Protenix predict command based on MSA and model preferences

    Args:
        json_file (str): Path to Protenix-compatible JSON input file
        output_dir (str): Output directory for prediction results
        use_msa (bool): Whether to use MSA features (default: True)
        precompute_msa (bool): Whether MSA is precomputed (default: True)
        model_name (str): Protenix model name (default: "protenix_base_default_v0.5.0")
        seeds (str): Random seeds as comma-separated string (default: "42")
        protenix_executable (str): Path to protenix executable (default: "protenix")

    Returns:
        list: Command line arguments list for subprocess.run()

    Examples:
        # Precomputed MSA mode
        cmd = build_protenix_predict_command("input.json", "./output", use_msa=True, precompute_msa=True)

        # Real-time MSA search
        cmd = build_protenix_predict_command("input.json", "./output", use_msa=True, precompute_msa=False)

        # ESM mode (no MSA)
        cmd = build_protenix_predict_command("input.json", "./output", use_msa=False)
    """

    base_command = [
        protenix_executable, "predict",
        "--input", json_file,
        "--out_dir", output_dir,
        "--seeds", seeds
    ]

    if not use_msa:
        # ESM mode: disable MSA and use ESM model
        return base_command + [
            "--model_name", "protenix_mini_esm_v0.5.0",
            "--use_msa", "false"
        ]
    elif precompute_msa:
        # Precomputed MSA mode: JSON contains MSA paths
        return base_command + ["--model_name", model_name]
    else:
        # Real-time MSA search mode: let Protenix search MSA automatically
        return base_command + ["--use_msa", "true"]


def structure_prediction(tts_json_file: str, output_dir: str = "./output",
                        protein_count: int = 1, dna_sequences: list = None, dna_counts: list = None,
                        ligands: list = None, ligand_counts: list = None,
                        use_msa: bool = True, precompute_msa: bool = True,
                        model_name: str = "protenix_base_default_v0.5.0", seeds: str = "42",
                        job_name: str = "complex_prediction"):
    """
    Perform structure prediction using Protenix with support for protein-DNA-ligand complexes

    This function performs a complete structure prediction workflow with three modes:
    1. Precomputed MSA: TTS JSON → FASTA → MSA generation → Protenix JSON → prediction
    2. Real-time MSA: TTS JSON → Protenix JSON → prediction (with --use_msa true)
    3. ESM mode: TTS JSON → Protenix JSON → prediction (with --use_msa false)

    Args:
        tts_json_file (str): Path to TTS JSON file containing protein sequences and scores
        output_dir (str): Output directory for all generated files (default: "./output")
        protein_count (int): Number of protein copies (default: 1)
        dna_sequences (list, optional): List of DNA sequences as strings ["ATCG...", "GCTA..."]
        dna_counts (list, optional): List of integers specifying count for each DNA sequence [1, 2, ...]
        ligands (list, optional): List of ligand SMILES strings ["SMILES1", "SMILES2", ...]
        ligand_counts (list, optional): List of integers specifying count for each ligand [1, 2, ...]
        use_msa (bool): Whether to use MSA features (default: True)
        precompute_msa (bool): Whether to precompute MSA using 'protenix msa' (default: True)
        model_name (str): Protenix model name (default: "protenix_base_default_v0.5.0")
        seeds (str): Random seeds as comma-separated string (default: "42")
        job_name (str): Name for the prediction job (default: "complex_prediction")

    Returns:
        dict: Dictionary containing paths to generated files and prediction results

    Raises:
        ValueError: If input parameters are invalid
        FileNotFoundError: If TTS JSON file doesn't exist

    Examples:
        # Simple protein prediction with precomputed MSA
        result = structure_prediction("input.json")

        # Protein-DNA complex with real-time MSA search
        result = structure_prediction("input.json",
                                    dna_sequences=["ATCGATCG"],
                                    dna_counts=[1],
                                    use_msa=True,
                                    precompute_msa=False)

        # Protein-ligand complex with ESM mode (no MSA)
        result = structure_prediction("input.json",
                                    protein_count=2,
                                    ligands=["CC(C)O"],
                                    ligand_counts=[1],
                                    use_msa=False)

        # Complex with multiple components
        result = structure_prediction("input.json",
                                    protein_count=1,
                                    dna_sequences=["ATCG", "GCTA"],
                                    dna_counts=[1, 2],
                                    ligands=["SMILES1", "SMILES2"],
                                    ligand_counts=[1, 1],
                                    seeds="42,43,44")
    """

    # Validate all parameters
    protein_count, validated_dna_data, validated_ligand_data, model_name, seeds = validate_complex_parameters(
        protein_count=protein_count,
        dna_sequences=dna_sequences,
        dna_counts=dna_counts,
        ligands=ligands,
        ligand_counts=ligand_counts,
        model_name=model_name,
        seeds=seeds
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Setup Protenix executable path
    conda_base = os.path.expanduser("~/miniconda3")  # or ~/anaconda3 depending on your installation
    protenix_env_path = os.path.join(conda_base, "envs", "protenix", "bin")
    protenix_executable = os.path.join(protenix_env_path, "protenix")

    # Alternative: use system protenix if conda path doesn't work
    # protenix_executable = "protenix"

    # Set environment variables for the protenix environment
    env = os.environ.copy()
    env["PATH"] = f"{protenix_env_path}:{env.get('PATH', '')}"

    msa_result = None
    msa_dir_for_json = None

    # Step 1: Generate MSA if using precomputed MSA mode
    if use_msa and precompute_msa:
        print("Step 1: Generating MSA files using protenix msa...")

        # Create FASTA file from TTS JSON (only for protein sequences)
        fasta_file = os.path.join(output_dir, "sequences.fasta")
        create_fasta_from_tts(tts_json_file, fasta_file)

        # Generate MSA using protenix msa command
        msa_command = [protenix_executable, "msa", "--input", fasta_file, "--out_dir", output_dir]

        try:
            msa_result = subprocess.run(msa_command, capture_output=True, text=True, check=True, env=env)
            print("MSA generation completed successfully")
            print(f"MSA stdout: {msa_result.stdout}")
            msa_dir_for_json = output_dir  # Use output_dir as MSA directory for JSON
        except subprocess.CalledProcessError as e:
            print(f"MSA generation failed: {e}")
            print(f"MSA stderr: {e.stderr}")
            return {
                "success": False,
                "error": f"MSA generation failed: {e}",
                "msa_stderr": e.stderr,
                "mode": "precomputed_msa"
            }
    else:
        print("Step 1: Skipping MSA generation (using real-time search or ESM mode)")

    # Step 2: Convert TTS JSON to Protenix format
    print("Step 2: Converting TTS JSON to Protenix format...")
    protenix_json_file = os.path.join(output_dir, "protenix_input.json")

    convert_tts_to_protenix_format(
        tts_json_file=tts_json_file,
        output_json_file=protenix_json_file,
        msa_dir=msa_dir_for_json,  # Only set if precomputed MSA
        protein_count=protein_count,
        dna_sequences=dna_sequences,
        dna_counts=dna_counts,
        ligands=ligands,
        ligand_counts=ligand_counts,
        job_name=job_name
    )

    # Step 3: Build and run prediction command
    print("Step 3: Performing structure prediction...")

    predict_command = build_protenix_predict_command(
        json_file=protenix_json_file,
        output_dir=output_dir,
        use_msa=use_msa,
        precompute_msa=precompute_msa,
        model_name=model_name,
        seeds=seeds,
        protenix_executable=protenix_executable
    )

    print(f"Prediction command: {' '.join(predict_command)}")

    try:
        predict_result = subprocess.run(predict_command, capture_output=True, text=True, check=True, env=env)
        print("Structure prediction completed successfully")
        print(f"Prediction stdout: {predict_result.stdout}")

        # Determine prediction mode for result reporting
        if not use_msa:
            mode = "esm_no_msa"
        elif precompute_msa:
            mode = "precomputed_msa"
        else:
            mode = "realtime_msa"

        # Return success information
        result = {
            "success": True,
            "mode": mode,
            "output_dir": output_dir,
            "protenix_json_file": protenix_json_file,
            "prediction_stdout": predict_result.stdout,
            "message": f"Complex structure prediction completed successfully in {mode} mode",
            "complex_info": {
                "proteins": protein_count,
                "dna_sequences": len(validated_dna_data) if validated_dna_data else 0,
                "ligands": len(validated_ligand_data) if validated_ligand_data else 0
            }
        }

        if msa_result:
            result.update({
                "fasta_file": fasta_file,
                "msa_stdout": msa_result.stdout
            })

        return result

    except subprocess.CalledProcessError as e:
        print(f"Structure prediction failed: {e}")
        print(f"Prediction stderr: {e.stderr}")

        # Determine mode for error reporting
        if not use_msa:
            mode = "esm_no_msa"
        elif precompute_msa:
            mode = "precomputed_msa"
        else:
            mode = "realtime_msa"

        error_result = {
            "success": False,
            "mode": mode,
            "error": f"Structure prediction failed: {e}",
            "prediction_stderr": e.stderr,
            "output_dir": output_dir,
            "protenix_json_file": protenix_json_file
        }

        if msa_result:
            error_result.update({
                "fasta_file": fasta_file,
                "msa_stdout": msa_result.stdout
            })

        return error_result

