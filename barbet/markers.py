import os
import subprocess
from typing import Dict, List, Tuple
import gzip
import shutil
import tempfile
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor

from rich import progress

# Marker information
BAC120_MARKERS = [
    "PF00380.20",
    "PF00410.20",
    "PF00466.21",
    "PF01025.20",
    "PF02576.18",
    "PF03726.15",
    "TIGR00006",
    "TIGR00019",
    "TIGR00020",
    "TIGR00029",
    "TIGR00043",
    "TIGR00054",
    "TIGR00059",
    "TIGR00061",
    "TIGR00064",
    "TIGR00065",
    "TIGR00082",
    "TIGR00083",
    "TIGR00084",
    "TIGR00086",
    "TIGR00088",
    "TIGR00090",
    "TIGR00092",
    "TIGR00095",
    "TIGR00115",
    "TIGR00116",
    "TIGR00138",
    "TIGR00158",
    "TIGR00166",
    "TIGR00168",
    "TIGR00186",
    "TIGR00194",
    "TIGR00250",
    "TIGR00337",
    "TIGR00344",
    "TIGR00362",
    "TIGR00382",
    "TIGR00392",
    "TIGR00396",
    "TIGR00398",
    "TIGR00414",
    "TIGR00416",
    "TIGR00420",
    "TIGR00431",
    "TIGR00435",
    "TIGR00436",
    "TIGR00442",
    "TIGR00445",
    "TIGR00456",
    "TIGR00459",
    "TIGR00460",
    "TIGR00468",
    "TIGR00472",
    "TIGR00487",
    "TIGR00496",
    "TIGR00539",
    "TIGR00580",
    "TIGR00593",
    "TIGR00615",
    "TIGR00631",
    "TIGR00634",
    "TIGR00635",
    "TIGR00643",
    "TIGR00663",
    "TIGR00717",
    "TIGR00755",
    "TIGR00810",
    "TIGR00922",
    "TIGR00928",
    "TIGR00959",
    "TIGR00963",
    "TIGR00964",
    "TIGR00967",
    "TIGR01009",
    "TIGR01011",
    "TIGR01017",
    "TIGR01021",
    "TIGR01029",
    "TIGR01032",
    "TIGR01039",
    "TIGR01044",
    "TIGR01059",
    "TIGR01063",
    "TIGR01066",
    "TIGR01071",
    "TIGR01079",
    "TIGR01082",
    "TIGR01087",
    "TIGR01128",
    "TIGR01146",
    "TIGR01164",
    "TIGR01169",
    "TIGR01171",
    "TIGR01302",
    "TIGR01391",
    "TIGR01393",
    "TIGR01394",
    "TIGR01510",
    "TIGR01632",
    "TIGR01951",
    "TIGR01953",
    "TIGR02012",
    "TIGR02013",
    "TIGR02027",
    "TIGR02075",
    "TIGR02191",
    "TIGR02273",
    "TIGR02350",
    "TIGR02386",
    "TIGR02397",
    "TIGR02432",
    "TIGR02729",
    "TIGR03263",
    "TIGR03594",
    "TIGR03625",
    "TIGR03632",
    "TIGR03654",
    "TIGR03723",
    "TIGR03725",
    "TIGR03953",
]

AR53_MARKERS = [
    "PF04919.13",
    "PF07541.13",
    "PF01000.27",
    "PF00687.22",
    "PF00466.21",
    "PF00827.18",
    "PF01280.21",
    "PF01090.20",
    "PF01200.19",
    "PF01015.19",
    "PF00900.21",
    "PF00410.20",
    "TIGR00037",
    "TIGR00064",
    "TIGR00111",
    "TIGR00134",
    "TIGR00279",
    "TIGR00291",
    "TIGR00323",
    "TIGR00335",
    "TIGR00373",
    "TIGR00405",
    "TIGR00448",
    "TIGR00483",
    "TIGR00491",
    "TIGR00522",
    "TIGR00967",
    "TIGR00982",
    "TIGR01008",
    "TIGR01012",
    "TIGR01018",
    "TIGR01020",
    "TIGR01028",
    "TIGR01046",
    "TIGR01052",
    "TIGR01171",
    "TIGR01213",
    "TIGR01952",
    "TIGR02236",
    "TIGR02338",
    "TIGR02389",
    "TIGR02390",
    "TIGR03626",
    "TIGR03627",
    "TIGR03628",
    "TIGR03629",
    "TIGR03670",
    "TIGR03671",
    "TIGR03672",
    "TIGR03673",
    "TIGR03674",
    "TIGR03676",
    "TIGR03680",
]


def read_fasta(path: str) -> Dict[str, str]:
    """
    Read a FASTA file into a dictionary of sequences.

    Parameters
    ----------
    path : str
        Filesystem path to the input FASTA file.

    Returns
    -------
    Dict[str, str]
        Mapping from sequence ID (the first token after '>' in the header)
        to the full sequence string, with any terminal “*” characters stripped.

    Raises
    ------
    IOError
        If the file cannot be opened for reading.
    """
    seqs: Dict[str, str] = {}
    with open(path) as fh:
        header, buffer = None, []
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    seqs[header] = "".join(buffer).strip("*")
                header = line[1:].split()[0]
                buffer = []
            else:
                buffer.append(line)
        if header:
            seqs[header] = "".join(buffer).strip("*")
    return seqs


def run_prodigal(
    genome_id: str,
    fasta_path: str,
    out_dir: str,
    force: bool
) -> str:
    """
    Run Prodigal to predict protein-coding genes from a FASTA file.
    Parameters
    ----------
    genome_id : str
        Unique identifier for the genome (used for output directory and file names).
    fasta_path : str
        Path to the input FASTA file containing genomic sequences.
    out_dir : str
        Directory where the output protein FASTA file will be saved.
    force : bool
        If True, overwrite existing output files.
    progress_dict : Dict
        Shared dictionary to track progress across multiple processes.
    task_id : int
        Unique task ID for this genome, used to update progress in the shared dict.
    Returns
    -------
    str
        Path to the output protein FASTA file generated by Prodigal.

    """
    prot_dir = os.path.join(out_dir, genome_id)
    os.makedirs(prot_dir, exist_ok=True)
    prot_fa = os.path.join(prot_dir, f"{genome_id}.faa")

    if force and os.path.exists(prot_fa):
        os.remove(prot_fa)

    # If input is gzipped, decompress to a temporary file
    if fasta_path.endswith(".gz"):
        with gzip.open(fasta_path, "rt") as gz_in, \
             tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".fasta") as tmp_fa:
            shutil.copyfileobj(gz_in, tmp_fa)
            input_path = tmp_fa.name
    else:
        input_path = fasta_path

    # Run Prodigal
    subprocess.run(
        ["prodigal", "-a", prot_fa, "-p", "meta", "-i", input_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return prot_fa


def parse_domtblout_top_hits(domtbl_path: str) -> Dict[str, List[str]]:
    """
    Parse a HMMER --domtblout file and select the top hit per sequence.

    Implements the GTDB-Tk comparator logic: for each query sequence,
    keeps the hit with highest bitscore; ties broken by lower e-value,
    then by lexicographically smaller HMM ID.

    Parameters
    ----------
    domtbl_path : str
        Path to the HMMER --domtblout output file.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from HMM ID to a list of sequence IDs that were chosen
        as top hit(s) for that HMM.

    Raises
    ------
    IOError
        If the domtblout file cannot be opened.
    ValueError
        If a non-numeric e-value or bitscore is encountered.
    """
    seq_matches: Dict[str, Tuple[str, float, float]] = {}
    with open(domtbl_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            seq_id = parts[0]
            hmm_id = parts[3]
            evalue = float(parts[4])
            bitscore = float(parts[5])
            # look for the best hit for this sequence
            prev = seq_matches.get(seq_id)
            if prev is None:
                seq_matches[seq_id] = (hmm_id, bitscore, evalue)
            else:
                # only keep the best hit
                prev_hmm_id, prev_b, prev_e = prev
                if (
                    bitscore > prev_b
                    or (bitscore == prev_b and evalue < prev_e)
                    or (
                        bitscore == prev_b and evalue == prev_e and hmm_id < prev_hmm_id
                    )
                ):
                    seq_matches[seq_id] = (hmm_id, bitscore, evalue)

    # now, invert the mapping to get HMM IDs to sequences
    hits: Dict[str, List[str]] = {}
    for seq_id, (hmm_id, _, _) in seq_matches.items():
        hits.setdefault(hmm_id, []).append(seq_id)
    return hits


def _process_single_genome(
    args: Tuple[
        str,  # genome_id
        str,  # fasta_path
        str,  # out_dir
        int,  # cpus_per_proc
        str,  # pfam_db
        str,  # tigr_db
        bool,  # force
        bool,  # skip_multiple_hits
        int,  # number_of_hits_to_keep
        Dict,  # shared progress dict (Manager().dict())
        int,  # task_id in Rich.Progress
    ],
) -> Tuple[str, Dict[str, List[str]]]:
    """
    Worker-function for one genome, updated to report progress after each subtask.
    Subtasks:
      1) Prodigal
      2) Pfam HMM search
      3) TIGRFAM HMM search
      4) Parse + write FASTAs

    Returns: ( "path/to/genome.fasta", { "bac120": [...], "ar53": [...] } )
    """
    (
        gid,
        path,
        out_dir,
        cpus_per_proc,
        pfam_db,
        tigr_db,
        force,
        skip_multiple_hits,
        number_of_hits_to_keep,
        progress_dict,
        task_id,
    ) = args

    genome_fastas: Dict[str, List[str]] = {"bac120": [], "ar53": []}

    # Signal that we are starting the job
    progress_dict[task_id] = {"progress": 0}

    # Prodigal
    prot_fa = run_prodigal(gid, path, out_dir, force)
    prot_seqs = read_fasta(prot_fa)
    # Signal to the shared dict that we've completed step 1 (Prodigal)
    progress_dict[task_id] = {"progress": 1}

    # Pfam HMM search
    pf_out = os.path.join(out_dir, gid, "pfam.tblout")
    if force and os.path.exists(pf_out):
        os.remove(pf_out)

    subprocess.run(
        [
            "hmmsearch",
            "--cpu",
            str(cpus_per_proc),
            "--notextw",
            "-E",
            "0.001",
            "--domE",
            "0.001",
            "--tblout",
            pf_out,
            pfam_db,
            prot_fa,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    progress_dict[task_id] = {"progress": 2}

    # TIGRFAM HMM search
    tg_out = os.path.join(out_dir, gid, "tigrfam.tblout")
    if force and os.path.exists(tg_out):
        os.remove(tg_out)

    subprocess.run(
        [
            "hmmsearch",
            "--cpu",
            str(cpus_per_proc),
            "--noali",
            "--notextw",
            "--cut_nc",
            "--tblout",
            tg_out,
            tigr_db,
            prot_fa,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    progress_dict[task_id] = {"progress": 3}

    # Write FASTAs for each marker
    pf_hits = parse_domtblout_top_hits(pf_out)
    tg_hits = parse_domtblout_top_hits(tg_out)
    combined_hits = {**pf_hits, **tg_hits}

    for marker, seq_ids in combined_hits.items():
        if not seq_ids:
            continue
        elif len(seq_ids) == 1:
            seqs = [prot_seqs[seq_ids[0]]]
        else:
            unique_seqs = set(prot_seqs[s] for s in seq_ids)
            if len(unique_seqs) != 1 and skip_multiple_hits:
                # faster but can miss some markers
                continue
            seqs = list(unique_seqs)[:number_of_hits_to_keep]

        for dom in ("bac120", "ar53"):
            if (dom == "bac120" and marker in BAC120_MARKERS) or (
                dom == "ar53" and marker in AR53_MARKERS
            ):
                genome_dir = os.path.join(out_dir, gid, dom)
                os.makedirs(genome_dir, exist_ok=True)
                for i, seq in enumerate(seqs, start=1):
                    fa_path = os.path.join(genome_dir, f"{marker}.{i}.fa")
                    with open(fa_path, "w") as fh:
                        fh.write(f">{gid}\n{seq}\n")
                    genome_fastas[dom].append(fa_path)

    # Signal completion of subtask 4 (and thus the entire genome job)
    progress_dict[task_id] = {"progress": 4}
    return (path, genome_fastas)


def extract_markers_genes(
    genomes: Dict[str, str],
    out_dir: str,
    cpus: int = 1,
    pfam_db: str = os.environ.get("PFAM_HMMDB"),
    tigr_db: str = os.environ.get("TIGR_HMMDB"),
    force: bool = False,
    skip_multiple_hits: bool = False,
    number_of_hits_to_keep: int = 1,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract marker genes from multiple genomes in parallel using Prodigal and HMMER. 
    """

    # Early exit if no genomes
    if not genomes:
        return {}

    n_genomes = len(genomes)
    n_workers = min(n_genomes, cpus)
    cpus_per_proc = max(1, cpus // n_workers)

    # This shared dict allows us to track progress 
    # across multiple processes
    manager = Manager()
    progress_dict = manager.dict()

    genome_to_task: Dict[str, int] = {}
    futures_to_task: Dict = {}

    results: Dict[str, Dict[str, List[str]]] = {}

    with progress.Progress(
        progress.TextColumn("[progress.description]{task.description}", justify="right"),
        progress.BarColumn(),
        progress.TaskProgressColumn(),
        progress.TimeElapsedColumn(),
    ) as rich_progress:

        # Over progress bar that counts how many genomes finished all 4 steps
        overall_task = rich_progress.add_task("[cyan]Extracting Markers", total=n_genomes)

        # Create one sub‐task per genome (each with total=4)
        for idx, (gid, fasta_path) in enumerate(genomes.items()):
            # 4 total steps per genome: Prodigal, Pfam, TIGRFAM, and writing FASTAs
            task_id = rich_progress.add_task(gid, total=4, visible=False, start=False)
            genome_to_task[gid] = task_id
            progress_dict[task_id] = {}

        # Submit all genome jobs to a ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for gid, fasta_path in genomes.items():
                task_id = genome_to_task[gid]
                args = (
                    gid,
                    fasta_path,
                    out_dir,
                    cpus_per_proc,
                    pfam_db,
                    tigr_db,
                    force,
                    skip_multiple_hits,
                    number_of_hits_to_keep,
                    progress_dict,
                    task_id,
                )
                future = executor.submit(_process_single_genome, args)
                futures_to_task[future] = (task_id, fasta_path)

            started_tasks = set()  # Track which tasks have started
            # Monitor progress until all futures complete
            while futures_to_task:
                for task_id, status in progress_dict.items():
                    started = "progress" in status # only started tasks have a "progress" key
                    if started and task_id not in started_tasks:
                        # If the task has started or progressed, ensure the progress bar is visible
                        rich_progress.update(task_id, visible=True)
                        rich_progress.start_task(task_id)
                        started_tasks.add(task_id)
                    elif started:
                    # Update the individual genome progress bar
                        rich_progress.update(task_id, completed=status["progress"])

                # Check for any finished futures
                done_now = []
                for fut, (task_id, fasta_path) in futures_to_task.items():
                    if fut.done():
                        done_now.append(fut)
                        rich_progress.update(task_id, visible=False) # Hide the task bar

                for fut in done_now:
                    task_id, fasta_path = futures_to_task.pop(fut)
                    try:
                        path_key, genome_dict = fut.result()
                    except Exception as e:
                        # If a worker failed, re‐raise
                        # TODO: consider logging the error instead
                        raise RuntimeError(f"Error processing {fasta_path}: {e}") from e

                    # Store the results
                    results[path_key] = genome_dict

                    status = progress_dict.get(task_id, {})
                    if status.get("progress", 0) >= 4:
                        # advance overall by one
                        rich_progress.update(overall_task, advance=1)

                # Small sleep to avoid busy‐waiting too tightly
                import time
                time.sleep(0.2)
            # Ensure we update the overall task to the final count
            rich_progress.update(overall_task, completed=n_genomes)

    return results

