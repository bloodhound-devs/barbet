import os
import subprocess
from typing import Dict, List, Tuple
import gzip
import shutil
import tempfile
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

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
        Mapping from sequence ID (the first token after ‘>’ in the header)
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
    genome_id: str, fasta_path: str, out_dir: str, force: bool = False
) -> str:
    """
    Run Prodigal to predict protein translations from a genome FASTA.

    Parameters
    ----------
    genome_id : str
        Identifier for this genome; used to name output files.
    fasta_path : str
        Path to the input genome FASTA file.
    out_dir : str
        Directory under which a subdirectory named `genome_id` will be created
        (if necessary) to hold the predicted proteins.
    force : bool, optional
        If True and the output file already exists, it will be removed and
        regenerated. Default is False.

    Returns
    -------
    str
        Path to the predicted protein FASTA (`.faa`) file.

    Raises
    ------
    RuntimeError
        If the `prodigal` executable cannot be found in `PATH`.
    subprocess.CalledProcessError
        If Prodigal exits with a non-zero status.
    """
    prot_dir = os.path.join(out_dir, genome_id)
    os.makedirs(prot_dir, exist_ok=True)
    prot_fa = os.path.join(prot_dir, f"{genome_id}.faa")

    if force and os.path.exists(prot_fa):
        os.remove(prot_fa)

    # check if gzipped
    if fasta_path.endswith(".gz"):
        with gzip.open(fasta_path, "rt") as gz_in, tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".fasta"
        ) as tmp_fa:
            shutil.copyfileobj(gz_in, tmp_fa)
            tmp_fa_path = tmp_fa.name
        input_path = tmp_fa_path
    else:
        input_path = fasta_path
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


def extract_single_copy_markers(
    genomes: Dict[str, str],
    out_dir: str,
    cpus: int = 1,
    pfam_db: str = os.environ.get("PFAM_HMMDB"),
    tigr_db: str = os.environ.get("TIGR_HMMDB"),
    force: bool = False,
) -> Dict[str, Dict[str, List[str]]]:

    out_fastas = { gid: {"bac120": [], "ar53": []} for gid in genomes }
    total = len(genomes)

    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_pred  = progress.add_task("Prodigal",      total=total)
        task_pfam  = progress.add_task("Pfam HMMs",     total=total)
        task_tigr  = progress.add_task("TIGRFAM HMMs",  total=total)
        task_fasta = progress.add_task("Writing FASTAs", total=total)

        for gid, path in genomes.items():
            progress.console.print(f"Processing genome {gid} from {path}")
            # 1) Prodigal
            prot_fa   = run_prodigal(gid, path, out_dir, force=force)
            prot_seqs = read_fasta(prot_fa)
            progress.update(task_pred, advance=1)

            # 2) Pfam search
            pf_out = os.path.join(out_dir, gid, "pfam.tblout")
            if force and os.path.exists(pf_out): os.remove(pf_out)
            subprocess.run(
                ["hmmsearch","--cpu",str(cpus),
                 "--notextw","-E","0.001","--domE","0.001",
                 "--tblout",pf_out, pfam_db, prot_fa],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            progress.update(task_pfam, advance=1)

            # 3) TIGRFAM search
            tg_out = os.path.join(out_dir, gid, "tigrfam.tblout")
            if force and os.path.exists(tg_out): os.remove(tg_out)
            subprocess.run(
                ["hmmsearch","--cpu",str(cpus),
                 "--noali","--notextw","--cut_nc",
                 "--tblout",tg_out, tigr_db, prot_fa],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            progress.update(task_tigr, advance=1)

            # 4) Parse & merge
            hits = parse_domtblout_top_hits(pf_out) | parse_domtblout_top_hits(tg_out)
            
            # 5) Write out the sequences
            for marker, recs in hits.items():
                if len(recs) == 0:
                    # no hits
                    continue
                elif len(recs) == 1:
                    # unique hit
                    seq = prot_seqs[recs[0]]
                else:
                    # multiple hits
                    # check if they’re all identical
                    seqs = set(prot_seqs[s] for s in recs)
                    if len(seqs) != 1:
                        # not identical, so skip
                        continue
                    # identical, so count it as a unique hit
                    seq = seqs[0] 

                for dom in ("bac120", "ar53"):
                    # write out the sequences to separate directories for each domain 
                    if (dom == "bac120" and marker in BAC120_MARKERS) or \
                       (dom == "ar53"   and marker in AR53_MARKERS):
                        genome_dir = os.path.join(out_dir, gid, dom)
                        os.makedirs(genome_dir, exist_ok=True)
                        fa_path = os.path.join(genome_dir, f"{marker}.fa")
                        with open(fa_path, "w") as fh:
                            fh.write(f">{gid}\n{seq}\n")
                        out_fastas[gid][dom].append(fa_path)
                
            progress.update(task_fasta, advance=1)

    return out_fastas
