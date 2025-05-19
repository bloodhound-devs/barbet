import os
from pathlib import Path

from barbet.markers import (
    run_prodigal,
    parse_domtblout_top_hits,
    extract_single_copy_markers
)


def test_parse_domtblout_top_hits(tmp_path):
    # (keep this purely unit‐level parser test unchanged)
    data = [
        "# comment line",
        "seq1 x y markerA 1e-5 50.0",
        "seq2 x y markerA 1e-6 100.0",
        "seq3 x y markerB 1e-3 20.0"
    ]
    f = tmp_path / "hits.domtblout"
    f.write_text("\n".join(data))

    top = parse_domtblout_top_hits(str(f))
    assert set(top.keys()) == {"markerA", "markerB"}
    assert top["markerA"] == ["seq1", "seq2"]
    assert top["markerB"] == ["seq3"]


def test_run_prodigal_integration(tmp_path):
    """
    Run the real prodigal on the test genome and check
    that proteins are predicted.
    """
    data_dir = Path(__file__).parent / "data"
    genome_fa = data_dir / "MAG-GUT41.fa.gz"
    assert genome_fa.exists(), f"Missing test genome at {genome_fa}"
    
    out_dir = tmp_path / "prodigal_out"
    out_dir.mkdir()

    prot_fa = run_prodigal(
        genome_id=genome_fa.stem,
        fasta_path=str(genome_fa),
        out_dir=str(out_dir),
        force=True
    )

    # Check the .faa was created and looks like a FASTA
    assert os.path.isfile(prot_fa)
    content = open(prot_fa).read().strip()
    assert content.startswith(">"), "Expected FASTA header in protein output"
    # At least one protein predicted
    assert "\n>" in content, "Expected multiple proteins (or at least one newline+>)"


def test_identify_single_copy_fasta_integration(tmp_path, monkeypatch):
    """
    Run the full marker‐gene identification against the mock data:
    - unpack the marker‐gene reps
    - set PFAM_HMMDB / TIGRFAM_HMMDB
    - call identify_single_copy_fasta
    - verify we got .fa files under each genome’s bac120 and ar53 dirs
    """
    data_dir = Path(__file__).parent / "data"
    # point at the real HMM DBs
    pfam_db = data_dir / "markers" / "pfam" / "Pfam-A.hmm"
    tigr_db = data_dir / "markers" / "tigrfam" / "tigrfam.hmm"
    assert pfam_db.exists(), f"Missing PFAM DB at {pfam_db}"
    assert tigr_db.exists(), f"Missing TIGRFAM DB at {tigr_db}"


    # run on the real test genome
    genome_fa = data_dir / "MAG-GUT41.fa.gz"
    assert genome_fa.exists()

    out = extract_single_copy_markers(
        genomes={genome_fa.stem: str(genome_fa)},
        out_dir=str(tmp_path),
        cpus=2,
        pfam_db=str(pfam_db),
        tigr_db=str(tigr_db),
        force=True
    )

    # should get one key per genome
    stem = genome_fa.stem
    assert stem in out, f"Expected genome '{stem}' in output"

    # now for each domain under that genome, we must have some FASTAs
    domains = out[stem]
    for domain in ("bac120", "ar53"):
        assert domain in domains, f"Missing domain key '{domain}' in result for genome {stem}"
        fa_list = domains[domain]
        assert isinstance(fa_list, list), f"Expected a list for {domain}"
        assert len(fa_list) > 0, f"No FASTA files generated for {domain}"
        for fa_path in fa_list:
            assert os.path.isfile(fa_path), f"Missing FASTA file: {fa_path}"
            text = open(fa_path).read().strip()
            assert text.startswith(">"), f"Expected FASTA header in {fa_path}"

