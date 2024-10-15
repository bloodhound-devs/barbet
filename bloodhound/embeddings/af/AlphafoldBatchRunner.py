from __future__ import annotations

import os
ENV = {"TF_FORCE_UNIFIED_MEMORY":"1", "XLA_PYTHON_CLIENT_MEM_FRACTION":"4.0"}
for k,v in ENV.items():
    if k not in os.environ: os.environ[k] = v

import warnings
import json
import logging
import math
import random
import sys
import time
import zipfile
import shutil
import pickle
import gzip
import pandas

import numpy as np
np.int = int

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from io import StringIO

from Bio import BiopythonDeprecationWarning # what can possibly go wrong...
warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import alphafold
from alphafold.common import protein, residue_constants

# delay imports of tensorflow, jax and numpy
# loading these for type checking only can take around 10 seconds just to show a CLI usage message
if TYPE_CHECKING:
    import haiku
    from alphafold.model import model
    from numpy import ndarray

import alphafold.common.protein as af_protein
from alphafold.data import feature_processing,msa_pairing,pipeline,pipeline_multimer,templates



from bloodhound.embeddings.af.models import load_models_and_params
from bloodhound.embeddings.af.cf_utils import make_fixed_size,ACCEPT_DEFAULT_TERMS,DEFAULT_API_SERVER,NO_GPU_FOUND,CIF_REVISION_DATE,safe_filename,setup_logging,CFMMCIFIO
from bloodhound.embeddings.af.mmseq2 import run_mmseqs2


from Bio.PDB import MMCIFParser, PDBParser, MMCIF2Dict
from Bio.PDB.PDBIO import Select

# logging settings
logger = logging.getLogger(__name__)
import jax
import jax.numpy as jnp

# from jax 0.4.6, jax._src.lib.xla_bridge moved to jax._src.xla_bridge
# suppress warnings: Unable to initialize backend 'rocm' or 'tpu'
logging.getLogger('jax._src.xla_bridge').addFilter(lambda _: False) # jax >=0.4.6
logging.getLogger('jax._src.lib.xla_bridge').addFilter(lambda _: False) # jax < 0.4.5

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

# Gabry adding logger to stdout and to file:
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)


# backward-compatibility with old options
OLD_AF_NAMES = {
    "AlphaFold2-multimer-v1":"alphafold2_multimer_v1",
    "AlphaFold2-multimer-v2":"alphafold2_multimer_v2",
    "AlphaFold2-multimer-v3":"alphafold2_multimer_v3",
    "AlphaFold2-ptm":        "alphafold2_ptm",
    "AlphaFold2":            "alphafold2",
    "DeepFold":              "deepfold_v1",
}

# backward-compatibility with old options
OLD_MMSEQ_NAMES = {"MMseqs2 (UniRef+Environmental)":"mmseqs2_uniref_env",
            "MMseqs2 (UniRef+Environmental+Env. Pairing)":"mmseqs2_uniref_env_envpair",
            "MMseqs2 (UniRef only)":"mmseqs2_uniref",
            "unpaired+paired":"unpaired_paired"}

modified_mapping = {
  "MSE" : "MET", "MLY" : "LYS", "FME" : "MET", "HYP" : "PRO",
  "TPO" : "THR", "CSO" : "CYS", "SEP" : "SER", "M3L" : "LYS",
  "HSK" : "HIS", "SAC" : "SER", "PCA" : "GLU", "DAL" : "ALA",
  "CME" : "CYS", "CSD" : "CYS", "OCS" : "CYS", "DPR" : "PRO",
  "B3K" : "LYS", "ALY" : "LYS", "YCM" : "CYS", "MLZ" : "LYS",
  "4BF" : "TYR", "KCX" : "LYS", "B3E" : "GLU", "B3D" : "ASP",
  "HZP" : "PRO", "CSX" : "CYS", "BAL" : "ALA", "HIC" : "HIS",
  "DBZ" : "ALA", "DCY" : "CYS", "DVA" : "VAL", "NLE" : "LEU",
  "SMC" : "CYS", "AGM" : "ARG", "B3A" : "ALA", "DAS" : "ASP",
  "DLY" : "LYS", "DSN" : "SER", "DTH" : "THR", "GL3" : "GLY",
  "HY3" : "PRO", "LLP" : "LYS", "MGN" : "GLN", "MHS" : "HIS",
  "TRQ" : "TRP", "B3Y" : "TYR", "PHI" : "PHE", "PTR" : "TYR",
  "TYS" : "TYR", "IAS" : "ASP", "GPL" : "LYS", "KYN" : "TRP",
  "CSD" : "CYS", "SEC" : "CYS"
}

class ReplaceOrRemoveHetatmSelect(Select):
  def accept_residue(self, residue):
    hetfield, _, _ = residue.get_id()
    if hetfield != " ":
      if residue.resname in modified_mapping:
        # set unmodified resname
        residue.resname = modified_mapping[residue.resname]
        # clear hetatm flag
        residue._id = (" ", residue._id[1], " ")
        t = residue.full_id
        residue.full_id = (t[0], t[1], t[2], residue._id)
        return 1
      return 0
    else:
      return 1


class FileManager:
    def __init__(self, prefix: str, result_dir: Path):
        self.prefix = prefix
        self.result_dir = result_dir
        self.tag = None
        self.files = {}

    def get(self, x: str, ext:str) -> Path:
        if self.tag not in self.files:
            self.files[self.tag] = []
        file = self.result_dir.joinpath(f"{self.prefix}_{x}_{self.tag}.{ext}")
        self.files[self.tag].append([x,ext,file])
        return file

    def set_tag(self, tag):
        self.tag = tag

class AlphafoldBatchRunner:
    DEVICE = "gpu"
    def __init__(
        self,        
        queries: List[Tuple[str, Union[str, List[str]], Optional[List[str]]]],
        result_dir: Union[str, Path],
        num_models: int,
        is_complex: bool,
        num_recycles: Optional[int] = None,
        recycle_early_stop_tolerance: Optional[float] = None,
        model_order: List[int] = [1,2,3,4,5],
        num_ensemble: int = 1,
        model_type: str = "auto",
        msa_mode: str = "mmseqs2_uniref_env",
        use_templates: bool = False,
        custom_template_path: str = None,
        num_relax: int = 0,
        relax_max_iterations: int = 0,
        relax_tolerance: float = 2.39,
        relax_stiffness: float = 10.0,
        relax_max_outer_iterations: int = 3,
        keep_existing_results: bool = True,
        rank_by: str = "auto",
        pair_mode: str = "unpaired_paired",
        pairing_strategy: str = "greedy",
        data_dir: Union[str, Path] = "", # We always use our own custom data directory
        host_url: str = DEFAULT_API_SERVER,
        user_agent: str = "",
        random_seed: int = 0,
        num_seeds: int = 1,
        recompile_padding: Union[int, float] = 10,
        zip_results: bool = False,
        prediction_callback: Callable[[Any, Any, Any, Any, Any], Any] = None,
        save_single_representations: bool = False,
        save_pair_representations: bool = False,
        jobname_prefix: Optional[str] = None,
        save_all: bool = False,
        save_recycles: bool = False,
        use_dropout: bool = False,
        use_gpu_relax: bool = False,
        stop_at_score: float = 100,
        dpi: int = 200,
        max_seq: Optional[int] = None,
        max_extra_seq: Optional[int] = None,
        pdb_hit_file: Optional[Path] = None,
        local_pdb_path: Optional[Path] = None,
        use_cluster_profile: bool = True,
        feature_dict_callback: Callable[[Any], Any] = None,
        **kwargs
    ):
        self.output = {'log': '', 'embeddings': {}}
        self.queries = queries
        self.result_dir = result_dir
        self.num_models = num_models
        self.is_complex = is_complex
        self.num_recycles = num_recycles
        self.recycle_early_stop_tolerance = recycle_early_stop_tolerance
        self.model_order = model_order
        self.num_ensemble = num_ensemble
        self.model_type = model_type
        self.msa_mode = msa_mode
        self.use_templates = use_templates
        self.custom_template_path = custom_template_path
        self.num_relax = num_relax
        self.relax_max_iterations = relax_max_iterations
        self.relax_tolerance = relax_tolerance
        self.relax_stiffness = relax_stiffness
        self.relax_max_outer_iterations = relax_max_outer_iterations
        self.keep_existing_results = keep_existing_results
        self.rank_by = rank_by
        self.pair_mode = pair_mode
        self.pairing_strategy = pairing_strategy
        self.data_dir = data_dir
        self.host_url = host_url
        self.user_agent = user_agent
        self.random_seed = random_seed
        self.num_seeds = num_seeds
        self.recompile_padding = recompile_padding
        self.zip_results = zip_results
        self.prediction_callback = prediction_callback
        self.save_single_representations = save_single_representations
        self.save_pair_representations = save_pair_representations
        self.jobname_prefix = jobname_prefix
        self.save_all = save_all
        self.save_recycles = save_recycles
        self.use_dropout = use_dropout
        self.use_gpu_relax = use_gpu_relax
        self.stop_at_score = stop_at_score
        self.dpi = dpi
        self.max_seq = max_seq
        self.max_extra_seq = max_extra_seq
        self.pdb_hit_file = pdb_hit_file
        self.local_pdb_path = local_pdb_path
        self.use_cluster_profile = use_cluster_profile
        self.feature_dict_callback = feature_dict_callback
        self.kwargs = kwargs
        
        
        self.feature_dict_callback = self.kwargs.pop("input_features_callback", self.feature_dict_callback)
        self.use_dropout           = self.kwargs.pop("training", self.use_dropout)
        self.use_fuse              = self.kwargs.pop("use_fuse", True)
        self.use_bfloat16          = self.kwargs.pop("use_bfloat16", True)
        self.max_msa               = self.kwargs.pop("max_msa",None)
        if self.max_msa is not None:
            self.max_seq, self.max_extra_seq = [int(x) for x in self.max_msa.split(":")]
        if self.kwargs.pop("use_amber", False) and self.num_relax == 0:
            self.num_relax = self.num_models * self.num_seeds
        if len(self.kwargs) > 0:
            print(f"WARNING: the following options are not being used: {kwargs}")
        
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_env = "env" in msa_mode
        self.use_msa = "mmseqs2" in msa_mode
        self.use_amber = num_models > 0 and num_relax > 0

        self.check_available_devices()
        self.config_setup() 
        self.setup_pdb()
        
        if self.custom_template_path is not None:
            self.mk_hhsearch_db()
        
        self.first_job = True
        
    def log_out(self, txt="\n"):
        self.output['log'] += f"\n{txt}"
        logger.info(txt)

        
    def check_available_devices(self):     
        # check what device is available
        try:
            # check if TPU is available
            import jax.tools.colab_tpu
            jax.tools.colab_tpu.setup_tpu()
            self.log_out('Running on TPU')
            DEVICE = "tpu"
            use_gpu_relax = False
        except:
            if jax.local_devices()[0].platform == 'cpu':
                self.log_out("WARNING: no GPU detected, will be using CPU")
                DEVICE = "cpu"
                use_gpu_relax = False
            else:
                import tensorflow as tf
                tf.get_logger().setLevel(logging.ERROR)
                self.log_out('Running on GPU')
                self.output['log'] += f"\n'Running on GPU'"
                DEVICE = "gpu"
                # disable GPU on tensorflow
                tf.config.set_visible_devices([], 'GPU')

                
    def config_setup(self):
        self.model_type = OLD_AF_NAMES.get(self.model_type, self.model_type)
        if self.model_type == "auto":
            self.model_type = "alphafold2_multimer_v3" if self.is_complex else "alphafold2_ptm"
        self.msa_mode   = OLD_MMSEQ_NAMES.get(self.msa_mode, self.msa_mode)
        self.pair_mode  = OLD_MMSEQ_NAMES.get(self.pair_mode, self.pair_mode)

        # decide how to rank outputs
        if self.rank_by == "auto":
            self.rank_by = "multimer" if self.is_complex else "plddt"
        if "ptm" not in self.model_type and "multimer" not in self.model_type:
            self.rank_by = "plddt"

        # get max length
        self.max_len = 0
        self.max_num = 0
        for _, query_sequence, _ in self.queries:
            N = 1 if isinstance(query_sequence,str) else len(query_sequence)
            L = len("".join(query_sequence))
            if L > self.max_len: self.max_len = L
            if N > self.max_num: self.max_num = N

        # get max sequences
        # 512 5120 = alphafold_ptm (models 1,3,4)
        # 512 1024 = alphafold_ptm (models 2,5)
        # 508 2048 = alphafold-multimer_v3 (models 1,2,3)
        # 508 1152 = alphafold-multimer_v3 (models 4,5)
        # 252 1152 = alphafold-multimer_v[1,2]

        set_if = lambda x,y: y if x is None else x
        if self.model_type in ["alphafold2_multimer_v1","alphafold2_multimer_v2"]:
            (self.max_seq, self.max_extra_seq) = (set_if(self.max_seq,252), set_if(self.max_extra_seq,1152))
        elif self.model_type == "alphafold2_multimer_v3":
            (self.max_seq, self.max_extra_seq) = (set_if(self.max_seq,508), set_if(self.max_extra_seq,2048))
        else:
            (self.max_seq, self.max_extra_seq) = (set_if(self.max_seq,512), set_if(self.max_extra_seq,5120))

        if self.msa_mode == "single_sequence":
            num_seqs = 1
            if self.is_complex and "multimer" not in self.model_type: num_seqs += self.max_num
            if self.use_templates: num_seqs += 4
            self.max_seq = min(num_seqs, self.max_seq)
            self.max_extra_seq = max(min(num_seqs - self.max_seq, self.max_extra_seq), 1)
        
         # Record the parameters of this run
        self.config = {
            "num_queries": len(self.queries),
            "use_templates": self.use_templates,
            "num_relax": self.num_relax,
            "relax_max_iterations": self.relax_max_iterations,
            "relax_tolerance": self.relax_tolerance,
            "relax_stiffness": self.relax_stiffness,
            "relax_max_outer_iterations": self.relax_max_outer_iterations,
            "msa_mode": self.msa_mode,
            "model_type": self.model_type,
            "num_models": self.num_models,
            "num_recycles": self.num_recycles,
            "recycle_early_stop_tolerance": self.recycle_early_stop_tolerance,
            "num_ensemble": self.num_ensemble,
            "model_order": self.model_order,
            "keep_existing_results": self.keep_existing_results,
            "rank_by": self.rank_by,
            "max_seq": self.max_seq,
            "max_extra_seq": self.max_extra_seq,
            "pair_mode": self.pair_mode,
            "pairing_strategy": self.pairing_strategy,
            "host_url": self.host_url,
            "user_agent": self.user_agent,
            "stop_at_score": self.stop_at_score,
            "random_seed": self.random_seed,
            "num_seeds": self.num_seeds,
            "recompile_padding": self.recompile_padding,
            "commit": "<bloodhound-commit>",
            "use_dropout": self.use_dropout,
            "use_cluster_profile": self.use_cluster_profile,
            "use_fuse": self.use_fuse,
            "use_bfloat16": self.use_bfloat16,
            "version": "bloodhound-colabfold",
        }
        self.config_out_file = self.result_dir.joinpath("config.json")
        self.config_out_file.write_text(json.dumps(self.config, indent=4))    
        
    def setup_pdb(self):        
        if self.pdb_hit_file is not None:
            if self.local_pdb_path is None:
                raise ValueError("local_pdb_path is not specified.")
            else:
                self.custom_template_path = self.result_dir / "templates"
                self.put_mmciffiles_into_resultdir()   


    def put_mmciffiles_into_resultdir(self, max_num_templates: int = 20,):
        """Put mmcif files from local_pdb_path into result_dir and unzip them.
        max_num_templates is the maximum number of templates to use (default: 20).
        Args:
            pdb_hit_file (Path): Path to pdb_hit_file
            local_pdb_path (Path): Path to local_pdb_path
            result_dir (Path): Path to result_dir
            max_num_templates (int): Maximum number of templates to use
        """
        pdb_hit_file = Path(self.pdb_hit_file)
        local_pdb_path = Path(self.local_pdb_path)
        result_dir = Path(self.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        query_ids = []
        with open(pdb_hit_file, "r") as f:
            for line in f:
                query_id = line.split("\t")[0]
                query_ids.append(query_id)
                if query_ids.count(query_id) > max_num_templates:
                    continue
                else:
                    pdb_id = line.split("\t")[1][0:4]
                    divided_pdb_id = pdb_id[1:3]
                    gzipped_divided_mmcif_file = local_pdb_path / divided_pdb_id / (pdb_id + ".cif.gz")
                    gzipped_mmcif_file = local_pdb_path / (pdb_id + ".cif.gz")
                    unzipped_mmcif_file = local_pdb_path / (pdb_id + ".cif")
                    result_file = result_dir / (pdb_id + ".cif")
                    possible_files = [gzipped_divided_mmcif_file, gzipped_mmcif_file, unzipped_mmcif_file]
                    for file in possible_files:
                        if file == gzipped_divided_mmcif_file or file == gzipped_mmcif_file:
                            if file.exists():
                                with gzip.open(file, "rb") as f_in:
                                    with open(result_file, "wb") as f_out:
                                        shutil.copyfileobj(f_in, f_out)
                                        break
                        else:
                            # unzipped_mmcif_file
                            if file.exists():
                                shutil.copyfile(file, result_file)
                                break
                    if not result_file.exists():
                        print(f"WARNING: {pdb_id} does not exist in {local_pdb_path}.")
                        
    
    def convert_pdb_to_mmcif(self, pdb_file: Path):
        """convert existing pdb files into mmcif with the required poly_seq and revision_date"""
        i = pdb_file.stem
        cif_file = pdb_file.parent.joinpath(f"{i}.cif")
        if cif_file.is_file():
            return
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(i, pdb_file)
        cif_io = CFMMCIFIO()
        cif_io.set_structure(structure)
        cif_io.save(str(cif_file), ReplaceOrRemoveHetatmSelect())
        

    def validate_and_fix_mmcif(self, cif_file: Path):
        """validate presence of _entity_poly_seq in cif file and add revision_date if missing"""
        # check that required poly_seq and revision_date fields are present
        cif_dict = MMCIF2Dict.MMCIF2Dict(cif_file)
        required = [
            "_chem_comp.id",
            "_chem_comp.type",
            "_struct_asym.id",
            "_struct_asym.entity_id",
            "_entity_poly_seq.mon_id",
        ]
        for r in required:
            if r not in cif_dict:
                raise ValueError(f"mmCIF file {cif_file} is missing required field {r}.")
        if "_pdbx_audit_revision_history.revision_date" not in cif_dict:
            self.log_out(
                f"Adding missing field revision_date to {cif_file}. Backing up original file to {cif_file}.bak."
            )
            shutil.copy2(cif_file, str(cif_file) + ".bak")
            with open(cif_file, "a") as f:
                f.write(CIF_REVISION_DATE)
                

    def mk_hhsearch_db(self):
        template_path = Path(self.custom_template_path)

        cif_files = template_path.glob("*.cif")
        for cif_file in cif_files:
            self.validate_and_fix_mmcif(cif_file)

        pdb_files = template_path.glob("*.pdb")
        for pdb_file in pdb_files:
            self.convert_pdb_to_mmcif(pdb_file)

        pdb70_db_files = template_path.glob("pdb70*")
        for f in pdb70_db_files:
            os.remove(f)

        with open(template_path.joinpath("pdb70_a3m.ffdata"), "w") as a3m,\
            open(template_path.joinpath("pdb70_cs219.ffindex"), "w") as cs219_index,\
            open(template_path.joinpath("pdb70_a3m.ffindex"), "w") as a3m_index,\
            open(template_path.joinpath("pdb70_cs219.ffdata"), "w") as cs219:
            n = 1000000
            index_offset = 0
            cif_files = template_path.glob("*.cif")
            for cif_file in cif_files:
                with open(cif_file) as f:
                    cif_string = f.read()
                cif_fh = StringIO(cif_string)
                parser = MMCIFParser(QUIET=True)
                structure = parser.get_structure("none", cif_fh)
                models = list(structure.get_models())
                if len(models) != 1:
                    logger.warning(f"WARNING: Found {len(models)} models in {cif_file}. The first model will be used as a template.", )
                    # raise ValueError(
                    #     f"Only single model PDBs are supported. Found {len(models)} models in {cif_file}."
                    # )
                model = models[0]
                for chain in model:
                    amino_acid_res = []
                    for res in chain:
                        if res.id[2] != " ":
                            logger.warning(f"WARNING: Found insertion code at chain {chain.id} and residue index {res.id[1]} of {cif_file}. "
                                        "This file cannot be used as a template.")
                            continue
                            # raise ValueError(
                            #     f"PDB {cif_file} contains an insertion code at chain {chain.id} and residue "
                            #     f"index {res.id[1]}. These are not supported."
                            # )
                        amino_acid_res.append(
                            residue_constants.restype_3to1.get(res.resname, "X")
                        )

                    protein_str = "".join(amino_acid_res)
                    a3m_str = f">{cif_file.stem}_{chain.id}\n{protein_str}\n\0"
                    a3m_str_len = len(a3m_str)
                    a3m_index.write(f"{n}\t{index_offset}\t{a3m_str_len}\n")
                    cs219_index.write(f"{n}\t{index_offset}\t{len(protein_str)}\n")
                    index_offset += a3m_str_len
                    a3m.write(a3m_str)
                    cs219.write("\n\0")
                    n += 1
                    
                    
    def unserialize_msa(self,
        a3m_lines: List[str],
        query_sequence: Union[List[str], str]
    ) -> Tuple[
        Optional[List[str]],
        Optional[List[str]],
        List[str],
        List[int],
        List[Dict[str, Any]],
    ]:
        a3m_lines = a3m_lines[0].replace("\x00", "").splitlines()
        if not a3m_lines[0].startswith("#") or len(a3m_lines[0][1:].split("\t")) != 2:
            assert isinstance(query_sequence, str)
            return (
                ["\n".join(a3m_lines)],
                None,
                [query_sequence],
                [1],
                [self.mk_mock_template(query_sequence)],
            )

        if len(a3m_lines) < 3:
            raise ValueError(f"Unknown file format a3m")
        tab_sep_entries = a3m_lines[0][1:].split("\t")
        query_seq_len = tab_sep_entries[0].split(",")
        query_seq_len = list(map(int, query_seq_len))
        query_seqs_cardinality = tab_sep_entries[1].split(",")
        query_seqs_cardinality = list(map(int, query_seqs_cardinality))
        is_homooligomer = (True if len(query_seq_len) == 1 and query_seqs_cardinality[0] > 1 else False)
        is_single_protein = (True if len(query_seq_len) == 1 and query_seqs_cardinality[0] == 1 else False)
        query_seqs_unique = []
        prev_query_start = 0
        # we store the a3m with cardinality of 1
        for n, query_len in enumerate(query_seq_len):
            query_seqs_unique.append(a3m_lines[2][prev_query_start : prev_query_start + query_len])
            prev_query_start += query_len
        paired_msa = [""] * len(query_seq_len)
        unpaired_msa = [""] * len(query_seq_len)
        already_in = dict()
        for i in range(1, len(a3m_lines), 2):
            header = a3m_lines[i]
            seq = a3m_lines[i + 1]
            if (header, seq) in already_in:
                continue
            already_in[(header, seq)] = 1
            has_amino_acid = [False] * len(query_seq_len)
            seqs_line = []
            prev_pos = 0
            for n, query_len in enumerate(query_seq_len):
                paired_seq = ""
                curr_seq_len = 0
                for pos in range(prev_pos, len(seq)):
                    if curr_seq_len == query_len:
                        prev_pos = pos
                        break
                    paired_seq += seq[pos]
                    if seq[pos].islower():
                        continue
                    if seq[pos] != "-":
                        has_amino_acid[n] = True
                    curr_seq_len += 1
                seqs_line.append(paired_seq)

            # if sequence is paired add them to output
            if (
                not is_single_protein
                and not is_homooligomer
                and sum(has_amino_acid) > 1 # at least 2 sequences are paired
            ):
                header_no_faster = header.replace(">", "")
                header_no_faster_split = header_no_faster.split("\t")
                for j in range(0, len(seqs_line)):
                    paired_msa[j] += ">" + header_no_faster_split[j] + "\n"
                    paired_msa[j] += seqs_line[j] + "\n"
            else:
                for j, seq in enumerate(seqs_line):
                    if has_amino_acid[j]:
                        unpaired_msa[j] += header + "\n"
                        unpaired_msa[j] += seq + "\n"
        if is_homooligomer:
            # homooligomers
            num = 101
            paired_msa = [""] * query_seqs_cardinality[0]
            for i in range(0, query_seqs_cardinality[0]):
                paired_msa[i] = ">" + str(num + i) + "\n" + query_seqs_unique[0] + "\n"
        if is_single_protein:
            paired_msa = None
        template_features = []
        for query_seq in query_seqs_unique:
            template_feature = self.mk_mock_template(query_seq)
            template_features.append(template_feature)

        return (
            unpaired_msa,
            paired_msa,
            query_seqs_unique,
            query_seqs_cardinality,
            template_features,
        )
        
    
    def pair_sequences(
        self,
        a3m_lines: List[str], query_sequences: List[str], query_cardinality: List[int]
    ) -> str:
        a3m_line_paired = [""] * len(a3m_lines[0].splitlines())
        for n, seq in enumerate(query_sequences):
            lines = a3m_lines[n].splitlines()
            for i, line in enumerate(lines):
                if line.startswith(">"):
                    if n != 0:
                        line = line.replace(">", "\t", 1)
                    a3m_line_paired[i] = a3m_line_paired[i] + line
                else:
                    a3m_line_paired[i] = a3m_line_paired[i] + line * query_cardinality[n]
        return "\n".join(a3m_line_paired)

    def pad_sequences(
        self,
        a3m_lines: List[str], query_sequences: List[str], query_cardinality: List[int]
    ) -> str:
        _blank_seq = [
            ("-" * len(seq))
            for n, seq in enumerate(query_sequences)
            for _ in range(query_cardinality[n])
        ]
        a3m_lines_combined = []
        pos = 0
        for n, seq in enumerate(query_sequences):
            for j in range(0, query_cardinality[n]):
                lines = a3m_lines[n].split("\n")
                for a3m_line in lines:
                    if len(a3m_line) == 0:
                        continue
                    if a3m_line.startswith(">"):
                        a3m_lines_combined.append(a3m_line)
                    else:
                        a3m_lines_combined.append(
                            "".join(_blank_seq[:pos] + [a3m_line] + _blank_seq[pos + 1 :])
                        )
                pos += 1
        return "\n".join(a3m_lines_combined)
        
        
    def get_msa_and_templates(
        self,
        jobname: str,
        query_sequences: Union[str, List[str]],
        a3m_lines: Optional[List[str]],
        msa_mode: str,
    ) -> Tuple[
        Optional[List[str]], Optional[List[str]], List[str], List[int], List[Dict[str, Any]]
    ]:
        use_env = msa_mode == "mmseqs2_uniref_env" or msa_mode == "mmseqs2_uniref_env_envpair"
        use_envpair = msa_mode == "mmseqs2_uniref_env_envpair"
        if isinstance(query_sequences, str): query_sequences = [query_sequences]

        # remove duplicates before searching
        query_seqs_unique = []
        for x in query_sequences:
            if x not in query_seqs_unique:
                query_seqs_unique.append(x)

        # determine how many times is each sequence is used
        query_seqs_cardinality = [0] * len(query_seqs_unique)
        for seq in query_sequences:
            seq_idx = query_seqs_unique.index(seq)
            query_seqs_cardinality[seq_idx] += 1

        # get template features
        template_features = []
        if self.use_templates:
            # Skip template search when custom_template_path is provided
            if self.custom_template_path is not None:
                if msa_mode == "single_sequence":
                    a3m_lines = []
                    num = 101
                    for i, seq in enumerate(query_seqs_unique):
                        a3m_lines.append(f">{num + i}\n{seq}")

                if a3m_lines is None:
                    a3m_lines_mmseqs2 = run_mmseqs2(
                        query_seqs_unique,
                        str(self.result_dir.joinpath(jobname)),
                        use_env,
                        use_templates=False,
                        host_url=self.host_url,
                        user_agent=self.user_agent,
                    )
                else:
                    a3m_lines_mmseqs2 = a3m_lines
                template_paths = {}
                for index in range(0, len(query_seqs_unique)):
                    template_paths[index] = self.custom_template_path
            else:
                a3m_lines_mmseqs2, template_paths = run_mmseqs2(
                    query_seqs_unique,
                    str(self.result_dir.joinpath(jobname)),
                    use_env,
                    use_templates=True,
                    host_url=self.host_url,
                    user_agent=self.user_agent,
                )
            if template_paths is None:
                self.log_out("No template detected")
                for index in range(0, len(query_seqs_unique)):
                    template_feature = self.mk_mock_template(query_seqs_unique[index])
                    template_features.append(template_feature)
            else:
                for index in range(0, len(query_seqs_unique)):
                    if template_paths[index] is not None:
                        template_feature = self.mk_template(
                            a3m_lines_mmseqs2[index],
                            template_paths[index],
                            query_seqs_unique[index],
                        )
                        if len(template_feature["template_domain_names"]) == 0:
                            template_feature = self.mk_mock_template(query_seqs_unique[index])
                            self.log_out(f"Sequence {index} found no templates")
                        else:
                            self.log_out(
                                f"Sequence {index} found templates: {template_feature['template_domain_names'].astype(str).tolist()}"
                            )
                    else:
                        template_feature = self.mk_mock_template(query_seqs_unique[index])
                        self.log_out(f"Sequence {index} found no templates")

                    template_features.append(template_feature)
        else:
            for index in range(0, len(query_seqs_unique)):
                template_feature = self.mk_mock_template(query_seqs_unique[index])
                template_features.append(template_feature)

        if len(query_sequences) == 1:
            pair_mode = "none"

        if pair_mode == "none" or pair_mode == "unpaired" or pair_mode == "unpaired_paired":
            if msa_mode == "single_sequence":
                a3m_lines = []
                num = 101
                for i, seq in enumerate(query_seqs_unique):
                    a3m_lines.append(f">{num + i}\n{seq}")
            else:
                # find normal a3ms
                a3m_lines = run_mmseqs2(
                    query_seqs_unique,
                    str(self.result_dir.joinpath(jobname)),
                    use_env,
                    use_pairing=False,
                    host_url=self.host_url,
                    user_agent=self.user_agent,
                )
        else:
            a3m_lines = None

        if msa_mode != "single_sequence" and (pair_mode == "paired" or pair_mode == "unpaired_paired"):
            # find paired a3m if not a homooligomers
            if len(query_seqs_unique) > 1:
                paired_a3m_lines = run_mmseqs2(
                    query_seqs_unique,
                    str(self.result_dir.joinpath(jobname)),
                    use_envpair,
                    use_pairing=True,
                    pairing_strategy=self.pairing_strategy,
                    host_url=self.host_url,
                    user_agent=self.user_agent,
                )
            else:
                # homooligomers
                num = 101
                paired_a3m_lines = []
                for i in range(0, query_seqs_cardinality[0]):
                    paired_a3m_lines.append(f">{num+i}\n{query_seqs_unique[0]}\n")
        else:
            paired_a3m_lines = None

        return (
            a3m_lines,
            paired_a3m_lines,
            query_seqs_unique,
            query_seqs_cardinality,
            template_features,
        )
        
    
    def msa_to_str(
        self,
        unpaired_msa: List[str],
        paired_msa: List[str],
        query_seqs_unique: List[str],
        query_seqs_cardinality: List[int],
    ) -> str:
        msa = "#" + ",".join(map(str, map(len, query_seqs_unique))) + "\t"
        msa += ",".join(map(str, query_seqs_cardinality)) + "\n"
        # build msa with cardinality of 1, it makes it easier to parse and manipulate
        query_seqs_cardinality = [1 for _ in query_seqs_cardinality]
        msa += self.pair_msa(query_seqs_unique, query_seqs_cardinality, paired_msa, unpaired_msa)
        return msa

    
    def generate_input_feature(
        self,
        query_seqs_unique: List[str],
        query_seqs_cardinality: List[int],
        unpaired_msa: List[str],
        paired_msa: List[str],
        template_features: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:

        input_feature = {}
        domain_names = {}
        if self.is_complex and "multimer" not in self.model_type:
            full_sequence = ""
            Ls = []
            for sequence_index, sequence in enumerate(query_seqs_unique):
                for cardinality in range(0, query_seqs_cardinality[sequence_index]):
                    full_sequence += sequence
                    Ls.append(len(sequence))

            # bugfix
            a3m_lines = f">0\n{full_sequence}\n"
            a3m_lines += self.pair_msa(query_seqs_unique, query_seqs_cardinality, paired_msa, unpaired_msa)

            input_feature = self.build_monomer_feature(full_sequence, a3m_lines, self.mk_mock_template(full_sequence))
            input_feature["residue_index"] = np.concatenate([np.arange(L) for L in Ls])
            input_feature["asym_id"] = np.concatenate([np.full(L,n) for n,L in enumerate(Ls)])
            if any(
                [
                    template != b"none"
                    for i in template_features
                    for template in i["template_domain_names"]
                ]
            ):
                logger.warning(f"{self.model_type} complex does not consider templates. Chose multimer model-type for template support.")

        else:
            features_for_chain = {}
            chain_cnt = 0
            # for each unique sequence
            for sequence_index, sequence in enumerate(query_seqs_unique):
                # get unpaired msa
                if unpaired_msa is None:
                    input_msa = f">{101 + sequence_index}\n{sequence}"
                else:
                    input_msa = unpaired_msa[sequence_index]

                feature_dict = self.build_monomer_feature(sequence, input_msa, template_features[sequence_index])

                # for each copy
                for cardinality in range(0, query_seqs_cardinality[sequence_index]):
                    features_for_chain[af_protein.PDB_CHAIN_IDS[chain_cnt]] = feature_dict
                    chain_cnt += 1

            # Only single-sequence 
            input_feature = features_for_chain[af_protein.PDB_CHAIN_IDS[0]]
            input_feature["asym_id"] = np.zeros(input_feature["aatype"].shape[0],dtype=int)
            domain_names = {
                af_protein.PDB_CHAIN_IDS[0]: [
                    name.decode("UTF-8")
                    for name in input_feature["template_domain_names"]
                    if name != b"none"
                ]
            }
        return (input_feature, domain_names)
    
    

    def mk_mock_template(self, query_sequence: Union[List[str], str], num_temp: int = 1) -> Dict[str, Any]:
        ln = (
            len(query_sequence)
            if isinstance(query_sequence, str)
            else sum(len(s) for s in query_sequence)
        )
        output_templates_sequence = "A" * ln
        output_confidence_scores = np.full(ln, 1.0)

        templates_all_atom_positions = np.zeros(
            (ln, templates.residue_constants.atom_type_num, 3)
        )
        templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
        templates_aatype = templates.residue_constants.sequence_to_onehot(
            output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
        )
        template_features = {
            "template_all_atom_positions": np.tile(
                templates_all_atom_positions[None], [num_temp, 1, 1, 1]
            ),
            "template_all_atom_masks": np.tile(
                templates_all_atom_masks[None], [num_temp, 1, 1]
            ),
            "template_sequence": [f"none".encode()] * num_temp,
            "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
            "template_confidence_scores": np.tile(
                output_confidence_scores[None], [num_temp, 1]
            ),
            "template_domain_names": [f"none".encode()] * num_temp,
            "template_release_date": [f"none".encode()] * num_temp,
            "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
        }
        return template_features
    
    

    def build_monomer_feature(self, sequence: str, unpaired_msa: str, template_features: Dict[str, Any]):
        msa = pipeline.parsers.parse_a3m(unpaired_msa)
        # gather features
        return {
            **pipeline.make_sequence_features(
                sequence=sequence, description="none", num_res=len(sequence)
            ),
            **pipeline.make_msa_features([msa]),
            **template_features,
        }


    def pair_msa(
        self,
        query_seqs_unique: List[str],
        query_seqs_cardinality: List[int],
        paired_msa: Optional[List[str]],
        unpaired_msa: Optional[List[str]],
    ) -> str:
        if paired_msa is None and unpaired_msa is not None:
            a3m_lines = self.pad_sequences(unpaired_msa, query_seqs_unique, query_seqs_cardinality)
        elif paired_msa is not None and unpaired_msa is not None:
            a3m_lines = (
                self.pair_sequences(paired_msa, query_seqs_unique, query_seqs_cardinality)
                + "\n" +self.pad_sequences(unpaired_msa, query_seqs_unique, query_seqs_cardinality)
            )
        elif paired_msa is not None and unpaired_msa is None:
            a3m_lines = self.pair_sequences(paired_msa, query_seqs_unique, query_seqs_cardinality)
        else:
            raise ValueError(f"Invalid pairing")
        return a3m_lines
                    
    def run(self):
        pad_len = 0
        ranks = []
        metrics = []
        job_number = 0
        for job_number, (raw_jobname, query_sequence, a3m_lines) in enumerate(self.queries):
            if self.jobname_prefix is not None:
                # pad job number based on number of queries
                fill = len(str(len(self.queries)))
                jobname = safe_filename(self.jobname_prefix) + "_" + str(job_number).zfill(fill)
                job_number += 1
            else:
                jobname = safe_filename(raw_jobname)
            self.log_out(f"Job '{jobname}' (job_n: {job_number}): {(raw_jobname, query_sequence, a3m_lines)}")

            #######################################
            # check if job has already finished
            #######################################
            # In the colab version and with --zip we know we're done when a zip file has been written
            result_zip = self.result_dir.joinpath(jobname).with_suffix(".result.zip")
            if self.keep_existing_results and result_zip.is_file():
                self.log_out(f"Skipping {jobname} (result.zip)")
                continue
            # In the local version we use a marker file
            is_done_marker = self.result_dir.joinpath(jobname + ".done.txt")
            if self.keep_existing_results and is_done_marker.is_file():
                self.log_out(f"Skipping {jobname} (already done)")
                continue

            seq_len = len("".join(query_sequence))
            self.log_out(f"Query {job_number + 1}/{len(self.queries)}: {jobname} (length {seq_len})")

            ###########################################
            # generate MSA (a3m_lines) and templates
            ###########################################
            try:
                pickled_msa_and_templates = self.result_dir.joinpath(f"{jobname}.pickle")
                if pickled_msa_and_templates.is_file():
                    with open(pickled_msa_and_templates, 'rb') as f:
                        (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features) = pickle.load(f)
                    self.log_out(f"Loaded {pickled_msa_and_templates}")
                else:
                    if a3m_lines is None:
                        self.log_out(f"a3m_lines is None, getting msa (msa mode: {self.msa_mode}) and templates")
                        (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features) = self.get_msa_and_templates(jobname, query_sequence, a3m_lines, self.msa_mode)

                    elif a3m_lines is not None:
                        self.log_out(f"a3m_lines is NOT None, unserialising msa")
                        (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features) = self.unserialize_msa(a3m_lines, query_sequence)
                        if self.use_templates:
                            self.log_out(f"\tUsing templates")
                            (_, _, _, _, template_features) = self.get_msa_and_templates(jobname, query_seqs_unique, unpaired_msa, 'single_sequence')

                    if self.num_models == 0:
                        self.log_out(f"num_models is 0")
                        with open(pickled_msa_and_templates, 'wb') as f:
                            pickle.dump((unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features), f)
                        self.log_out(f"Saved {pickled_msa_and_templates}")

                # save a3m
                msa = self.msa_to_str(unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality)
                self.result_dir.joinpath(f"{jobname}.a3m").write_text(msa)

            except Exception as e:
                logger.exception(f"Could not get MSA/templates for {jobname}: {e}")
                continue

            #######################
            # generate features
            #######################
            try:
                self.log_out(f"Generating input feature:\n\t- query_seqs_unique: {query_seqs_unique}\n\t- query_seqs_cardinality: {query_seqs_cardinality}\n\t- unpaired_msa: {unpaired_msa}\n\t- paired_msa: {paired_msa}\n\t- template_features: {template_features}")
                (feature_dict, domain_names) = self.generate_input_feature(query_seqs_unique, query_seqs_cardinality, unpaired_msa, paired_msa, template_features)

                # to allow display of MSA info during colab/chimera run (thanks tomgoddard)
                if self.feature_dict_callback is not None:
                    self.feature_dict_callback(feature_dict)
            except Exception as e:
                logger.exception(f"Could not generate input features {jobname}: {e}")
                continue

            if self.use_templates:
                t_file = f"{jobname}_template_domain_names.json"
                self.log_out(f"Using template file: {t_file}")
                templates_file = self.result_dir.joinpath(t_file)
                templates_file.write_text(json.dumps(domain_names))

            ######################
            # predict structures
            ######################
            if self.num_models <= 0:
                self.log_out(f"num_models <= 0")
                logger.error(f"num_models <= 0")
                return
                
            try:
                # get list of lengths
                query_sequence_len_array = sum([[len(x)] * y for x,y in zip(query_seqs_unique, query_seqs_cardinality)],[])
                self.log_out(f"query_sequence_len_array: {query_sequence_len_array}")
                self.log_out(f"seq_len: {seq_len}, pad_len: {pad_len}")

                # decide how much to pad (to avoid recompiling)
                if seq_len > pad_len:
                    if isinstance(self.recompile_padding, float):
                        pad_len = math.ceil(seq_len * self.recompile_padding)
                    else:
                        pad_len = seq_len + self.recompile_padding
                    pad_len = min(pad_len, self.max_len)

                # prep model and params
                if self.first_job:
                    self.log_out(f"FIRST JOB")
                    self.log_out(f"Total queries: {len(self.queries)}, msa_mode: {self.msa_mode}")
                    # if one job input adjust max settings
                    if len(self.queries) == 1 and self.msa_mode != "single_sequence":
                        # get number of sequences
                        if "msa_mask" in feature_dict:
                            num_seqs = int(sum(feature_dict["msa_mask"].max(-1) == 1))
                        else:
                            num_seqs = int(len(feature_dict["msa"]))

                        if self.use_templates: num_seqs += 4

                        # adjust max settings
                        self.max_seq = min(num_seqs, self.max_seq)
                        self.max_extra_seq = max(min(num_seqs - self.max_seq, self.max_extra_seq), 1)
                        self.log_out(f"Setting max_seq={self.max_seq}, max_extra_seq={self.max_extra_seq}")
                    self.log_out(f"max_seq: {self.max_seq}, max_extra_seq: {self.max_extra_seq}")

                    model_runner_and_params = load_models_and_params(
                        num_models=self.num_models,
                        use_templates=self.use_templates,
                        num_recycles=self.num_recycles,
                        num_ensemble=self.num_ensemble,
                        model_order=self.model_order,
                        model_type=self.model_type,
                        data_dir=self.data_dir,
                        stop_at_score=self.stop_at_score,
                        rank_by=self.rank_by,
                        use_dropout=self.use_dropout,
                        max_seq=self.max_seq,
                        max_extra_seq=self.max_extra_seq,
                        use_cluster_profile=self.use_cluster_profile,
                        recycle_early_stop_tolerance=self.recycle_early_stop_tolerance,
                        use_fuse=self.use_fuse,
                        use_bfloat16=self.use_bfloat16,
                        save_all=self.save_all,
                    )
                    self.first_job = False

                self.output['embeddings'] = self.predict_structure(
                    prefix=jobname,
                    feature_dict=feature_dict,
                    sequences_lengths=query_sequence_len_array,
                    pad_len=pad_len,
                    model_runner_and_params=model_runner_and_params
                )
            except RuntimeError as e:
                # This normally happens on OOM. TODO: Filter for the specific OOM error message
                logger.error(f"Could not predict {jobname}. Not Enough GPU memory? {e}")
                continue

        self.log_out("Done")
        return self.output
    
    
    
    def pad_input(self,
        input_features: model.features.FeatureDict,
        model_runner: model.RunModel,
        model_name: str,
        pad_len: int,
        use_templates: bool,
    ) -> model.features.FeatureDict:

        model_config = model_runner.config
        eval_cfg = model_config.data.eval
        crop_feats = {k: [None] + v for k, v in dict(eval_cfg.feat).items()}

        max_msa_clusters = eval_cfg.max_msa_clusters
        max_extra_msa = model_config.data.common.max_extra_msa
        # templates models
        if (model_name == "model_1" or model_name == "model_2") and use_templates:
            pad_msa_clusters = max_msa_clusters - eval_cfg.max_templates
        else:
            pad_msa_clusters = max_msa_clusters

        max_msa_clusters = pad_msa_clusters

        # let's try pad (num_res + X)
        input_fix = make_fixed_size(
            input_features,
            crop_feats,
            msa_cluster_size=max_msa_clusters,  # true_msa (4, 512, 68)
            extra_msa_size=max_extra_msa,  # extra_msa (4, 5120, 68)
            num_res=pad_len,  # aatype (4, 68)
            num_templates=4,
        )  # template_mask (4, 4) second value
        return input_fix

    
    def predict_structure(self,
        prefix: str,
        feature_dict: Dict[str, Any],
        sequences_lengths: List[int],
        pad_len: int,
        model_runner_and_params: List[Tuple[str, model.RunModel, haiku.Params]],
    ):
        """Predicts structure using AlphaFold for the given sequence."""
        mean_scores = []
        conf = []
        unrelaxed_pdb_lines = []
        prediction_times = []
        model_names = []
        files = FileManager(prefix, self.result_dir)
        seq_len = sum(sequences_lengths)
        model_embeddings = {}
        
        # iterate through random seeds
        # Gabry: Technically this should always be one.
        # random_seed is always 0 while num_seeds is always 1
        for seed_num, seed in enumerate(range(self.random_seed, self.random_seed+self.num_seeds)):
            # iterate through models
            for model_num, (model_name, model_runner, params) in enumerate(model_runner_and_params):
                # swap params to avoid recompiling
                model_runner.params = params
                #########################
                # process input features
                #########################
                if "multimer" in self.model_type:
                    if model_num == 0 and seed_num == 0:
                        # TODO: add pad_input_mulitmer()
                        input_features = feature_dict
                        input_features["asym_id"] = input_features["asym_id"] - input_features["asym_id"][...,0]
                else:
                    if model_num == 0:
                        input_features = model_runner.process_features(feature_dict, random_seed=seed)
                        r = input_features["aatype"].shape[0]
                        input_features["asym_id"] = np.tile(feature_dict["asym_id"],r).reshape(r,-1)
                        if seq_len < pad_len:
                            input_features = self.pad_input(input_features, model_runner,model_name, pad_len, self.use_templates)
                            self.log_out(f"Padding length to {pad_len}")

                tag = f"{self.model_type}_{model_name}_seed_{seed:03d}"
                model_names.append(tag)
                files.set_tag(tag)
                
                
                # monitor intermediate results
                def callback(result, recycles):
                    if recycles == 0: result.pop("tol",None)
                    if not self.is_complex: result.pop("iptm",None)
                    print_line = ""
                    for x,y in [["mean_plddt","pLDDT"],["ptm","pTM"],["iptm","ipTM"],["tol","tol"]]:
                        if x in result:
                            print_line += f" {y}={result[x]:.3g}"
                    self.log_out(f"{tag} recycle={recycles}{print_line}")

                    if self.save_recycles:
                        final_atom_mask = result["structure_module"]["final_atom_mask"]
                        b_factors = result["plddt"][:, None] * final_atom_mask
                        unrelaxed_protein = af_protein.from_prediction(
                            features=input_features,
                            result=result, b_factors=b_factors,
                            remove_leading_feature_dimension=("multimer" not in self.model_type))
                        files.get("unrelaxed",f"r{recycles}.pdb").write_text(af_protein.to_pdb(unrelaxed_protein))

                        if self.save_all:
                            with files.get("all",f"r{recycles}.pickle").open("wb") as handle:
                                pickle.dump(result, handle)
                        del unrelaxed_protein

                return_representations = self.save_all or self.save_single_representations or self.save_pair_representations

                ########################
                # predict
                ########################
                start = time.time()
                result, recycles = model_runner.predict(input_features, random_seed=seed, return_representations=return_representations, callback=callback) # predict
                model_exec_time = time.time() - start
                prediction_times.append(model_exec_time)

                ########################
                # parse results
                ########################
                # summary metrics
                mean_scores.append(result["ranking_confidence"])
                if recycles == 0: result.pop("tol",None)
                if not self.is_complex: result.pop("iptm",None)
                print_line = ""
                conf.append({})
                for x,y in [["mean_plddt","pLDDT"],["ptm","pTM"],["iptm","ipTM"]]:
                    if x in result:
                        print_line += f" {y}={result[x]:.3g}"
                        conf[-1][x] = float(result[x])
                conf[-1]["print_line"] = print_line
                self.log_out(f"{tag} took {prediction_times[-1]:.1f}s ({recycles} recycles)")

                # create protein object
                final_atom_mask = result["structure_module"]["final_atom_mask"]
                b_factors = result["plddt"][:, None] * final_atom_mask
                unrelaxed_protein = af_protein.from_prediction(
                    features=input_features,
                    result=result,
                    b_factors=b_factors,
                    remove_leading_feature_dimension=("multimer" not in self.model_type))

                # callback for visualization
                if self.prediction_callback is not None:
                    self.prediction_callback(unrelaxed_protein, sequences_lengths,result, input_features, (tag, False))

                embeddings = result["representations"]["single"]
                model_embeddings[model_name] = {}
                model_embeddings[model_name]['embeddings'] = embeddings
                model_embeddings[model_name]['time'] = model_exec_time
                
                if self.save_single_representations:
                    np.save(files.get("single_repr","npy"), embeddings)           
                    
                del result, unrelaxed_protein

                # early stop criteria fulfilled
                if mean_scores[-1] > self.stop_at_score: break

            # early stop criteria fulfilled
            if mean_scores[-1] > self.stop_at_score: break

            # cleanup
            if "multimer" not in self.model_type: del input_features
        if "multimer" in self.model_type: del input_features

        return model_embeddings



