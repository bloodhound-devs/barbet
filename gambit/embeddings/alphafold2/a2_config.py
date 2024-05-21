import enum, os, sys, random, shutil
import jax, tqdm
import numpy as np
import haiku as hk
import re
import hashlib

import alphafold as a2
from a2_model import RunRepresentationsModel

def add_hash(x,y):
  return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]

protein_test_1 = 'MLILISPAKTLDYQSPLTTTRYTLPELLDNSQQLIHEARKLTPPQISTLMRISDKLAGINAARFHDWQPDFTPANARQAILAFKGDVYTGLQAETFSEDDFDFAQQHLRMLSGLYGVLRPLDLMQPYRLEMGIRLENARGKDLYQFWGDIITNKLNEALAAQGDNVVINLASDEYFKSVKPKKLNAEIIKPVFLDEKNGKFKIISFYAKKARGLMSRFIIENRLTKPEQLTGFNSEGYFFDEDSSSNGELVFKRYEQR' #@param {type:"string"}

protein_test_2 = 'MTVKTEAAKGTLTYSRMRGMVAILIAFMKQRRMGLNDFIQKIANNSYACKHPEVQSILKISQPQEPELMNANPSPPPSPSQQINLGPSSNPHAKPSDFHFLKVIGKGSFGKVLLARHKAEEVFYAVKVLQKKAILKKKEEKHIMSERNVLLKNVKHPFLVGLHFSFQTADKLYFVLDYINGGELFYHLQRERCFLEPRARFYAAEIASALGYLHSLNIVYRDLKPENILLDSQGHIVLTDFGLCKENIEHNSTTSTFCGTPEYLAPEVLHKQPYDRTVDWWCLGAVLYEMLYGLPPFYSRNTAEMYDNILNKPLQLKPNITNSARHLLEGLLQKDRTKRLGAKDDFMEIKSHVFFSLINWDDLINKKITPPFNPNVSGPNDLRHFDPEFTEEPVPNSIGKSPDSVLVTASVKEAAEAFLGFSYAPPTDSFL'

protein = protein_test_1
# remove whitespaces and force uppercase
protein = "".join(protein.split())
protein = re.sub(r'[^a-zA-Z]','', protein).upper()
query_sequence = protein

jobname = 'gambit_AF2' #@param {type:"string"}
jobname = "".join(jobname.split())
jobname = re.sub(r'\W+', '', jobname)
jobname = add_hash(jobname, query_sequence)

# Create fasta file from the sequence
with open(f"{jobname}.fasta", "w") as text_file:
    text_file.write(">1\n%s" % query_sequence)


msa_mode = "single_sequence" #@param ["MMseqs2 (UniRef+Environmental)", "MMseqs2 (UniRef only)","single_sequence","custom"]
use_msa = True if msa_mode.startswith("MMseqs2") else False
use_env = True if msa_mode == "MMseqs2 (UniRef+Environmental)" else False
use_custom_msa = True if msa_mode == "custom" else False
use_amber = False #@param {type:"boolean"}
use_templates = True #@param {type:"boolean"}
homooligomer = 1 #@param [1,2,3,4,5,6,7,8] {type:"raw"}


# decide which a3m to use
a3m_file = 'none.a3m'
if use_msa:
  a3m_file = f"{jobname}.a3m"
elif use_custom_msa:
  a3m_file = f"{jobname}.custom.a3m"
  if not os.path.isfile(a3m_file):
    # custom_msa_dict = files.upload() # this is from google.colab: `from google.colab import files`
    # No idea I'm just making this up as a placeholder, in case we'll need it in the future
    with open('custom_msa.json', 'r') as f:
        custom_msa_dict = f.read()
    custom_msa = list(custom_msa_dict.keys())[0]
    header = 0
    import fileinput
    for line in fileinput.FileInput(custom_msa,inplace=1):
      if line.startswith(">"):
         header = header + 1 
      if line.startswith("#"):
        continue
      if line.rstrip() == False:
        continue
      if line.startswith(">") == False and header == 1:
         query_sequence = line.rstrip() 
      print(line, end='')
    os.rename(custom_msa, a3m_file)
    print(f"moving {custom_msa} to {a3m_file}")
else:
  a3m_file = f"{jobname}.single_sequence.a3m"
  with open(a3m_file, "w") as text_file:
    text_file.write(">1\n%s" % query_sequence)


# AlphaFold2 offer 5 models, each model can be monomer or multimer. Monomer is the default one.
# The model parameter files are in the params folder under `/data/scratch/datasets/alphafold/v2.3.2/params` on Spartan.
# Each model consists of 4 parameter files where <N> is 1,2,3,4 or 5:
#  
# - params_model_<N>.npz              # Monomer 
# - params_model_<N>_ptm.npz          # Monomer
# - params_model_<N>_multimer_v2.npz  # Multimer v2 (I guess from alphafold v2.2.0)
# - params_model_<N>_multimer_v3.npz  # Multimer v2 (I guess from alphafold v2.3.2)

num_models = 1 #@param [1,2,3,4,5] {type:"raw"} number of models to use
params_dir = '/data/scratch/datasets/alphafold/v2.3.2/params'
available_models = ["model_1","model_2","model_3","model_4","model_5"]
model_runners = {} # collect model weights
if "model_params" not in dir():
   model_params = {}
for model_name in available_models[:num_models]:
  if model_name not in model_params:
    model_params[model_name] = a2.model.data.get_model_haiku_params(model_name=model_name+"_ptm", data_dir=params_dir)
    if model_name == "model_1":
      model_config = a2.model.config.model_config(model_name+"_ptm") # is '_ptm' even needed?
      model_config.data.eval.num_ensemble = 1
    if model_name == "model_3":
      model_config = a2.model.config.model_config(model_name+"_ptm") # is '_ptm' even needed?
      model_config.data.eval.num_ensemble = 1
    model_runners[model_name] = RunRepresentationsModel(model_config, model_params[model_name])
      