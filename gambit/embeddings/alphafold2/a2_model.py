# Gabry: This is from https://www.kaggle.com/code/alexandervc/alphafold2predictstructure and from the AlphaFold notebook in alphafold's github
# Gabry: Check https://github.com/xinformatics/alphafold/blob/main/Representations_AlphaFold2_v3.ipynb

import enum, os, sys, random, shutil
import jax, tqdm
import numpy as np
import haiku as hk

import alphafold as a2
# from alphafold.common import protein
# from alphafold.data import pipeline
# from alphafold.data import templates
# from alphafold.model import data
# from alphafold.model import config
# from alphafold.model import model
# from alphafold.relax import relax

@enum.unique
class ModelType(enum.Enum):
  MONOMER = 0
  MULTIMER = 1


# Gabry: This is our override

class RunRepresentationsModel(a2.model.model.RunModel):
  def __init__(self, *args, **kwargs,  ):
    super().__init__(*args, **kwargs)

    self.multimer_mode = False # Default to Monomer mode

    if self.multimer_mode:
      def _forward_fn(batch):
        model = a2.modules_multimer.AlphaFold(self.config.model)
        return model(
            batch,
            is_training=False,
            return_representations=True)
    else:
      def _forward_fn(batch):
        model = a2.modules.AlphaFold(self.config.model)
        # Here we call the AlphaFold model with the extra `return_representations` param set to True
        return model(
            batch,
            is_training=False,
            compute_loss=False,
            ensemble_representations=True,
            return_representations=True)

    self.apply = jax.jit(hk.transform(_forward_fn).apply)
    self.init = jax.jit(hk.transform(_forward_fn).init)



def mk_mock_template(query_sequence):
  # since alphafold's model requires a template input
  # we create a blank example w/ zero input, confidence -1
  ln = len(query_sequence)
  output_templates_sequence = "-"*ln
  output_confidence_scores = np.full(ln,-1)
  templates_all_atom_positions = np.zeros((ln, a2.data.templates.residue_constants.atom_type_num, 3))
  templates_all_atom_masks = np.zeros((ln, a2.data.templates.residue_constants.atom_type_num))
  templates_aatype = a2.data.templates.residue_constants.sequence_to_onehot(output_templates_sequence, a2.data.templates.residue_constants.HHBLITS_AA_TO_ID)
  template_features = {'template_all_atom_positions': templates_all_atom_positions[None],
                       'template_all_atom_masks': templates_all_atom_masks[None],
                       'template_sequence': [f'none'.encode()],
                       'template_aatype': np.array(templates_aatype)[None],
                       'template_confidence_scores': output_confidence_scores[None],
                       'template_domain_names': [f'none'.encode()],
                       'template_release_date': [f'none'.encode()]}
  return template_features


def predict_structure(prefix, feature_dict, model_runners, do_relax=True, random_seed=0):  
  """Predicts structure using AlphaFold for the given sequence."""

  # Run the models.
  plddts = {}
  embeddings = {}
  
  for model_name, model_runner in model_runners.items():
    processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)
    prediction_result = model_runner.predict(processed_feature_dict)
    # unrelaxed_protein = a2.common.protein.from_prediction(processed_feature_dict,prediction_result)
    # unrelaxed_pdb_path = f'{prefix}_unrelaxed_{model_name}.pdb'
    plddts[model_name] = prediction_result['plddt']
    embeddings[model_name] = prediction_result['representations']

    # print(f"{model_name} {plddts[model_name].mean()}")

    # with open(unrelaxed_pdb_path, 'w') as f:
    #   f.write(a2.common.protein.to_pdb(unrelaxed_protein))

    # if do_relax:
    #   # Relax the prediction.
    #   amber_relaxer = a2.relax.relax.AmberRelaxation(max_iterations=0, tolerance=2.39, stiffness=10.0,exclude_residues=[], max_outer_iterations=20)      
    #   relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
    #   relaxed_pdb_path = f'{prefix}_relaxed_{model_name}.pdb'
    #   with open(relaxed_pdb_path, 'w') as f:
    #     f.write(relaxed_pdb_str)

  return plddts, embeddings

