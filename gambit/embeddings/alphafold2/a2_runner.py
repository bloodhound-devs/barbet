from a2_model import mk_mock_template, predict_structure
from a2_config import jobname, a3m_file, model_runners
import alphafold as a2
import pickle

def run_af2():
    a3m_lines = ""
    with open(a3m_file,"r") as f:
        a3m_lines = "".join(f.readlines())
    msa, deletion_matrix = a2.data.pipeline.parsers.parse_a3m(a3m_lines)
    query_sequence = msa[0]

    feature_dict = {
        **a2.data.pipeline.make_sequence_features(sequence=query_sequence, description="none", num_res=len(query_sequence)),
        **a2.data.pipeline.make_msa_features(msas=[msa], deletion_matrices=[deletion_matrix]),
        **mk_mock_template(query_sequence)
    }
    plddts, embeddings = predict_structure(jobname, feature_dict, model_runners)
    with open(f"{jobname}_embeddings.pkl", 'wb') as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{jobname}_plddts.pkl", 'wb') as f:
        pickle.dump(plddts, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  run_af2()
