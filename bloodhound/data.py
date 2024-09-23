import os
import numpy as np

RANKS = ["phylum", "class", "order", "family", "genus", "species"]

def read_memmap(path, count, dtype:str="float16") -> np.memmap:
    file_size = os.path.getsize(path)
    dtype_size = np.dtype(dtype).itemsize
    num_elements = file_size // dtype_size
    embedding_size = num_elements // count
    shape = (count, embedding_size)
    return np.memmap(path, dtype=dtype, mode='r', shape=shape)


def gene_id_from_accession(accession:str):
    return accession.split("/")[-1]
