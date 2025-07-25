================================
Preprocessing
================================

Required files
================================

If you want to prepare your own data for training, you need the marker genes and taxonomy from GTDB. 
Use these commands to download the bacteria files from the latest GTDB release.

.. code-block:: bash

    wget https://data.gtdb.ecogenomic.org/releases/latest/genomic_files_all/bac120_marker_genes_all.tar.gz
    wget https://data.gtdb.ecogenomic.org/releases/latest/bac120_taxonomy.tsv.gz


Use these commands to download the archaea files from the latest GTDB release.

.. code-block:: bash

    wget https://data.gtdb.ecogenomic.org/releases/latest/genomic_files_all/ar53_marker_genes_all.tar.gz
    wget https://data.gtdb.ecogenomic.org/releases/latest/ar53_taxonomy.tsv.gz

If you want just the representative genomes, then download files from the ``genomic_files_reps`` directory.

If you have a custom dataset, you can use the same file format as these.

Individual Gene Embeddings
==========================

The next step in preprocessing is to build embedding files for each gene family.

.. code-block:: bash

    for INDEX in $(seq 0 119); do
        barbet-esm build-gene-array \
            --marker-genes bac120_marker_genes_all.tar.gz \
            --output-dir preprocessed/ \
            --family-index $INDEX
    done

Each gene family will be saved in a separate file in the ``preprocessed`` directory.
Each of the jobs in this loop are independent, so if you are on a system with multiple GPUs, you can run them in parallel.

This loop assumes that you have 120 gene families, which is the case for the GTDB bacteria dataset. If you are using archaea, 
you will need to change the number of iterations to 53 (i.e. ``seq 0 52``).

By default, it will use the ESM model with 6 layers to build the embeddings. 
You can choose the number of layers with the ``--layers`` option with one of the following values: 6, 12, 30, 33, 36, or 48.

Final preprocessing step
=================================

The final preprocessing step is to collect the embeddings into a single file and to build the TreeDict.

.. code-block:: bash

    barbet-esm preprocess \
        --taxonomy bac120_taxonomy.tsv.gz \
        --marker-genes bac120_marker_genes_all.tar.gz \
        --output-dir preprocessed

This will create three files called:
    - ``preprocessed.npy`` - the memmap array with embeddings for each gene family
    - ``preprocessed.txt`` - the index to the memmap array
    - ``preprocessed.st`` - the TreeDict which stores the location on the taxonomy tree for each genome.

Now you are ready to begin :doc:`training` your model.
