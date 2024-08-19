import csv
from pathlib import Path


def read_tophits(path:Path|str) -> dict[str,str]:
    # Initialize an empty dictionary to store the data
    gene_family_dict = {}

    # Read the TSV file
    with open(path, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            # Extract the Gene Id and Top hits (Family id,e-value,bitscore)
            gene_id = row['Gene Id']
            top_hits = row['Top hits (Family id,e-value,bitscore)']
            
            # Split the top_hits to get the Family id
            family_id = top_hits.split(',')[0]
            
            # Add to the dictionary
            gene_family_dict[gene_id] = family_id

    return gene_family_dict


def read_tigrfam(file_path:Path|str) -> dict[str,str]:
    # Initialize an empty dictionary to store the data
    gene_family_dict = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Find the header line with the actual column names
        header_line = ""
        for line in lines:
            if line.startswith('#') and 'target name' in line:
                header_line = line.strip('# \n')
                break
        
        # Re-open the file to use the CSV DictReader from the correct position
        with open(file_path, 'r') as file:
            # Skip lines until we find the header line
            while True:
                line = file.readline()
                if header_line in line:
                    break

            # Read the TSV data starting from the header line
            reader = csv.DictReader(file, delimiter='\t', fieldnames=header_line.split())
            next(reader)  # Skip the header row itself
            for row in reader:
                # Extract the target name and query name
                gene_id = row['target name'].strip()
                family_id = row['query name'].strip()
                
                # Add to the dictionary
                gene_family_dict[gene_id] = family_id

    return gene_family_dict


def read_pfam(file_path:Path|str) -> dict[str,str]:
    # Initialize an empty dictionary to store the data
    gene_family_dict = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Find the header line with the actual column names
        for i, line in enumerate(lines):
            if line.startswith('#') and '<seq id>' in line:
                header_index = i
                break

        # Read the TSV data starting from the header line
        reader = csv.DictReader(lines[header_index + 1:], delimiter='\t', 
                                fieldnames=["seq_id", "alignment_start", "alignment_end", "envelope_start", "envelope_end", "hmm_acc", "hmm_name", "type", "hmm_start", "hmm_end", "hmm_length", "bit_score", "e_value", "significance", "clan"])
        for row in reader:
            # Extract the seq id and hmm acc
            gene_id = row['seq_id'].strip()
            family_id = row['hmm_acc'].strip()

            # Add to the dictionary
            gene_family_dict[gene_id] = family_id

    return gene_family_dict
