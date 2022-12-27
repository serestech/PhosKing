import datetime as dt
import os
from uniprot_utils import *

# Search query for UniProt
UNIPROT_QUERY = 'reviewed:true'  # Only manually annotated (> 500.000 proteins)

# Fields to retrieve from UniProt
RETRIEVE_FIELDS = ['accession', 'id', 'protein_name', 'organism_name', 'sequence',
                   'ft_carbohyd', 'ft_lipid', 'ft_mod_res', 'cc_ptm']

# Size of the batch of proteins to retrieve on every search
UNIPROT_FETCH_SIZE = 500   # UniProt sais 500 is the optimal size (https://www.uniprot.org/help/api_queries)

# How many searches between logs
LOG_EVERY = 10

# Download folder for UniProt data
TIMESTAMP = dt.datetime.now().strftime("%d_%m_%Y_%H_%M")
DOWNLOAD_FOLDER = f'data/uniprot_dump_{TIMESTAMP}'

os.makedirs(DOWNLOAD_FOLDER)

start = t.time()

finished = False
fetched_results = 0
next_cursor = None
i = 0
while not finished:
    uniprot_request, next_cursor = search_uniprot(query=UNIPROT_QUERY,
                                                  fields=RETRIEVE_FIELDS,
                                                  size=UNIPROT_FETCH_SIZE,
                                                  cursor=next_cursor,
                                                  json_dump=f'{DOWNLOAD_FOLDER}/{i}_{fetched_results}_{fetched_results + UNIPROT_FETCH_SIZE}.json',
                                                  verbose=True if (i + 1) % LOG_EVERY == 0 else False)

    fetched_results += len(uniprot_request.json()['results'])

    if (i + 1) % LOG_EVERY == 0:
        now = t.time()
        total_sequences = int(uniprot_request.headers["X-Total-Results"])
        print(f'Done {i + 1} searches in {now - start:.1f}s')
        print(f'Fetched {fetched_results}/{total_sequences} squences ({fetched_results / total_sequences * 100:.1f}% done)')

    if next_cursor is None:
        finished = True

    i += 1
