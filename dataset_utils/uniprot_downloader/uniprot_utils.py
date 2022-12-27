import requests as rq
import json
import re
import time as t 

UNIPROT_API_URL = 'https://rest.uniprot.org/uniprotkb/search'

def search_uniprot(query: str = '', fields: list[str] = [], size: int = 5, cursor: str = None, format: str = 'json', 
                   params: dict = None, json_dump: str = None, verbose = False) -> tuple[rq.models.Response, str]:
    '''
    Do a UniProt search through its API. Parameters are explained here: https://www.uniprot.org/help/api_queries

    Providing a custom params dict (https://requests.readthedocs.io/en/latest/api/?highlight=get#requests.request)
    overridess ALL other search parameters.
    '''
    if params is None:
        params = {
            'query': query,
            'format': format,
            'fields': fields,
            'size': size,
            'cursor': cursor,
        }

    uniprot_request = rq.Request(url=UNIPROT_API_URL, params=params, method='GET')
    uniprot_request = uniprot_request.prepare()

    log(f'Requesting data using URL {uniprot_request.url}', verbose)

    session = rq.Session()
    uniprot_request = session.send(uniprot_request)
    
    log(f'UniProt returned code {uniprot_request.status_code}', verbose)

    uniprot_request.raise_for_status()

    # Cursor is explained here: https://www.uniprot.org/help/pagination
    # Basically, it's a hash pointing to the next page of results. Passing it as
    # as parameter will retrieve the next set of results, following the current request.
    # This extracts it from the link returned in the headers.
    if 'link' in uniprot_request.headers.keys():
        header_link = uniprot_request.headers['link']
        cursor_match = re.search(string=header_link, 
                                 pattern=r'&cursor=.+&').group(0)
        next_cursor = cursor_match[8:-1]
    else:
        next_cursor = None

    if json_dump is not None:
        with open(json_dump, 'w') as out_json:
            log(f'Dumping response to json', verbose)
            json.dump(obj=uniprot_request.json(), fp=out_json)

    return uniprot_request, next_cursor

def log(msg: str, verbose: bool):
    if verbose:
        print(msg)