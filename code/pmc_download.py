# ==============================================================================
# utilities to download pubmed central full text files
# ==============================================================================

# load modules
import requests, json, sys
from pathlib import Path
from requests.exceptions import HTTPError
from requests.models import Response


# setting paths
DIR_PROJECT = Path().cwd()
DIR_CODE = DIR_PROJECT / "code"
DIR_DOCS = DIR_PROJECT / "documents"

# pubmed config file
FILE_CONFIG = "config.json"

# load pubmed config file
with open(DIR_DOCS / FILE_CONFIG) as f:
    config = json.load(f)

# set parameters to put in pmc header
params = {
    "tool": config["tool"],
    "email": config["email"],
    "api_key": config["pubmed_api_key"]
}

# function that takes an id and returns pmc nxml response
def get_nxml(id, params):
    URL = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={id}"
    try:
        response = requests.get(URL, params=params)
        # error structure from : https://realpython.com/python-requests/
        # If the response was successful, no Exception will be raised
        response.raise_for_status()
        return response
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')  # Python 3.6
    except Exception as err:
        print(f'Other error occurred: {err}')  # Python 3.6

# make calls for list of ids
xml_list = []
for i in id_list: #id list comes from biopython
    xml_file = get_nxml(i, params=params)
    xml_list.append(xml_file)
    sys.sleep(.334)

# test it out
test = get_nxml(id="7772474", params=params)

# save as xml
with open('/Users/brandonsepulvado/Documents/synbio/output/7772474.xml', 'wb') as file:
    file.write(test.content)
