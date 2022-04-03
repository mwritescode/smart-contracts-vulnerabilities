import requests
import pandas as pd
import tqdm
import time
import os

from web3 import Web3
from credentials import INFURA_URI, ETHERSCAN_API

w3 = Web3(Web3.HTTPProvider(INFURA_URI))
print(w3.isConnected())

def scrape_bytecode(web3_provider, address):
    file_path = 'data/bytecode/'+address+'_ext.txt'
    checksum_addr = Web3.toChecksumAddress(address)
    out = web3_provider.eth.get_code(checksum_addr)
    with open(file_path, 'wb') as out_file:
        out_file.write(out)

def get_solidity_code(address):
    api_call = 'https://api.etherscan.io/api?module=contract&action=getsourcecode&address='\
    + address + '&apikey=' + ETHERSCAN_API
    file_path = 'data/sol/'+address+'_ext.sol'
    response = requests.get(api_call)
    if response.status_code == 200:
        print(response.json())
        code = response.json()['result'][0]['SourceCode']
        with open(file_path, 'w') as out_file:
            out_file.write(code)
            #time.sleep(0.3)

addresses = pd.read_csv('data/contracts.csv')
for addr in tqdm.tqdm(addresses['address']):
    get_solidity_code(addr)
    scrape_bytecode(w3, addr)