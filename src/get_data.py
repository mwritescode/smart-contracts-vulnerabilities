import os
import time
import requests
from hexbytes import HexBytes

import tqdm
import pandas as pd
from web3 import Web3

from .credentials import INFURA_URI, ETHERSCAN_API

class DataManager():
    def __init__(self, base_path, infura_uri=INFURA_URI, etherscan_api=ETHERSCAN_API):
        self.bytecode_path = os.path.join(base_path, 'bytecode')
        self.solidity_path = os.path.join(base_path, 'sol')
        self.web3_provider = Web3(Web3.HTTPProvider(INFURA_URI))
        self.etherscan_api = ETHERSCAN_API
        self.contracts = pd.read_csv(os.path.join(base_path, 'contracts.csv'))

    def scrape_bytecode(self, address):
        file_path = os.path.join(self.bytecode_path, address+'_ext.txt')
        checksum_addr = Web3.toChecksumAddress(address)
        out = self.web3_provider.eth.get_code(checksum_addr)
        with open(file_path, 'w') as out_file:
            out_file.write(out.hex())

    def get_solidity_code(self, address):
        api_call = 'https://api.etherscan.io/api?module=contract&action=getsourcecode&address='\
                    + address + '&apikey=' + self.etherscan_api
        file_path = os.path.join(self.solidity_path, address+'_ext.sol')
        response = requests.get(api_call)
        if response.status_code == 200:
            response = response.json()
            if response['status'] == '0':
                raise Exception(response['result'] + \
                    '. Please make max 5 API calls a second on Etherescan free tier.')
            code = response['result'][0]['SourceCode']
            with open(file_path, 'w') as out_file:
                out_file.write(code)
            time.sleep(0.3)

    def download(self):
        for addr in tqdm.tqdm(self.contracts['address']):
                self.get_solidity_code(addr)
                self.scrape_bytecode(addr)