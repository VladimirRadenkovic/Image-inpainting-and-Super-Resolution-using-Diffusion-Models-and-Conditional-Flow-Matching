from typing import List, Optional, Union

import numpy as np
import requests
from loguru import logger


class PDBFlexAPI:
    BASE_URL = "https://pdbflex.org/php/api/"
    API_REFERNCE = "https://pdbflex.org/api.html"

    def __init__(self):
        pass

    def __repr__(self) -> str:
        return f"PDBFlexAPI({self.API_REFERNCE})"

    def _handle_query(self, url: str) -> dict:
        """Query API and return response."""

        # Query API
        logger.debug(f"Querying PDBFlex API: {url}")
        response = requests.get(url)

        # Check response status
        if response.status_code != 200:
            try:
                response = response.json()
                raise Exception(response["error"])
            except:
                raise RuntimeError(f"Response status code: {response.status_code}")
        elif response.content == b"[]":
            raise RuntimeError(f"Response is empty: {response.content.decode()}")

        # Turn response into dict
        response = response.json()
        return response

    def _process_rmsd_profile(self, response: dict) -> np.ndarray:
        """Load numpy array from string."""
        assert "profile" in response.keys(), f"Response does not contain 'profile' key: {response}"
        profile = np.fromstring(response["profile"][1:-1], sep=",")
        if len(profile) == 0:
            raise RuntimeError(f"RMSD profile is empty.")
        return profile

    def query_pdb_stats(
        self, pdb_id: str, chain_id: Optional[str] = None
    ) -> Union[dict, List[dict]]:
        """Request flexibility data about one particular PDB.

        NOTE: you can omit the chainID and PDBFlex will return information for all chains.
        """
        url = self.BASE_URL + "PDBStats.php?"
        url += f"pdbID={pdb_id}"
        if chain_id is not None:
            url += f"&chainID={chain_id}"

        return self._handle_query(url) if chain_id is None else self._handle_query(url)[0]

    def query_rmsd_profile(self, pdb_id: str, chain_id: str) -> dict:
        """Request RMSD array used for local flexibility plots"""
        url = self.BASE_URL + "rmsdProfile.php?"
        url += f"pdbID={pdb_id}&chainID={chain_id}"

        response = self._handle_query(url)
        response["profile"] = self._process_rmsd_profile(response)

        return response

    def query_representative_structures(self, pdb_id: str, chain_id: str) -> List[str]:
        """Request representatives for a PDB's own cluster. Returns a list of chains that represent the most distinct structures in the cluster."""
        url = self.BASE_URL + "representatives.php?"
        url += f"pdbID={pdb_id}&chainID={chain_id}"

        return self._handle_query(url)

    def query_rmsd_for_sequence(self, sequence: str) -> dict:
        """Request RMSD array used for local flexibility plots but for any protein sequence currently not analyzed by PDBFlex via POST request."""
        url = self.BASE_URL + "sequence.php"

        response = requests.post(url, data={"sequence": sequence})
        response = response.json()
        response["profile"] = self._process_rmsd_profile(response)

        return response