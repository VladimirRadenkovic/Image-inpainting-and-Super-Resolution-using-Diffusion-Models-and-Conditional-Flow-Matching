import os
import urllib.request as request
from functools import cached_property, lru_cache
from typing import Iterable, List

import requests
from loguru import logger

PDB_AVAILABILITY_URL = "https://data.rcsb.org/rest/v1/holdings/current/entry_ids"


@lru_cache()
def get_updated_pdb(pdb_code: str) -> str:
    """Check the status of a pdb, if it is obsolete return the superceding PDB ID else return None"""
    pdb_code = pdb_code.lower()
    r = requests.get(f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/status/{pdb_code}").json()
    if r[pdb_code][0]["status_code"] == "OBS":
        new_pdb_code = r[pdb_code][0]["superceded_by"][0]
        return new_pdb_code
    return pdb_code


def update_obsolete_pdb_ids(pdb_ids: Iterable[str]) -> List[str]:
    """Update obsolete PDB IDs to superceding PDB IDs"""
    available_pdbs = set(map(str.lower, requests.get(PDB_AVAILABILITY_URL).json()))
    outdated_pdbs = set(map(str.lower, pdb_ids)) - available_pdbs

    if len(outdated_pdbs) == 0:
        logger.info("No obsolete PDB IDs found")
        return pdb_ids

    logger.info(f"Found {len(outdated_pdbs)} obsolete PDB IDs")
    updated_pdb_ids = []
    failed = []
    for pdb_id in pdb_ids:
        if pdb_id in outdated_pdbs:
            try:
                updated_pdb_id = get_updated_pdb(pdb_id).lower()
                assert updated_pdb_id in available_pdbs, "Updated PDB ID is not available"
                updated_pdb_ids.append(updated_pdb_id)
            except Exception:
                logger.info(f"Failed to update obsolete PDB ID {pdb_id}")
                failed += [pdb_id]
                updated_pdb_ids.append("")
        else:
            updated_pdb_ids.append(pdb_id)
    logger.info(f"Updated {len(outdated_pdbs)} PDB IDs. {len(failed)} failed.")

    return updated_pdb_ids


@lru_cache()
def get_pdb_entities(pdb_code: str):
    """
    Tries to fetch the macromolecular entities of a PDB code from the EBI
    else returns None. See https://www.rcsb.org/docs/general-help/identifiers-in-pdb
    """
    pdb_code = pdb_code.lower()
    response = requests.get(f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{pdb_code}")
    entities = response.json()[pdb_code]
    return entities


@lru_cache()
def match_pdb_chain_to_entity(pdb_code: str, chain_id: str):
    """
    Tries to match a PDB file chain ID to an macromolecular entitiy, else returns None.
    See https://www.rcsb.org/docs/general-help/identifiers-in-pdb
    """

    pdb_code = pdb_code.lower()
    chain_id = chain_id.upper()

    # Get PDB entities
    entities = get_pdb_entities(pdb_code)

    for entity in entities:
        # Check if chain present in entity and protein then return 'entity_id'
        if chain_id in entity["in_chains"] and entity["molecule_type"] == "polypeptide(L)":
            return entity["entity_id"]
    else:
        return None


class RcsbPdbClusters:
    def __init__(self, identity: int = 30, cluster_dir="."):
        """Get PDB clusters from the RCSB PDB website using RCSB mmseq2/blastclust predefined clusters.
        Clusters info at https://www.rcsb.org/docs/programmatic-access/file-download-services

        Args:
                identity (int, optional): % sequence identity threshold for chain clusters. Defaults to 30.
                cluster_dir (str, optional): directory to store the downloaded cluster files. Defaults to '.'.
        """
        self.cluster_dir = cluster_dir
        self.identity = identity
        self.clusters = {}
        self._fetch_cluster_file()

    def _download_cluster_sets(self, cluster_file_path):
        """Download cluster file from RCSB PDB website"""
        os.makedirs(os.path.dirname(cluster_file_path), exist_ok=True)

        # Note that the files changes frequently as do the ordering of cluster within
        request.urlretrieve(
            f"https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{self.identity}.txt",
            cluster_file_path,
        )

    def _fetch_cluster_file(self):
        """Load cluster file if found else download and load"""

        cluster_file_path = os.path.join(self.cluster_dir, f"pdb_clusters_{self.identity}.txt")
        logger.info(f"cluster file path: {cluster_file_path}")

        # Fetch cluster file
        if not os.path.exists(cluster_file_path):
            logger.info("Downloading new cluster file...")
            logger.warning(
                "This might lead to silent incompatibilities with any old 'pdbcode_mappings.pickle' files! Please better remove those manually."
            )
            self._download_cluster_sets(cluster_file_path)

        # Extract cluster IDs
        for n, line in enumerate(open(cluster_file_path, "rb")):
            for id in line.decode("ascii").split():
                self.clusters[id] = n

    def get_seqclust(
        self,
        pdb_code: str,
        *,
        entity_id: str = None,
        chain_id: str = None,
        check_obsolete: bool = True,
    ):
        """Get sequence cluster ID for a pdb_code chain using RCSB mmseq2/blastclust predefined clusters
        Returns None if no cluster found.
        Give either entity_id or chain_id.

        Args:
                pdb_code (str): PDB code
                entity_id (str, optional): PDB entity ID. Defaults to None.
                chain_id (str, optional): PDB chain ID. Defaults to None.
                check_obsolete (bool, optional): Check if PDB is obsolete and return superceding PDB ID. Defaults to True

        Returns:
                int: Sequence cluster ID
        """
        try:
            # Fetch entity_id if only given chain id
            if chain_id:
                assert entity_id == None, "Only define either `chain_id` or `entity_id`"
                entity_id = match_pdb_chain_to_entity(pdb_code, chain_id)
            if entity_id == None:
                assert chain_id != None, "Define either `chain_id` or `entity_id`"
                raise RuntimeError(
                    f"Unable to match to entity_id for {pdb_code}_{chain_id}. The chain might not be a protein."
                )

            # Make query and get cluster ID
            query_str = f"{pdb_code.upper()}_{str(entity_id).upper()}"  # e.g. 1ATP_I
            return self.clusters[query_str]

        except KeyError as e:
            if check_obsolete:
                new_pdb_code = get_updated_pdb(pdb_code)
                if new_pdb_code != pdb_code:
                    logger.warning(
                        f"Assigning cluster for obsolete entry via superceding: {pdb_code}->{new_pdb_code} {chain_id}"
                    )
                    return self.get_seqclust(
                        new_pdb_code,
                        chain_id=chain_id,
                        entity_id=entity_id,
                        check_obsolete=False,
                    )
            raise e

    @cached_property
    def cluster_ids(self) -> set:
        """Get all cluster IDs"""
        return set(self.clusters.values())

    @cached_property
    def n_clusters(self) -> int:
        """Get number of clusters"""
        return len(self.cluster_ids)

    def __repr__(self) -> str:
        cluster_file = os.path.abspath(self.cluster_dir) + f"/pdb_clusters_{self.identity}.txt"
        return f"RcsbPdbClusters(identity={self.identity}%, cluster_file='{cluster_file}', n_clusters={self.n_clusters:,})"