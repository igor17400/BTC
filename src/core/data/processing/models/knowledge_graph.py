import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import normalize
from tqdm import tqdm

logger = logging.getLogger(__name__)


class KnowledgeGraphProcessor:
    def __init__(
        self,
        cache_dir: Path,
        dataset_path: Path,
        max_entities: int = 100000,
        max_relations: int = 1000000,
    ):
        self.cache_dir = cache_dir
        self.dataset_path = dataset_path
        self.max_entities = max_entities
        self.max_relations = max_relations
        self.graph = nx.Graph()  # Using undirected graph like in the example
        self.entity_embeddings = {}
        self.context_embeddings = {}

    def find_wikidata_id(self, name: str) -> str:
        """Find Wikidata ID for a given name."""
        if not name:
            return "entityNotFound"

        # Wikidata API endpoint
        url = "https://www.wikidata.org/w/api.php"

        # Parameters for the API call
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": name,
            "type": "item",
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if "search" in data and len(data["search"]) > 0:
                return data["search"][0]["id"]
            return "entityNotFound"
        except Exception as e:
            logger.error(f"Error finding Wikidata ID for {name}: {str(e)}")
            return "entityNotFound"

    def query_entity_links(self, entity_id: str) -> dict:
        """Query entity links from Wikidata."""
        if entity_id == "entityNotFound":
            return {}

        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": entity_id,
            "languages": "en",
            "props": "claims",
        }

        try:
            response = requests.get(url, params=params)
            return response.json()
        except Exception as e:
            logger.error(f"Error querying entity links for {entity_id}: {str(e)}")
            return {}

    def read_linked_entities(self, json_links: dict) -> list[tuple[str, str]]:
        """Extract linked entities from JSON response."""
        if not json_links or "entities" not in json_links:
            return []

        linked_entities = []
        for _entity_id, entity_data in json_links["entities"].items():
            if "claims" in entity_data:
                for _prop_id, claims in entity_data["claims"].items():
                    for claim in claims:
                        if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
                            value = claim["mainsnak"]["datavalue"]["value"]
                            if isinstance(value, dict) and "id" in value:
                                linked_entities.append(
                                    (value["id"], value.get("label", ""))
                                )

        return linked_entities

    def search_wikidata(self, names: list[str], extras: dict = None) -> pd.DataFrame:
        """Search Wikidata for a list of names."""
        results_list = []

        for name in tqdm(names, desc="Searching Wikidata"):
            # Get Wikidata ID
            entity_id = self.find_wikidata_id(name)
            if entity_id == "entityNotFound":
                continue

            # Get entity links
            json_links = self.query_entity_links(entity_id)
            related_links = self.read_linked_entities(json_links)

            # Add to results
            for related_entity, related_name in related_links:
                result = {
                    "name": name,
                    "original_entity": entity_id,
                    "linked_entities": related_entity,
                    "name_linked_entities": related_name,
                }

                # Add extra information if provided
                if extras:
                    for key, value in extras.items():
                        if name in value:
                            result[key] = value[name]

                results_list.append(result)

        return pd.DataFrame(results_list)

    def build_knowledge_graph(self, results_df: pd.DataFrame) -> None:
        """Build knowledge graph from search results."""
        logger.info("Building knowledge graph...")

        # Create graph from edge list
        self.graph = nx.from_pandas_edgelist(
            results_df, "original_entity", "linked_entities"
        )

        # Create name mapping
        target_names = results_df[
            ["linked_entities", "name_linked_entities"]
        ].drop_duplicates()
        target_names = target_names.rename(
            columns={"linked_entities": "labels", "name_linked_entities": "name"}
        )

        source_names = results_df[["original_entity", "name"]].drop_duplicates()
        source_names = source_names.rename(columns={"original_entity": "labels"})

        names = pd.concat([target_names, source_names])
        names = names.set_index("labels")
        self.name_mapping = names.to_dict()["name"]

        logger.info(
            f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
        )

    def generate_embeddings(self) -> None:
        """Generate entity and context embeddings."""
        logger.info("Generating embeddings...")

        # Generate random embeddings for entities
        for node in self.graph.nodes():
            self.entity_embeddings[node] = np.random.randn(100)

        # Normalize entity embeddings
        entity_embeddings_matrix = np.array(list(self.entity_embeddings.values()))
        normalized_embeddings = normalize(entity_embeddings_matrix)

        for i, node in enumerate(self.graph.nodes()):
            self.entity_embeddings[node] = normalized_embeddings[i]

        # Generate context embeddings
        for node in self.graph.nodes():
            # Get neighbors
            neighbors = list(self.graph.neighbors(node))
            if not neighbors:
                self.context_embeddings[node] = self.entity_embeddings[node]
                continue

            # Average neighbor embeddings
            context_embedding = np.zeros(100)
            for neighbor in neighbors:
                context_embedding += self.entity_embeddings[neighbor]
            context_embedding /= len(neighbors)

            self.context_embeddings[node] = context_embedding

        # Save embeddings
        self._save_embeddings()

        logger.info("Generated and saved embeddings")

    def _save_embeddings(self) -> None:
        """Save entity and context embeddings to files."""
        # Save entity embeddings
        for mode in ["train", "dev", "test"]:
            entity_file = self.dataset_path / mode / "entity_embedding.vec"
            context_file = self.dataset_path / mode / "context_embedding.vec"

            with open(entity_file, "w", encoding="utf-8") as f:
                for entity, embedding in self.entity_embeddings.items():
                    f.write(entity + "\t" + "\t".join(map(str, embedding)) + "\n")

            with open(context_file, "w", encoding="utf-8") as f:
                for entity, embedding in self.context_embeddings.items():
                    f.write(entity + "\t" + "\t".join(map(str, embedding)) + "\n")

    def process(self, news_titles: list[str]) -> None:
        """Process the knowledge graph data."""
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Search Wikidata for news titles
        logger.info("Searching Wikidata for news titles...")
        results_df = self.search_wikidata(news_titles)

        # Build knowledge graph
        self.build_knowledge_graph(results_df)

        # Generate embeddings
        self.generate_embeddings()

        logger.info("Knowledge graph processing complete")
