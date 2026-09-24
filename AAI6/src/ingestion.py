from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(data_path="Datasets"):
    """
    Load all .txt documents from the Datasets folder.
    """

    documents = SimpleDirectoryReader(
        input_dir=data_path,
        required_exts=[".txt"]
    ).load_data()

    return documents


def create_index(data_path="Datasets"):
    """
    Load documents, split them into chunks,
    generate embeddings and create vector index.
    """

    documents = load_documents(data_path)

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    nodes = splitter.get_nodes_from_documents(documents)

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    Settings.embed_model = embed_model

    index = VectorStoreIndex(nodes)

    return index