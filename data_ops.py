# -*- coding: utf-8 -*-

import os

from transformers import GPT2TokenizerFast

from langchain_text_splitters.character import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

from langchain_community.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    CohereEmbeddings,
)
from langchain_community.vectorstores import FAISS, Chroma, DocArrayInMemorySearch

# TODO: Replace file reader from llama-index with one from langchain
from typing import List, Tuple, Sequence, Callable, Optional, Dict
from langchain.schema import Document

from functools import lru_cache

from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

from langchain_community.document_loaders import (
    PDFMinerLoader,
    PyPDFDirectoryLoader,
    Docx2txtLoader,
    TextLoader,
    DirectoryLoader,
)


__all__ = ["DocumentProcessor", "DocumentLoader"]


class DocumentLoader:
    def __init__(self, chunk_size=1000, chunk_overlap=10):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._loaders = {
            ".docx": Docx2txtLoader,
            ".doc": Docx2txtLoader,
            ".pdf": PDFMinerLoader,
            ".csv": TextLoader,
        }

        self.text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def load_file(self, file_path):
        ext = os.path.splitext(file_path)[-1]
        return self._loaders.get(ext, TextLoader)(file_path).load()

    @lru_cache(maxsize=128)
    def from_directory(self, file_directory):
        chunks = []

        file_list = os.listdir(file_directory)

        file_paths = [os.path.join(file_directory, f) for f in file_list]
        _ = [chunks := chunks + self.load_file(f) for f in file_paths]

        print("While processing:", len(chunks))

        split_documents = self.text_splitter.split_documents(chunks)

        print("After splitting:", len(split_documents))

        return split_documents

    def from_files(self, file_paths, base_dir=None):
        if base_dir is not None:
            file_paths = [os.path.join(base_dir, f) for f in file_paths]

        chunks = []

        _ = [chunks := chunks + self.load_file(f) for f in file_paths]

        print("While processing:", len(chunks))

        split_documents = self.text_splitter.split_documents(chunks)

        print("After splitting:", len(split_documents))

        return split_documents


class DocumentProcessor:
    PERSIST_DIRECTORY = "./vector_database/"
    num_chunks = 0

    def __init__(
        self,
        tokenizer_class: Callable = GPT2TokenizerFast,
        tokenizer_model: str = "gpt2",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        embedding_model: str = "msmarco-bert-base-dot-v5",
        embedding_class: Callable = HuggingFaceEmbeddings,
        text_splitter_class: Callable = RecursiveCharacterTextSplitter,
        database_class: Callable = Chroma,
        chunks: bool = False,
        as_retriever: bool = False,
        compress_retriever: bool = False,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.database_class = database_class
        self.as_retriever = as_retriever
        self.compress_retriever = compress_retriever

        self.chunks = chunks

        self.text_tokenizer = None
        self.text_splitter = None
        self.text_embedder = None

        # self.pdf_reader = PDFReader(return_full_document=True)
        self.pdf_reader = None

        self.tokenizer = [tokenizer_model, tokenizer_class]
        self.splitter = [text_splitter_class, chunk_size, chunk_overlap, None]
        self.embedder = [embedding_model, embedding_class]

        self.document_loader = DocumentLoader(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        if os.path.exists(self.PERSIST_DIRECTORY):
            self.vector_database = Chroma(
                persist_directory=self.PERSIST_DIRECTORY,
                embedding_function=self.embedder,
            )
            print(f"Vector database loaded from {self.PERSIST_DIRECTORY}!")
        else:
            self.vector_database = None

    @property
    def tokenizer(self):
        return self.text_tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer_info):
        tokenizer_model, tokenizer_class = tokenizer_info
        self.text_tokenizer = tokenizer_class.from_pretrained(tokenizer_model)
        return

    @property
    def splitter(self):
        return self.text_splitter

    @splitter.setter
    def splitter(self, splitter_info):
        text_splitter_class, chunk_size, chunk_overlap, tokenizer = splitter_info
        if tokenizer is None:
            tokenizer = self.text_tokenizer

        self.text_splitter = text_splitter_class.from_huggingface_tokenizer(
            tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return

    @property
    def embedder(self):
        return self.text_embedder

    @embedder.setter
    def embedder(self, embedder_info):
        embedding_model, embedding_class = embedder_info
        self.text_embedder = embedding_class(model_name=embedding_model)
        return

    def generate_chunks_from_directory(self, directory):
        return self.document_loader.from_directory(directory)

    def generate_chunks_from_file(self, file_path):
        return self.document_loader.load_file(file_path)

    def generate_chunks_from_files(self, file_paths, base_dir=None):
        return self.document_loader.from_files(file_paths=file_paths, base_dir=base_dir)

    def embed_chunks(self, chunks):
        return [self.embed_chunk(chunk) for chunk in chunks]

    def embed_chunk(self, chunk):
        return self.text_embedder.embed_query(
            chunk.page_content if not isinstance(chunk, str) else chunk
        )

    def _embeddings_from_documents(self, documents):
        documents_ = [
            (p.page_content if isinstance(p, Document) else p) for p in documents
        ]
        return self.text_embedder.embed_documents(documents_)

    # @lru_cache(maxsize=128)
    def generate_embeddings(
        self, documents, search_type="mmr", k=2, fetch_k=4, update_database=True
    ):
        if self.vector_database is None:
            self.vector_database = self._configure_vector_database(
                documents=documents,
                search_type=search_type,
                k=k,
                fetch_k=fetch_k,
            )

        if update_database:
            self._update_vector_database(documents=documents)

        return self._embeddings_from_documents(documents=documents)

    def _update_vector_database(self, documents):
        try:
            self.vector_database.add_documents(documents=documents)
            self.num_chunks += 1
            exit_code = 0
        except:
            exit_code = 1
            pass

        return exit_code

    def _configure_vector_database(
        self, documents, search_type="mmr", k=2, fetch_k=4, similarity_threshold=0.76
    ):
        vector_database = Chroma.from_documents(
            documents=documents,
            embedding=self.embedder,
            persist_directory=self.PERSIST_DIRECTORY,
        )

        if self.as_retriever:
            vector_database = vector_database.as_retriever(
                search_type=search_type, search_kwargs={"k": k, "fetch_k": fetch_k}
            )

            if self.compress_retriever:
                embeddings_filter = EmbeddingsFilter(
                    embeddings=self.embedder, similarity_threshold=similarity_threshold
                )
                return ContextualCompressionRetriever(
                    base_compressor=embeddings_filter, base_retriever=vector_database
                )

        return vector_database

    def simple_similarity_search(self, query, k=4):
        return self.vector_database.similarity_search(query, k=k)

    def relevance_search(self, query, top_k=1, total_k=10):
        return self.vector_database.max_marginal_relevance_search(
            query, k=top_k, fetch_k=total_k
        )

    def similarity_search(self, query, mode=0, k=4, top_k=1, total_k=10):
        """Perform similarity search in different modes. Mode 0 for simple search. Model 1 for relevance similarity search."""
        assert mode in range(0, 2), "Pass in `mode` in [0, 1]."
        if mode == 0:
            results = self.simple_similarity_search(query=query, k=k)
        elif mode == 1:
            results = self.relevance_search(query=query, top_k=top_k, total_k=total_k)
        else:
            results = [query]

        return results

    def paired_similarity_search(self, queries, mode=0, k=4, top_k=1, total_k=10):
        return [
            self.similarity_search(
                query=q, mode=mode, k=k, top_k=top_k, total_k=total_k
            )
            for q in queries
        ]
