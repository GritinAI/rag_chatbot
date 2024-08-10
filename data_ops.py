# -*- coding: utf-8 -*-

import PyPDF3 as pypdf

from transformers import GPT2TokenizerFast
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, CohereEmbeddings

from langchain_community.vectorstores import FAISS

from llama_index.readers.file import PDFReader
from llama_index.core.schema import MetadataMode

from typing import List, Tuple, Sequence, Callable, Optional, Dict
from langchain.docstore.document import Document


__all__ = [
    "DocumentProcessor",
    "load_pdf",
]


class DocumentProcessor:
    def __init__(
        self,
        tokenizer_class: Callable = GPT2TokenizerFast,
        tokenizer_model: str = "gpt2",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        embedding_model: str = "msmarco-bert-base-dot-v5",
        embedding_class: Callable = HuggingFaceEmbeddings,
        text_splitter_class: Callable = RecursiveCharacterTextSplitter,
        database_class: Callable = FAISS,
        chunks: bool = False,
        as_retreiver: bool = False,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.database_class = database_class
        self.as_retreiver = as_retreiver

        self.chunks = chunks

        self.text_tokenizer = None
        self.text_splitter = None
        self.text_embedder = None
        self.vector_database = None

        self.pdf_reader = PDFReader(return_full_document=True)

        self.tokenizer = [tokenizer_model, tokenizer_class]
        self.splitter = [text_splitter_class, chunk_size, chunk_overlap, None]
        self.embedder = [embedding_model, embedding_class]

    def load_pdf(self, file_path: str):
        return self.pdf_reader.load_data(file=file_path)[0]

    @property
    def tokenizer(self):
        return self.text_tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer_info: List[str, Callable]):
        tokenizer_model, tokenizer_class = tokenizer_info
        self.text_tokenizer = tokenizer_class.from_pretrained(tokenizer_model)
        return

    @property
    def splitter(self):
        return self.text_splitter

    @splitter.setter
    def splitter(self, splitter_info: List[Callable, int, int, Callable]):
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
    def embedder(self, embedder_info: List[str, Callable]):
        embedding_model, embedding_class = embedder_info
        self.text_embedder = embedding_class(model_name=embedding_model)
        return

    def split_document(self, document: str | Document):
        document_ = document.page_content if isinstance(document, Document) else document
        document_chunks = self.splitter.split_text(document_)
        # return document_chunks
        return self.splitter.create_documents(document_chunks)

    # def split_documents(self, documents):
    #     return self.splitter.create_documents(documents)

    def split_documents(self, documents: Sequence[str | Document]):
        # documents_ = [(d.page_content if isinstance(d, Document) else d) for d in documents]
        return [(self.splitter.create_documents([d])[0] if not isinstance(d, Document) else d) for d in documents]

    def split_text(self, documents):
        return (
            self.split_documents(documents=documents)
            if self.chunks
            else self.split_document(document=documents)
        )

    def embed_chunks(self, chunks):
        return [self.embed_chunk(chunk) for chunk in chunks]

    def embed_chunk(self, chunk):
        return self.text_embedder.embed_query(
            chunk.page_content if not isinstance(chunk, str) else chunk
        )

    def create_embeddings(self, documents):
        self.splitter.create_embeddings(documents)

    def persist_embeddings(self, documents: Document | Sequence[str | Document], metadata_mode: Callable = MetadataMode.EMBED):
        if isinstance(documents, list):
            documents_ = [d.get_content(metadata_mode=metadata_mode) for d in documents]
        else:
            documents_ = documents.get_content(metadata_mode=metadata_mode)

        documents_ = self.split_text(documents=documents_)
        embeddings = self.embed_chunks(documents_)

        # print("Number of documents:", len(documents_))
        # print("Number of embeddings:", len(embeddings))
        # print(embeddings)

        print(dir(self.text_embedder))

        print(type(documents_[0]), dir(documents_[0]))

        # print(documents[0])

        self.vector_database = self.database_class.from_documents(documents=documents_, embedding=embeddings)

        if self.as_retreiver:
            self.vector_database = self.vector_database.as_retreiver()

        print("Embeddings generated and persisted.")

        return

    # def persist_embeddings(self, documents, embeddings):
    #     self.vector_database = self.database_class.from_documents(documents, embeddings)

    #     if self.as_retreiver:
    #         self.vector_database = self.vector_database.as_retreiver()

    #     print("Embeddings generated and persisted.")

    #     return

    # def process_documents(self, documents):
    #     documents = self.split_text(documents=documents)
    #     embeddings = self.embed_chunks(documents)
        
    #     self.persist_embeddings(documents, embeddings)

    #     return

    def simple_similarity_search(self, query):
        return self.vector_database.similarity_search(query)

    def relevance_search(self, query, top_k=1, total_k=10):
        return self.vector_database.max_marginal_relevance_search(
            query, k=top_k, fetch_k=total_k
        )

    def similarity_search(self, query, mode=0, top_k=1, total_k=10):
        if mode == 0:
            results = self.simple_similarity_search(query=query)
        elif mode == 1:
            results = self.relevance_search(query=query, top_k=top_k, total_k=total_k)
        else:
            results = [query]

        return results

    def paired_similarity_search(self, queries, mode=0, top_k=1, total_k=10):
        return [
            self.similarity_search(query=q, mode=mode, top_k=top_k, total_k=total_k)
            for q in queries
        ]


def load_pdf(fpath):
    with open(fpath, "rb") as f:
        pdf = pypdf.PdfFileReader(f)
        text = str()
        for page_num in range(pdf.numPages):
            page = pdf.getPage(page_num)
            text = text + " " + page.extractText()

    return text


if __name__ == "__main__":
    processor = DocumentProcessor()
    loaded_document = processor.load_pdf("data/file.pdf")
    # print(loaded_document.get_content(metadata_mode = MetadataMode.EMBED))
    # print(loaded_document.get_content(metadata_mode=MetadataMode.LLM))
    print([d for d in dir(loaded_document) if "page" in d])

    print(processor.persist_embeddings(loaded_document, metadata_mode=MetadataMode.LLM))
