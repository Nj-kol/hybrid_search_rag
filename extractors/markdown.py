from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MarkdownLoader:
    def __init__(self):
        pass

    def load_docs(self, directory):
        loader = DirectoryLoader(directory, glob="**/*.md",  show_progress=True, use_multithreading=True)
        documents = loader.load()
        return documents
    
    def split_docs(self,documents,chunk_size=500,chunk_overlap=20):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)
        return docs