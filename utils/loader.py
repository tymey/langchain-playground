from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

loader = PyPDFLoader("resume.pdf")
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
chunks = splitter.split_documents(docs)