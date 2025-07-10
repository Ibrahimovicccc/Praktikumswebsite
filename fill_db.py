from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  
import chromadb

# setting the environment
#Hier werden die Ordner erstellt
DATA_PATH = r"data"   
CHROMA_PATH = r"chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(name="Abi-Vorgaben")

# loading the document

loader = PyPDFDirectoryLoader(DATA_PATH)

raw_documents = loader.load()

# splitting the document die Zahlen müssen noch angepasst werden bei chunck size wie viel sich überlappendarf und die länge

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(raw_documents)
print(chunks)
# preparing to be added in chromadb

documents = []
metadata = []  #Hiermit kann der Rag die Source dir geben
ids = []  #Nicht immer notwendig wird benötigtz wenn man die datenbank editieren will

i = 0

for chunk in chunks:
    documents.append(chunk.page_content)
    ids.append("ID"+str(i))
    metadata.append(chunk.metadata)

    i += 1
#Hier wird alles geladen so das chromapd damit arbeiten kann
# adding to chromadb


collection.upsert(
    documents=documents,
    metadatas=metadata,
    ids=ids
)
#Hier kriegt die ai die Dokumente