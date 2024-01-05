from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
loader = DirectoryLoader('./articles', glob="*.txt", loader_cls=TextLoader)
documents = loader.load()
print(len(documents))
# 1000 characters per chunk, 200 characters overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

print(len(texts))
print(texts[2:4])

persist_directory = 'db'

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=texts, 
    embedding=embedding,
    persist_directory=persist_directory)

vectordb.persist()
vectordb = None

vectordb = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embedding)

retriever = vectordb.as_retriever()

docs = retriever.get_relevant_documents("What is Generative AI?")

for doc in docs:
    print(doc.metadata["source"])

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(), 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True)

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

query = "겨울 계절 학기 교육 봉사 서류 제출 마감일은 언제인가요?"
llm_response = qa_chain(query)
process_llm_response(llm_response)