from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings

# TODO regarder les differents embeddings
embedding = GPT4AllEmbeddings()

list_image_urls = []
for filename in os.listdir(os.path.abspath(os.path.join('assets'))):
    list_image_urls.append(os.path.abspath(os.path.join('assets', filename)))


reresponse_schema = ResponseSchema(
             name="images",
             description="""array of of images informations in the following format: [
     { "caption": string // image caption' }]""")

response_schemas = [reresponse_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

vectordb = FAISS.load_local("image_faiss_index", embedding)

def _save_vector_db(vectordb):
    vectordb.save_local("image_faiss_index")

def vector_query_index(query):
    return vectordb.similarity_search(query, k=100)