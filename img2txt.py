from langchain.document_loaders import ImageCaptionLoader
from langchain.embeddings import GPT4AllEmbeddings
import os

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from loadLLM import LoadLLM

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

embedding = GPT4AllEmbeddings()

# add file in folder assets to list_image_urls. Each entry is a path to an image
list_image_urls = []
for filename in os.listdir(os.path.abspath(os.path.join('assets'))):
    list_image_urls.append(os.path.abspath(os.path.join('assets', filename)))


# list_image_urls = [
#     os.path.abspath(os.path.join('assets', 'Messier83_-_Heic1403a.jpg')),
#     os.path.abspath(os.path.join('assets', 'Hyla_japonica_sep01.jpg')),
#     os.path.abspath(os.path.join('assets', '4539-le-grand-bal-masque-du-chateau-de-versai-diaporama_big-1.jpg')),
# ]

# loader = ImageCaptionLoader(images=list_image_urls)

# documents = loader.load()

# # #Parcourir et afficher les légendes et les métadonnées
# # for doc in documents:
# #    print("Légende:", doc.page_content)
# #    print("Métadonnées:", doc.metadata)

# vectordb = FAISS.from_documents(documents, embedding)

# #docs = vectordb.similarity_search(query)
# # print(docs[0].page_content)
# # print(docs[0].metadata)
# vectordb.save_local("image_faiss_index")

vectordb = FAISS.load_local("image_faiss_index", embedding)

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

image_name = ResponseSchema(name="name",
                             description="image name")
image_caption = ResponseSchema(name="caption",
                                      description="image caption")

test = ResponseSchema(
             name="images",
             description="""array of of images informations in the following format: [
     { "name": string // image name',  "caption": string // image caption' }]""")

response_schemas = [test]
#response_schemas = [image_name, image_caption]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()




template_string = """
Use the following context delimited by backticks
context: ```{context}```    
and extract all objects which are related to the text :"{question}". 
Don't explain how to extract data, just extract data.
{format_instructions}
"""

#prompt_template = PromptTemplate(template=template_string, 
#    input_variables=["context"],
#    partial_variables={"format_instructions"})
#prompt_template.partial(format_instructions=format_instructions)
prompt_template = PromptTemplate(template=template_string, input_variables=['context', 'question'], partial_variables={"format_instructions": format_instructions} )
#messages = prompt_template.format(format_instructions=format_instructions,context="context")

# print (prompt_template.format(context='{\
#     "page_content": "an image of a frog on a flower [SEP]",\
#     "metadata": {\
#         "image_path": "C:\\Users\\Shadow\\Documents\\aiFilterPhoto\\assets\\Hyla_japonica_sep01.jpg"\
#     },\
#     "type": "Document"\
# }', question="galaxy"))
# exit()

def format_docs(docs):
    output = "["
    for doc in docs:
        #output += '{ "caption": "'+doc.page_content+'", "name": "'+r"{}".format(doc.metadata['image_path'])+'" }' 
        output += '{ "caption": "'+doc.page_content+'", "name": "'+doc.metadata['image_path'].replace("\\", "/")+'" }' 
    output += "]"
    return output


# test = vectordb.as_retriever(search_kwargs={'k':1})  
# toto = test.get_relevant_documents("galaxy")

# print( toto)
#exit()
llm = LoadLLM(isLMStudio=True).llm
#qa_chain = RetrievalQA.from_chain_type(
#    llm,
#    retriever=vectordb.as_retriever()
#)
#qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(search_kwargs={'k':1}) ,  chain_type_kwargs={"prompt": prompt_template})
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
chain = (
    {"context": vectordb.as_retriever(search_kwargs={'k':3}) | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | output_parser
)

# export function to query the index
def query_index(query, vectorSearchOnly=False):
    #output_dict = output_parser.parse(qa_chain({"query":query})['result'])
    if vectorSearchOnly:
        return vectordb.similarity_search(query)
    print("query: ", query)
    return chain.invoke(query)
