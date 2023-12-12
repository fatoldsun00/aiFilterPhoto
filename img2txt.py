from langchain.document_loaders import ImageCaptionLoader
from langchain.embeddings import GPT4AllEmbeddings
import os

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from loadLLM import LoadLLM

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

embedding = GPT4AllEmbeddings()

list_image_urls = [
    os.path.abspath(os.path.join('assets', 'Messier83_-_Heic1403a.jpg')),
    os.path.abspath(os.path.join('assets', 'Hyla_japonica_sep01.jpg')),
    os.path.abspath(os.path.join('assets', '4539-le-grand-bal-masque-du-chateau-de-versai-diaporama_big-1.jpg')),
]

# loader = ImageCaptionLoader(images=list_image_urls)

# documents = loader.load()

# #Parcourir et afficher les légendes et les métadonnées
# for doc in documents:
#    print("Légende:", doc.page_content)
#    print("Métadonnées:", doc.metadata)

# vectordb = FAISS.from_documents(documents, embedding)

# docs = vectordb.similarity_search(query)
# print(docs[0].page_content)
# print(docs[0].metadata)
vectordb = FAISS.load_local("image_faiss_index", embedding)



vectordb.save_local("image_faiss_index")

llm = LoadLLM(isLMStudio=True).llm
#qa_chain = RetrievalQA.from_chain_type(
#    llm,
#    retriever=vectordb.as_retriever()
#)
 

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

image_name = ResponseSchema(name="name",
                             description="image name\
                             String")
image_caption = ResponseSchema(name="caption",
                                      description="image caption\
                                      String")


response_schemas = [image_name, 
                    image_caption,
                    ]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()




def promptTemplate(format_instructions):
    template_string = """
    Use the following pieces of context, delimited by backticks, as a JSON.\ 
    The format of the JSON is as follows and represent an image path and its caption:
    ***
    {{
        "page_content": "an image of a frog on a flower [SEP]",
        "metadata": {{
            "image_path": "C:\\Users\\Shadow\\Documents\\aiFilterPhoto\\assets\\Hyla_japonica_sep01.jpg"
        }},
        "type": "Document"
    }}
    ***
    From this context extract most revelant image for the query.
    ```{context}```    
    {format_instructions}
    """
    prompt_template = PromptTemplate(template=template_string, input_variables=["format_instructions","context"])
    #prompt_template.format(format_instructions=format_instructions)
    return prompt_template


qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(search_kwargs={'k':1}),  chain_type_kwargs={"prompt": promptTemplate(format_instructions)})
  

# export function to query the index
def query_index(query):
    output_dict = output_parser.parse(qa_chain({"query":query})['result'])
    return output_dict
