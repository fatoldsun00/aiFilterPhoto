# aiFilterPhoto
Analyse the photos in a folder and outpout photo which match request

# Install

First install python, then we will use pip to install requirements

`pip install langchain uvicorn fastapi pydantic`

install faiss-gpu (nvidia with cuda supported) or faiss-cpu

`pip install faiss-gpu`

# Select a llm

loadLLM is a class used for load llm from file or from LM studio.
LM studio is a gui for dwl and load model, use ll, through a chat or launch a web serveur with openAI API structure. Give it a try !

For llm i advise to use stableLM 3B zephyr which is efficient and light

# Launch app

`python -m uvicorn main:app --reload`

which launch web server.

POST /query {"query": "ma query"}