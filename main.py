#fastapi main file
#fastapi is a python framework for building apis
#this way we can perfom query without change the code

from fastapi import FastAPI, HTTPException
from img2txt import query_index
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
async def root():
    return {"Hello": "World"}



class QueryItem(BaseModel):
    query: str
    fastSearch: bool = False

@app.post("/query")
async def post_query(item: QueryItem):
    if not item.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return query_index(item.query, item.fastSearch)