from typing import List, Optional, Union
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import format_tool_to_openai_function
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langserve import add_routes
from pydantic import BaseModel
from pydantic import Field
from app.set_prompt import prompt, rag_prompt, agent_prompt
from app.set_model import llm
from app.set_documents import vectorstore
from app.set_tools import tools

basic_chain = (prompt
         | llm 
         | StrOutputParser()
)

rag_chain = ({"context" : vectorstore.as_retriever(search_kwargs={'k': 5,
                                                    }),
              "question" : RunnablePassthrough()}
         | rag_prompt
         | llm 
         | StrOutputParser()
)

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

# agent_chain = ({

# })


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

#-------------- routes -----------------

add_routes(
    app,
    basic_chain,
    path="/chat"    ,
)

add_routes(
    app,
    rag_chain,
    path="/rag",
)

add_routes(
    app,
    agent_chain,
    path="/agent",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8501)