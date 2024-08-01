from typing import Any
from fastapi import FastAPI
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain_core.utils.function_calling import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langserve import add_routes
from langchain.pydantic_v1 import BaseModel
from app.set_prompt import prompt, rag_prompt, agent_prompt
from app.set_model import llm
from app.set_documents import vectorstore
from app.set_tools import tools

class Input(BaseModel):
    input: str


class Output(BaseModel):
    output: Any

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

agent = (
    {
        "question": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_functions(
            x["intermediate_steps"]
        ),
    }
    | agent_prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

#-------------- routes -----------------

add_routes(
    app,
    basic_chain,
    path="/chat",
)

add_routes(
    app,
    rag_chain,
    path="/rag",
)

add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "agent"}
    ),
    path="/agent",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8501)