from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 프롬프트 설정
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant.
            If you DON't know the answer, just say you don't know the answer.
            """,
        ),
        ("human", "{question}"),
    ]
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful document searching assistant. 
            Answer questions using only the following context. 
            And tell me the part you are cited.
            
            If you don't know the answer just say you don't know, don't make it up:\n\n{context}""",
        ),
        ("human", "{question}"),
    ]
)


agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are mathematics. answer the question"),
    ("human", "{question}"),
    ("placeholder", "{agent_scratchpad}")
])