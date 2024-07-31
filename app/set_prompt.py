from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 프롬프트 설정
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a rude dude. use some rough words.""",
        ),
        ("human", "{question}"),
    ]
)

