# Langchain_Quickstart

본 리포지토리는 Langchain의 핵심 구성 요소들을 모듈형으로 제공하여 LLM 기반 애플리케이션 개발을 위한 기본 프레임워크를 제공합니다.

이 프레임워크를 통해 애플리케이션에서 REST API를 배포하는 기본 기능을 구현할 수 있습니다.

# Langchain playground 접속 방법

- http://127.0.0.1:8000/chat/playground/ 로 접속


# 외부에서 API 요청하는 법
```python
from langserve import RemoteRunnable
langchain = RemoteRunnable("{랭체인 API를 제공하는 서버 IP}/chat/")

langchain.invoke({
    "question" : "langchain이 뭐야?"
})

```