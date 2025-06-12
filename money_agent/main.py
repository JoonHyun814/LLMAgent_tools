# -------------------------------------
# ✅ 1. 라이브러리 임포트
# -------------------------------------
from langchain.agents import tool, AgentExecutor, create_openai_functions_agent
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from google.cloud import aiplatform
from typing import TypedDict
import os
from datetime import datetime

# -------------------------------------
# ✅ 2. GCP 인증 및 환경 설정
# -------------------------------------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../application_credentials.json"
aiplatform.init(project=os.environ["PROJECT_NAME"], location=os.environ["LOCATION"])

# -------------------------------------
# ✅ 3. 사용자 DB 및 입출금 내역 초기화
# -------------------------------------
user_db = {
    "name": "홍길동",
    "age": 30,
    "money": 50000,
    "transaction_history": []
}


# -------------------------------------
# ✅ 4. 툴 정의
# -------------------------------------

@tool
def get_user_info(field: str) -> str:
    """사용자의 정보를 반환합니다. 예: name, age, money, transaction_history"""
    return str(user_db.get(field, "해당 정보 없음"))

@tool
def update_user_money(change: str) -> str:
    """사용자의 돈을 입금(+)/출금(-)하고 기록합니다. 예: '+10000' 또는 '-5000'"""
    try:
        amount = int(change)
        user_db["money"] += amount

        # 입출금 내역에 추가
        user_db["transaction_history"].append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "change": amount,
            "balance": user_db["money"]
        })

        return f"변경 완료. 현재 돈: {user_db['money']}"
    except Exception as e:
        return f"오류 발생: {e}"


# -------------------------------------
# ✅ 5. LLM & Agent 설정
# -------------------------------------
llm = ChatVertexAI(
    model_name="gemini-2.0-flash-001",
    temperature=0.7
)

tools = [get_user_info, update_user_money]

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 사용자 정보와 입출금 내역을 관리하는 한국어 도우미야."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# -------------------------------------
# ✅ 6. LangGraph 정의
# -------------------------------------
class GraphState(TypedDict):
    user_input: str
    response: str

def agent_node(state: GraphState) -> GraphState:
    result = agent_executor.invoke({"input": state["user_input"]})
    return {
        "user_input": state["user_input"],
        "response": f"[Agent 응답] {result['output']}"
    }

builder = StateGraph(GraphState)
builder.add_node("agent", agent_node)
builder.set_entry_point("agent")
builder.add_edge("agent", END)
graph = builder.compile()

# -------------------------------------
# ✅ 7. 채팅 루프
# -------------------------------------
while True:
    user_input = input("👤 입력 (종료하려면 'exit'): ")
    if user_input.lower() in ["exit", "quit"]:
        break

    result = graph.invoke({"user_input": user_input})
    print(result["response"])
