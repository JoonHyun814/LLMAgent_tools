# -------------------------------------
# 1. 라이브러리 임포트
# -------------------------------------
from langchain.agents import tool, AgentExecutor, create_openai_functions_agent
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from google.cloud import aiplatform
from typing import TypedDict
import os
import random

# -------------------------------------
# 2. GCP 설정
# -------------------------------------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../application_credentials.json"
aiplatform.init(project=os.environ["PROJECT_NAME"], location=os.environ["LOCATION"])

# -------------------------------------
# 3. 게임 상태 초기화
# -------------------------------------
VALID_LOCATIONS = ["A", "B", "C", "D"]

player_db = {
    name: {
        "position": random.choice(["A", "B", "C", "D"]),
        "conversation_log":[]
    }
    for name in ["플레이어1", "플레이어2", "플레이어3", "플레이어4"]
}

# -------------------------------------
# 4. LangChain 툴 정의
# -------------------------------------
@tool
def get_position(player: str) -> str:
    """플레이어의 현재 위치를 반환합니다."""
    return f"{player}의 현재 위치는 {player_db[player]['position']}입니다."

@tool
def move_player(player: str, location: str) -> str:
    """플레이어를 지정된 위치로 이동시킵니다."""
    if location not in VALID_LOCATIONS:
        return f"'{location}'은(는) 유효한 장소가 아닙니다. 이동 가능한 장소는 {', '.join(VALID_LOCATIONS)}입니다. 정확한 명칭을 입력해 주세요."
    
    player_db[player]["position"] = location
    return f"{player}이(가) {location}으로 이동했습니다."


@tool
def talk_to_player(from_player: str, to_player: str) -> str:
    """두 플레이어 사이에 대화를 진행 합니다."""
    if player_db[from_player]['position'] != player_db[to_player]['position']:
        return f"{to_player}은(는) 같은 장소에 있지 않아 대화할 수 없습니다."
    
    conversation_log = []
    for _ in range(3):
        q = input(f"{from_player} 의 질문 :")
        conversation_log.append(f"{from_player}: {q}")
        a = input(f"{to_player} 의 답변 :")
        conversation_log.append(f"{to_player}: {a}")
    player_db[from_player]["conversation_log"] += conversation_log
    player_db[to_player]["conversation_log"] += conversation_log
        
    return f"talk_to_player {from_player} {to_player}"

@tool
def get_available_talk_targets(player: str) -> str:
    """현재 같은 위치에 있는 다른 플레이어 목록을 보여줍니다."""
    player_position = player_db[player]['position']
    others = [
        name for name, data in player_db.items()
        if name != player and data['position'] == player_position
    ]
    if not others:
        return f"{player}와 같은 장소에 있는 플레이어가 없습니다."
    return f"{player}가 대화할 수 있는 대상: {', '.join(others)}"

@tool
def get_conversation_log(player: str) -> str:
    """해당 플레이어의 대화 기록을 출력합니다."""
    log = player_db.get(player, {}).get("conversation_log", [])
    if not log:
        return f"{player}의 대화 기록이 없습니다."
    return f"{player}의 대화 기록:\n" + "\n".join(log)

# -------------------------------------
# 5. Agent 설정
# -------------------------------------
llm = ChatVertexAI(model_name="gemini-2.0-flash-001", temperature=0.5)
tools = [get_position, move_player, talk_to_player, get_available_talk_targets, get_conversation_log]

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """ 너는 턴제 게임의 한국어 도우미야. 사용자의 자연어 입력을 이해해서 '이동', '대화', '대화 가능한 사람 확인' 등을 입력했을 때 적절한 하나의 도구를 찾아서 호출해야 해.
     - '대화'는 같은 위치에 있는 사람에게만 가능해.
     - '누구랑 대화 가능해?' 혹은 '나와 같은 장소에 있는 사람은 누구야?' 라는 식의 질문이 오면, 'get_available_talk_targets'를 사용해.
     - talk_to_player 를 사용하고 나서는 talk_to_player의 결과 그대로 'talk_to_player 플레이어1 플레이어2' 로 결과를 출력해줘
     """),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# -------------------------------------
# 6. 게임 루프
# -------------------------------------
player_list = list(player_db.keys())
turn = 0

print("🎮 게임을 시작합니다! 플레이어는 이동 또는 대화할 수 있습니다.")

while True:
    current_player = player_list[turn % len(player_list)]
    print(f"\n{current_player}의 턴입니다.")
    user_input = input("행동 입력 (종료는 'exit'): ")

    if user_input.lower() in ["exit", "quit"]:
        break

    result = agent_executor.invoke({
        "input": f"{current_player}의 명령: {user_input}"
    })

    # 대화 시도(talk_to_player)만 턴을 소모
    if "talk_to_player" in result['output']:
        f,p1,p2 = result['output'].split(" ")
        location = player_db[p1]['position']
        print(f"{p1}과 {p2[:-1]}가 {location}에서 대화했습니다.")
        turn += 1
    else:
        print(f"[{current_player} 수행 결과] {result['output']}")
        print(f"{current_player}는 아직 턴을 소모하지 않았습니다.")
