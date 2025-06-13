# -------------------------------------
# 1. 라이브러리 임포트
# -------------------------------------
from langchain.agents import tool, AgentExecutor, create_openai_functions_agent
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from google.cloud import aiplatform
import os
import random
import json

import google.generativeai as genai

# -------------------------------------
# 2. GCP 설정
# -------------------------------------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../application_credentials.json"
aiplatform.init(project=os.environ["PROJECT_NAME"], location=os.environ["LOCATION"])
# genai.configure(api_key=os.environ["API_KEY"])


# -------------------------------------
# 3. LangChain 툴 정의
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
    if to_player not in player_db:
        return f"{to_player}는 게임 상 존재하지 않습니다. {', '.join(list(player_db.keys()))}중 정확한 이름을 입력해 주세요"
    
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
def get_evidence_list(player: str) -> str:
    """현재 위치에서 볼 수 있는 증거품 목록을 보여줍니다"""
    player_position = player_db[player]['position']
    evidence_list = list(map_dict[player_position].keys())
    return f"{player_position}이 있는 증거품들: {', '.join(evidence_list)}"

@tool
def get_evidence_info(player: str,evidence: str) -> str:
    """사용자가 탐색하고 싶은 증거품의 세부내용을 보여줍니다."""
    player_position = player_db[player]['position']
    evidence_list = list(map_dict[player_position].keys())
    if evidence in evidence_list:
        return f"serching---{evidence}---{map_dict[player_position][evidence]}"
    else:
        return f"{evidence}가 {', '.join(evidence_list)} 중에 없습니다. 정확한 증거품 명을 입력하세요"

@tool
def get_conversation_log(player: str) -> str:
    """해당 플레이어의 대화 기록을 출력합니다."""
    log = player_db.get(player, {}).get("conversation_log", [])
    if not log:
        return f"{player}의 대화 기록이 없습니다."
    return f"{player}의 대화 기록:\n" + "\n".join(log)


if __name__ == "__main__":
    # -------------------------------------
    # 4. 게임 상태 초기화
    # -------------------------------------
    turn = 0
    story_name = "story1"
    with open(f"storys/{story_name}/private_story.json") as f:
        player_dict = json.load(f)
        player_list = list(player_dict.keys())
        
    with open(f"storys/{story_name}/map.json") as f:
        map_dict = json.load(f)
        map_list = list(map_dict.keys())

    VALID_LOCATIONS = map_list

    player_db = {
        name: {
            "position": map_list[0],
            "conversation_log":[]
        }
        for name in player_list
    }
    print(player_db)

    # -------------------------------------
    # 5. Agent 설정
    # -------------------------------------
    llm = ChatVertexAI(model_name="gemini-2.0-flash-001", temperature=0.5)
    tools = [get_position, move_player, talk_to_player, get_available_talk_targets, get_conversation_log, get_evidence_list, get_evidence_info]
    sample_evidence = list(map_dict[map_list[0]].keys())[0]
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        f""" 너는 턴제 게임의 한국어 도우미야. 사용자의 자연어 입력을 이해해서 '이동', '대화', '대화 가능한 사람 확인' 등을 입력했을 때 적절한 하나의 도구를 찾아서 호출해야 해.
        - '대화'는 같은 위치에 있는 사람에게만 가능해.
        - '누구랑 대화 가능해?' 혹은 '나와 같은 장소에 있는 사람은 누구야?' 라는 식의 질문이 오면, 'get_available_talk_targets'를 사용해.
        - talk_to_player 를 사용하고 나서는 talk_to_player의 결과가 'talk_to_player {player_list[0]} {player_list[1]}' 과 같은 형식이면 그대로 출력해주고, 아니면 결과 내용에 맞춰서 사용자에게 안내해줘
        - '주변에 뭐가 있는지 보고 싶어' 와같은 탐색 질문을 하면 get_evidence_list를 호출해서 현 위치에서 볼수 있는 증거 리스트를 출력해줘
        - get_evidence_info 를 사용하고 나서는 get_evidence_info 결과가 'serching---{sample_evidence}---{map_dict[map_list[0]][sample_evidence]}' 과 같은 형식이면 그대로 출력해주고, 아니면 결과 내용에 맞춰서 사용자에게 안내해줘
        """),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])


    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # player_model = genai.GenerativeModel(
    # "gemini-2.0-flash-lite",
    # )
    
    
    

    # -------------------------------------
    # 6. 게임 루프
    # -------------------------------------
    while True:
        current_player = player_list[turn % len(player_list)]
        print(f"\n{current_player}의 턴입니다.")
        if current_player == "":
            pass
        else:
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
        elif "serching" in result['output']:
            f,evidence,evidence_info = result['output'].split("---")
            print(f"{current_player}가 {player_db[current_player]['position']}에서 {evidence}를 확인했습니다.")
            turn += 1
        else:
            print(f"[{current_player} 수행 결과] {result['output']}")
            print(f"{current_player}는 아직 턴을 소모하지 않았습니다.")
