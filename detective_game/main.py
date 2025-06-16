# -------------------------------------
# 1. 라이브러리 임포트
# -------------------------------------
from langchain.agents import tool, AgentExecutor, create_openai_functions_agent
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import HumanMessage
from google.cloud import aiplatform
import os
from datetime import datetime
import uuid
import json
from dotenv import load_dotenv

# -------------------------------------
# 2. GCP 설정
# -------------------------------------
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../application_credentials.json"
aiplatform.init(project=os.environ["PROJECT_NAME"], location=os.environ["LOCATION"])

# -------------------------------------
# 3. LangChain 툴 정의
# -------------------------------------
@tool
def move_player(player: str, location: str) -> str:
    """플레이어를 지정된 위치로 이동시킵니다."""
    print("tool 사용: move_player")
    if location not in VALID_LOCATIONS:
        return f"'{location}'은(는) 유효한 장소가 아닙니다. 이동 가능한 장소는 {', '.join(VALID_LOCATIONS)}입니다. 정확한 명칭을 입력해 주세요."
    player_db[player]["position"] = location
    position = player_db[player]["position"]
    talkable = [p for p in player_db if p != player and player_db[p]["position"] == position]
    evidences = list(map_dict[position].keys())  # 해당 장소의 증거들
    player_db[player]["talkable"] = talkable
    player_db[player]["evidences"] = evidences
    player_db[player]["conversation_log"].append(f"{location}으로 이동했습니다.")
    return f"{{'player':'{player}','location':'{location}'}}"


@tool
def talk_to_player(from_player: str, to_player: str) -> str:
    """두 플레이어 사이에 대화를 진행 합니다."""
    global turn
    print("tool 사용: talk_to_player")
    if to_player not in player_db:
        return f"{to_player}는 게임 상 존재하지 않습니다. {', '.join(list(player_db.keys()))}중 정확한 이름을 입력해 주세요"
    
    if player_db[from_player]['position'] != player_db[to_player]['position']:
        return f"{to_player}은(는) 같은 장소에 있지 않아 대화할 수 없습니다."
    
    player_db[from_player]["conversation_log"].append(f"{to_player}이 {from_player}에게 대화를 걸었습니다.")
    player_db[to_player]["conversation_log"].append(f"{to_player}이 {from_player}에게 대화를 걸었습니다.")
    for _ in range(3):
        player_db[from_player]["conversation_log"].append(f"{to_player}에게 질문하세요")
        if from_player == "매기":
            q = get_player2_action("매기")
            print(q)
        elif from_player == "톰":
            q = get_player2_action("톰")
            print(q)
        else:
            q = input(f"{from_player} 의 질문 :")
            
        player_db[from_player]["conversation_log"].append(f"{from_player}: {q}")
        player_db[to_player]["conversation_log"].append(f"{from_player}: {q}")
        
        player_db[to_player]["conversation_log"].append(f"{from_player}에게 답변하세요")
        if to_player == "매기":
            a = get_player2_action("매기")
            print(a)
        elif to_player == "톰":
            a = get_player2_action("톰")
            print(a)
        else:
            a = input(f"{to_player} 의 답변 :")
            
        player_db[from_player]["conversation_log"].append(f"{to_player}: {a}")
        player_db[to_player]["conversation_log"].append(f"{to_player}: {a}")
    player_db[from_player]["conversation_log"].append(f"대화가 끝났습니다.")
    player_db[to_player]["conversation_log"].append(f"대화가 끝났습니다.")
    turn += 1
    return f"{{'turn_used':True,'from_player':'{from_player}','to_player':'{to_player}'}}"

@tool
def get_evidence_info(player: str,evidence: str) -> str:
    """명령을 내린 player가 탐색하고 싶은 evidence의 세부내용을 보여줍니다."""
    global turn
    print("tool 사용: get_evidence_info")
    player_position = player_db[player]['position']
    evidence_list = list(map_dict[player_position].keys())
    if evidence in evidence_list:
        turn += 1
        return f"{{'player':'{player}','evidence':'{evidence}','evidence_info':'{map_dict[player_position][evidence]}'}}"
    else:
        return f"{{'error':'{evidence}가 {', '.join(evidence_list)} 중에 없습니다. 정확한 증거품 명을 입력하세요.'}}"


if __name__ == "__main__":
    # -------------------------------------
    # 4. 게임 상태 초기화
    # -------------------------------------
    game_start_time = datetime.today().strftime("%Y%m%d%H%M%S")
    game_id = f"{game_start_time}_{uuid.uuid4()}"
    turn = 0
    story_name = "story1"
    with open(f"storys/{story_name}/private_story.json") as f:
        player_dict = json.load(f)
        player_list = list(player_dict.keys())
        
    with open(f"storys/{story_name}/map.json") as f:
        map_dict = json.load(f)
        map_list = list(map_dict.keys())

    with open(f"storys/{story_name}/story.txt") as f:
        game_story_list = f.readlines()
        game_story_prompt = "\n".join(game_story_list)
        
    with open(f"gamemanager_prompt.txt") as f:
        gamemanager_prompt_list = f.readlines()
        gamemanager_prompt = "\n".join(gamemanager_prompt_list)

    VALID_LOCATIONS = map_list

    player_db = {
        name: {
            "position": map_list[0],
            "talkable":player_list,
            "evidences":list(map_dict[map_list[0]].keys()),
            "conversation_log":[]
        }
        for name in player_list
    }

    # -------------------------------------
    # 5. Game manage Agent 설정
    # -------------------------------------
    llm = ChatVertexAI(model_name="gemini-2.0-flash-001", temperature=0.5)
    tools = [move_player, talk_to_player, get_evidence_info]
    sample_evidence = list(map_dict[map_list[0]].keys())[0]
    gamemanager_prompt = gamemanager_prompt.replace("{to_player}",player_list[0])
    gamemanager_prompt = gamemanager_prompt.replace("{from_player}",player_list[1])
    gamemanager_prompt = gamemanager_prompt.replace("{sample_evidence}",sample_evidence)
    gamemanager_prompt = gamemanager_prompt.replace("{sample_evidence_info}",map_dict[map_list[0]][sample_evidence])
    prompt = ChatPromptTemplate.from_messages([
        ("system", gamemanager_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])


    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # -------------------------------------
    # 6. Game play Agent 설정
    # -------------------------------------
    def get_player2_action(player):
        """게임 정보 기반으로 LLM에게 한 줄의 액션 요청"""
        position = player_db[player]["position"]
        talkable = [p for p in player_db if p != player and player_db[p]["position"] == position]
        evidences = list(map_dict[position].keys())  # 해당 장소의 증거들
        conversation = "\n".join(player_db[player]["conversation_log"])

        prompt = game_play_prompt.format(
            position=position,
            player = player,
            player_story = player_dict[player],
            talkable=", ".join(talkable),
            evidences=", ".join(evidences),
            conversation=conversation
        )
        result = game_play_llm.generate([
            [HumanMessage(content=prompt)]
        ])

        return result.generations[0][0].text

    with open("character_system_prompt.txt") as f:
        prompt_list = f.readlines()
        character_system_prompt = "\n".join(prompt_list)
        character_system_prompt = character_system_prompt.replace("{game_story}",game_story_prompt)
        character_system_prompt = character_system_prompt.replace("{player_list}",",".join(player_list))
        character_system_prompt = character_system_prompt.replace("{map_list}",",".join(map_list))
    game_play_prompt = ChatPromptTemplate.from_messages([
        ("system", 
        character_system_prompt),
        ("human", """다음은 현재게임의 정보입니다:
    - 나의 현재위치: {position}
    - 같은 장소에 있는 사람들: {talkable}
    - 이 장소에 있는 증거들: {evidences}
    - 나의 대화로그:{conversation}
    한 가지 행동을 선택해서 한 줄의 자연어로 대답해주세요."""),
    ])

    game_play_llm = ChatVertexAI(model_name="gemini-2.0-flash-001", temperature=0.7)


    # -------------------------------------
    # 7. 게임 루프
    # -------------------------------------
    print(game_story_prompt)
    def game_logging(game_id,message):
        with open(f"logs/{game_id}.txt","a") as f:
            f.write(message)
    while True:
        current_player = player_list[turn % len(player_list)]
        print(f"\n{current_player}의 턴입니다.")
        game_logging(game_id,f"{current_player}의 턴입니다.\n")
        print(f"""
    현제 상황
        위치: {player_db[current_player]["position"]}
        같은 장소에 있는 사람: {",".join(player_db[current_player]["talkable"])}
        탐색 가능한 증거품: {",".join(player_db[current_player]["evidences"])}
        장소 목록: {",".join(map_list)}
        """)
        print(player_db[current_player]["conversation_log"])
        player_db[current_player]["conversation_log"].append("다음 행동을 선택하세요")
        if current_player == "매기":
            # 모델에게 한 줄의 액션 요청
            user_input = get_player2_action("매기")
            print(f"🤖 {current_player}(AI)가 선택한 행동: {user_input}")
        elif current_player == "톰":
            user_input = get_player2_action("톰")
            print(f"🤖 {current_player}(AI)가 선택한 행동: {user_input}")
        else:
            user_input = input("행동 입력 (종료는 'exit'): ")
            if user_input == "pass":
                turn += 1
                continue

        player_db[current_player]["conversation_log"].append(f"{current_player}의 명령: {user_input}")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        game_logging(game_id,f"{user_input}\n")
        result = agent_executor.invoke({
            "input": f"{current_player}의 명령: {user_input}",
            "player_list":",".join(player_db[current_player]["talkable"]),
            "evidence_list":",".join(player_db[current_player]["evidences"])
        })
        print(result)
        game_logging(game_id,f"{result}\n")
        player_db[current_player]["conversation_log"].append(result["output"])
