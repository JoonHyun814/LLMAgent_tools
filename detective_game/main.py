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
import time
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
def conversation_logging(player_list,conversation):
    for player in player_list:
        player_db[player]["conversation_log"].append(conversation)
    game_logging(game_id,conversation)


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
    
    conversation_logging(player_list,f"{to_player}이 {from_player}에게 대화를 걸었습니다.")
    for _ in range(3):
        # player_db[from_player]["conversation_log"].append(f"{to_player}에게 질문하세요")
        if from_player == person_player:
            q = input(f"{from_player} 의 질문 :")
        else:
            q = get_player2_action(current_player,f"{to_player}에게 질문하세요")
            print(q)
        conversation_logging(player_list,f"{from_player}: {q}")
        
        # player_db[to_player]["conversation_log"].append(f"{from_player}에게 답변하세요")
        if to_player == person_player:
            a = input(f"{to_player} 의 답변 :")
        else:
            a = get_player2_action(current_player,f"{from_player}에게 답변하세요")
            print(a)
        conversation_logging(player_list,f"{to_player}: {a}")
        
    conversation_logging(player_list,f"{to_player}와 {from_player}가 대화를 마쳤습니다.")
    turn += 1
    return f"{to_player}와 {from_player}가 대화를 마쳤습니다."

@tool
def get_evidence_info(player: str,evidence: str) -> str:
    """명령을 내린 player가 탐색하고 싶은 evidence의 세부내용을 보여줍니다."""
    global turn
    print("tool 사용: get_evidence_info")
    player_position = player_db[player]['position']
    evidence_list = list(map_dict[player_position].keys())
    if evidence in evidence_list:
        turn += 1
        conversation_logging(player_list,f"{player}이(가) {player_position}에서 {evidence}를 확인했습니다.")
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
    def get_player2_action(player,next_action):
        """게임 정보 기반으로 LLM에게 한 줄의 액션 요청"""
        position = player_db[player]["position"]
        evidences = list(map_dict[position].keys())  # 해당 장소의 증거들
        conversation = "\n".join(player_db[player]["conversation_log"])

        prompt = game_play_prompt.format(
            position=position,
            player = player,
            player_story = player_dict[player],
            evidences=", ".join(evidences),
            player_list=player_list,
            conversation=conversation,
            next_action=next_action
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
    - 이 장소에 있는 증거들: {evidences}
    - 대화 가능한 캐릭터 목록: {player_list}
    - 나의 대화로그:{conversation}
    
    {next_action}
    """),
    ])

    game_play_llm = ChatVertexAI(model_name="gemini-2.0-flash-001", temperature=0.7)


    # -------------------------------------
    # 7. 게임 루프
    # -------------------------------------
    print(game_story_prompt)
    while True:
        person_player = input(f"{','.join(player_list)} 중 플레이할 캐릭터를 선택하세요: ")
        if person_player in player_list:
            print("다음은 당신만 알고 있는 당신 캐릭터의 설명입니다. 이 내용을 참고하여 범인을 찾고, 범인이시라면 그 사실을 숨기세요.")
            print(f"<{person_player}에 대한 설명>")
            print(player_dict[person_player])
            input("\n\n시작하려면 아무키나 입력하세요: ")
            break
        else:
            print(f"{','.join(player_list)} 중 정확한 캐릭터 이름을 입력하세요")
    
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
        탐색 가능한 증거품: {",".join(player_db[current_player]["evidences"])}
        캐릭터 목록: {",".join(player_list)}
        장소 목록: {",".join(map_list)}
        """)
        print(player_db[current_player]["conversation_log"])
        # player_db[current_player]["conversation_log"].append("다음 행동을 선택하세요")
        if current_player == person_player:
            user_input = input("행동 입력 (종료는 'exit'): ")
            if user_input == "pass":
                turn += 1
                continue
        else:
            user_input = get_player2_action(current_player,"다음 행동을 선택하세요")

        player_db[current_player]["conversation_log"].append(f"{current_player}의 명령: {user_input}")
        
        if user_input.lower() in ["exit", "quit"]:
            # conversation_logging(player_list,"가장 의심가는 상대를 선택하시오.")
            for player in player_list:
                if player == person_player:
                    result = input("가장 의심가는 상대를 선택하시오.: ")
                else:
                    result = get_player2_action(player,"가장 의심가는 상대를 선택하시오.")
                print(f"{player}의 답변: {result}")
            break
        
        game_logging(game_id,f"{current_player}의 명령: {user_input}")
        result = agent_executor.invoke({
            "input": f"{current_player}의 명령: {user_input}",
            "player_list":",".join(player_list),
            "evidence_list":",".join(player_db[current_player]["evidences"])
        })
        game_logging(game_id,f"{result['output']}\n\n")
