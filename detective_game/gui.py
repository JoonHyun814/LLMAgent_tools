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
import gradio as gr

# -------------------------------------
# 2. GCP 설정
# -------------------------------------
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../application_credentials.json"
aiplatform.init(project=os.environ["PROJECT_NAME"], location=os.environ["LOCATION"])

# -------------------------------------
# 3. LangChain 툴 정의
# -------------------------------------
# def game_logging(game_id,message):
#     with open(f"logs/{game_id}.txt","a") as f:
#         f.write(message)
def conversation_logging(player_list,conversation):
    for player in player_list:
        player_db[player]["conversation_log"].append(conversation)
    # game_logging(game_id,conversation)


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
    """두 플레이어 사이에 대화를 진행 합니다. 명령을 내린 사람이 반드시 from_player가 되어야 합니다."""
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
            q = get_player2_action(current_player,f"당신은 {from_player} 입니다. {to_player} 에게 질문하세요")
            print(q)
        conversation_logging(player_list,f"{from_player}: {q}")
        
        # player_db[to_player]["conversation_log"].append(f"{from_player}에게 답변하세요")
        if to_player == person_player:
            a = input(f"{to_player} 의 답변 :")
        else:
            a = get_player2_action(current_player,f"당신은 {to_player} 입니다. {from_player}의 마지막 질문에 답변하세요")
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

    
# -------------------------------------
# 7. Gradio
# -------------------------------------
def select_character(selected, game_id, person_player):
    player_db = game_db[game_id]["player_db"]
    player_dict = game_db[game_id]["player_dict"]
    player_list = list(player_dict.keys())
    person_player = selected
    description = "\n".join(player_dict[person_player])
    game_db[game_id]["log_history"] += f"'{selected}' 캐릭터로 게임을 시작합니다.\n<{person_player}의 비밀 정보>\n{description}\n"
    
    current_player = player_list[game_db[game_id]["turn"] % len(player_list)]
    game_db[game_id]["log_history"] += f"\n[{current_player}의 턴 시작]\n"
    if current_player == person_player:
        game_db[game_id]["log_history"] + "\n명령을 입력하고 [다음 턴]을 누르세요."
    return game_db[game_id]["log_history"], gr.update(visible=(current_player == person_player)), person_player, gr.update(visible=False), gr.update(visible=False)

def advance_turn(user_input, game_id, current_player, person_player):
    player_dict = game_db[game_id]["player_dict"]
    player_list = list(player_dict.keys())
    current_player = player_list[game_db[game_id]["turn"] % len(player_list)]
    next_player = player_list[(game_db[game_id]["turn"]+1) % len(player_list)]
    
    if current_player == person_player:
        game_db[game_id]["log_history"] += f"{current_player}의 명령: {user_input}\n"
        result = f"(에이전트 응답 예시)"
        game_db[game_id]["turn"] += 1
    else:
        # LLM으로부터 명령 생성
        llm_command = "(LLM 응답 예시)"
        game_db[game_id]["log_history"] += f"{current_player}의 명령: {llm_command}\n"
        result = f"(에이전트 응답 예시)"
        game_db[game_id]["turn"] += 1
        
    if person_player == next_player:
        game_db[game_id]["log_history"] + "\n명령을 입력하고 [다음 턴]을 누르세요."
    
    game_db[game_id]["log_history"] += f"결과: {result}\n"
    game_db[game_id]["log_history"] += f"\n[{next_player}의 턴 시작]\n"
    return game_db[game_id]["log_history"], gr.update(visible=(next_player == person_player),value=""), current_player, person_player

def game_start(story_name):
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

    player_db = {
        name: {
            "position": map_list[0],
            "talkable":player_list,
            "evidences":list(map_dict[map_list[0]].keys()),
            "conversation_log":[]
        }
        for name in player_list
    }
    game_start_time = datetime.today().strftime("%Y%m%d%H%M%S")
    game_id = f"{game_start_time}_{uuid.uuid4()}"
    game_db[game_id] = {}
    game_db[game_id]["player_db"] = player_db
    game_db[game_id]["player_dict"] = player_dict
    game_db[game_id]["map_dict"] = map_dict
    game_db[game_id]["turn"] = 0
    game_db[game_id]["log_history"] = ""
    
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

        player_story = "\n".join(player_dict[player])
        prompt = game_play_prompt.format(
            position=position,
            player = player,
            player_story = player_story,
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
        ("human", """다음은 현재게임의 정보 및 대화 맥락입니다:
    - 나의 현재위치: {position}
    - 이 장소에 있는 증거들: {evidences}
    - 대화 가능한 캐릭터 목록: {player_list}
    - 게임 맥락:{conversation}

    다음 지시에 따릅니다:{next_action}
    """),
    ])

    game_play_llm = ChatVertexAI(model_name="gemini-2.0-flash-001", temperature=0.5)
    return game_id, player_list, game_story_prompt, gr.update(choices=player_list,value=player_list[0])

with gr.Blocks() as demo:
    game_db = {}
    game_id = gr.State("")
    player_list = gr.State([])
    person_player = gr.State("")
    current_player = gr.State("")
    turn = gr.State(0)
    log_history = gr.State("")
    
    gr.Markdown("## Crime Scene")
    game_story_viewer = gr.Textbox(label ="게임 스토리")
    story_selector = gr.Dropdown(choices=["story1"], label="스토리 종류 선택")
    game_start_button = gr.Button("게임 시작")
    
    
    with gr.Row():    
        char_selector = gr.Dropdown(choices=player_list.value, label="당신의 캐릭터 선택")
        select_button = gr.Button("선택")
    
    output_box = gr.Textbox(label="게임 로그", lines=25, interactive=False)
    user_input = gr.Textbox(label="당신의 명령", visible=False)
    next_button = gr.Button("다음 턴")

    game_start_button.click(
        game_start,
        inputs=[story_selector],
        outputs=[game_id, player_list, game_story_viewer, char_selector]
    )
    select_button.click(
        select_character, 
        inputs=[char_selector, game_id, person_player], 
        outputs=[output_box, user_input, person_player, select_button,char_selector]
    )
    next_button.click(
        advance_turn, 
        inputs=[user_input, game_id, current_player, person_player], 
        outputs=[output_box, user_input, current_player, person_player]
    )

demo.launch()
