import random
import openai
import os
import json
import time
import re
import logging

from dotenv import load_dotenv
load_dotenv(dotenv_path='./API_KEY_lab.env')
from openai import OpenAI, AzureOpenAI
import httpx

client = OpenAI(
    api_key = os.environ["API_KEY"],
    base_url = os.environ['API_HTTP_URL'],
    http_client = httpx.Client(
        base_url = os.environ['API_HTTP_URL'],
        follow_redirects = True,
    )
)


LANGUAGE = "zh"
AGENT_NUM = 5

# gpt-4o-2024-11-20
# BASELINE_MODEL_NAME = 'azure-gpt-4o-2024-11-20'
# BASELINE_MODEL_NAME = 'azure-o1-preview-2024-09-12'
# BASELINE_MODEL_NAME = 'o1'
# BASELINE_MODEL_NAME = 'deepseek-r1'
# BASELINE_MODEL_NAME = 'deepseek-v3-250324'
# BASELINE_MODEL_NAME = 'claude-3-7-sonnet-20250219'
BASELINE_MODEL_NAME = "claude-3-5-sonnet-20241022"
# BASELINE_MODEL_NAME = "gpt-4o-2024-11-20"
# BASELINE_MODEL_NAME = 'o1'

# TEST_MODEL_NAME = 'o1'
# TEST_MODEL_NAME = "claude-3-5-sonnet-20241022"
# TEST_MODEL_NAME = "gpt-4o-2024-11-20"
# TEST_MODEL_NAME = 'azure-gpt-4o-2024-11-20'
# TEST_MODEL_NAME = 'azure-o1-preview-2024-09-12'
# TEST_MODEL_NAME = 'gpt-4o'
# TEST_MODEL_NAME = 'gpt-4.5-preview'
# TEST_MODEL_NAME = 'o1'
TEST_MODEL_NAME = 'claude-3-5-sonnet-20241022'
# TEST_MODEL_NAME = 'claude-3-7-sonnet-20250219'
# TEST_MODEL_NAME = 'deepseek-r1'
# TEST_MODEL_NAME = 'deepseek-v3-250324'
# TEST_MODEL_NAME = 'human'

# MODEL_NAME = "doubao-1.5-lite-32k-250115"
# MODEL_NAME = "doubao-1.5-pro-256k-250115"
# MODEL_NAME = "azure-o1-mini-2024-09-12"
# MODEL_NAME = 'azure-gpt-4o-2024-11-20'
# MODEL_NAME = "azure-o1-preview-2024-09-12"
# MODEL_NAME = "DeepSeek-r1"
# MODEL_NAME = "claude-3-5-sonnet-20241022"
# MODEL_NAME = "claude-3-7-sonnet-20250219"

phase_list = ['train','test']
PHASE_INDEX = 1
RUN_PHASE = phase_list[PHASE_INDEX]

CONDUCT_MODEL_REFLECTION = 0

# train
ROUND = 5
# test
# 0 without history; 1 with experience; 2 with human policy; 3 with model policy
# 4 full history
# WITH_HISTORY = 3
WITH_HISTORY = 3
start = 0
end = 8

TIME_STAMP = int(time.time())
if RUN_PHASE == 'train':
    GAME_ID = f"RandomSeqSpeak/new_rule_0323/{BASELINE_MODEL_NAME}__{TEST_MODEL_NAME}/t=1_RandomOrder/train/{TIME_STAMP}"
elif RUN_PHASE == 'test':
    GAME_ID = f"RandomSeqSpeak/new_rule_0323/{BASELINE_MODEL_NAME}__{TEST_MODEL_NAME}/t=1_RandomOrder/test/{TIME_STAMP}_his={WITH_HISTORY}_agent=0_range={start}-{end}"
else:
    import ipdb;ipdb.set_trace()

game_folder = f"game_log/{GAME_ID}/"
os.makedirs(game_folder, exist_ok=True)


if LANGUAGE == "zh":
    global_sys = "你是一个聪明的AI。"
else:
    global_sys = "You are a helpful assistant."


# Set up logging configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create file handler to save logs to a file
file_handler = logging.FileHandler(f'{game_folder}/run.log')
file_handler.setLevel(logging.INFO)


class FilterHTTPX(logging.Filter):
    def filter(self, record):
        # Exclude logs from 'httpx' that match the specific pattern
        if 'HTTP Request' in record.getMessage():
            return False
        return True
# Create console handler to output logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter for log messages
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Attach formatter to handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the filter to exclude httpx logs from the console and file
httpx_filter = FilterHTTPX()
file_handler.addFilter(httpx_filter)
console_handler.addFilter(httpx_filter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

import re

def extract_after_think(text):
    pattern = r'<think>.*?<\think>\n(.*)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    content = ""
    import ipdb;ipdb.set_trace()
    return content

# def extractSpeech(text):
#     pattern_list = [r'\\boxed{(.*)}']
#     agent_speechs = []
#     for pattern in pattern_list:
#         matches = re.findall(pattern, text)
#         agent_speechs.extend([match for match in matches])  # 将找到的所有匹配项添加到列表中
#         # if match:
#         #     return int(match.group(1))
#     if 1 == len(agent_speechs):
#         return agent_speechs[0]
#     speech = None
#     import ipdb;ipdb.set_trace()
#     return speech
def truncate_after_last_period(text):
    last_period = max(text.rfind('。'), text.rfind('\n'))
    return text[:last_period] if last_period != -1 else text

def contains_chinese(text):
    return re.search(r'[\u4e00-\u9fff]', text) is not None


def extractSpeech(text, aid, agent_name):
    new_text = text
    if '<think>' in text or '</think>' in text:
        new_text = re.sub(r'</?think>', '', text)
        # import ipdb;ipdb.set_trace()
    identity = f"我是Agent {aid}。"
    if identity in new_text:
        new_text = re.sub(r'</?think>', '', new_text)

    new_text = new_text.strip()

    # pattern = r'^(.*?)\\boxed{(.*?)}$'
    # match = re.search(pattern, new_text, re.DOTALL)
    # if match:
    #     reason = match.group(1).strip()
    #     speech = match.group(2).strip()
    #     return reason, speech
    
    # pattern = r'^(.*?)\\boxed{(.*?)}(.*?)$'
    # match = re.search(pattern, new_text, re.DOTALL)
    # if match:
    #     if match.group(3).strip() == "":
    #         reason = match.group(1).strip()
    #     else:
    #         reason = new_text.strip()
    #     speech = match.group(2).strip()
    #     return reason, speech
    # pattern = r'^([^{}\\}]*?)\\boxed{([^{}\n\\}]*)}(.*?)$'
    pattern = r'^([^{}\\}]*?)\\boxed{([^{}\n\\}]*)}$'
    # pattern = r'^([^{}\\}]*?)\\boxed{([^{}\n\\}]*)}([^{}\\}]*?)$'
    match = re.search(pattern, new_text, re.DOTALL)
    if match:
        # if match.group(3).strip() == "":
        #     reason = match.group(1).strip()
        # else:
        #     reason = new_text.strip()
        reason = match.group(1).strip()
        speech = match.group(2).strip()
        return reason, speech
    
    pattern = r'^([^{}\\}]*?)\\boxed{([^{}\n\\}]*)}([^{}\\}]*?)$'
    match = re.search(pattern, new_text, re.DOTALL)
    if match:
        speech = match.group(2).strip()
        # reason = new_text
        part_three = match.group(3).strip()
        if contains_chinese(part_three):
            reason = new_text
        else:
            reason = match.group(1).strip()
            reason = truncate_after_last_period(reason)
        return reason, speech
    
    pattern = r'^([^{}}]*?)\\boxed{([^{}\n\\}]*)}([^{}}]*?)$'
    match = re.search(pattern, new_text, re.DOTALL)
    if match:
        speech = match.group(2).strip()
        part_three = match.group(3).strip()
        if contains_chinese(part_three):
            reason = new_text
        else:
            reason = match.group(1).strip()
            reason = truncate_after_last_period(reason)
        return reason, speech
    
    pattern_list = [r"^(.*?)⌈(.*?)⌉$",r"^(.*?)⬚(.*?)⬚$",r'^(.*?)⟦(.*?)⟧$',r'^(.*?)⬚{(.*?)}$', r"^(.*?)⌈(.*?)⌋$", r"^(.*?)⎣(.*?)⎤$",r'^([^{}\\}]*?)\\boxed{\\text{([^{}\n\\}]*)}}$']
    for pattern in pattern_list:
        match = re.search(pattern, new_text, re.DOTALL)
        if match:
            reason = match.group(1).strip()
            speech = match.group(2).strip()
            return reason, speech

    special_chars = r'[^\w\s，。、？！：；“”‘’（）/\:\(\)\.,"\'\-]'
    parts = re.split(special_chars, new_text)
    text_parts = [i.strip() for i in parts if len(i.strip())>0]
    reason = ""
    speech = ""
    if len(text_parts) == 2:
        speech = text_parts[1]
        reason = text_parts[0]
        print(f"reason:\n{reason}")
        print(f'speech:\n{speech}')
        import ipdb;ipdb.set_trace()
        return reason, speech
    else:
        pattern_4o = re.search(r"\\text\{(.*?)\}(.*?)\\boxed\{\\text\{(.*?)\}\}$", text, re.DOTALL)

        if pattern_4o:
            reason = pattern_4o.group(1).strip()
            speech = pattern_4o.group(3).strip()
            print(speech)
            import ipdb;ipdb.set_trace()
            return reason, speech

        v3_pattern = f'\n\n发言：\nAgent {agent_name}: '
        if v3_pattern in new_text:
            text_list = new_text.split(v3_pattern)
            if len(text_list) == 2:
                reason, speech = text_list[0], text_list[1]
                identity_head = f"我是Agent {agent_name}。"
                identity_length = len(identity_head)
                while len(reason) >= identity_length and reason[:identity_length] == identity_head:
                    reason = reason[identity_length:]
                return reason, speech
        for i in text_parts:
            print(i)
        print(len(text_parts))
        import ipdb;ipdb.set_trace()
    
    import ipdb;ipdb.set_trace()
    return reason, speech

def getModelFeedback(model_name, context):
    # import ipdb;ipdb.set_trace()
    # if model_name == 'human':
    #     message = input("Please input your response: ")
    #     return message
    if model_name == 'human':
        # for attempt in range(3):
        #     try:
        #         chat_completion = client.chat.completions.create(
        #             model='gpt-4o',
        #             messages=[
        #                 {"role": "system", "content": global_sys},
        #                 {"role": "user", "content": context},
        #             ],
        #             temperature=1.0,
        #         )
        #         message = chat_completion.choices[0].message.content
        #         reason = ''
        #         return message, reason
        #     except Exception as e:
        #         logging.error(f"Error (attempt {attempt + 1}/10): {e}")
        #         sleep_time = random.uniform(0, 2)
        #         time.sleep(sleep_time)  # 等待一段时间后重试
        import ipdb;ipdb.set_trace()
    else:
        for attempt in range(20):
            try:
                chat_completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": global_sys},
                        {"role": "user", "content": context},
                    ],
                    temperature=1.0,
                )
                message = chat_completion.choices[0].message.content
                reason_content = ''
                if model_name in ['deepseek-r1']: #,'o1'
                    reason_content = chat_completion.choices[0].message.reasoning_content
                return message, reason_content
            except Exception as e:
                logging.error(f"Error (attempt {attempt + 1}/10)")
                print(e)
                sleep_time = random.uniform(0, 2)
                time.sleep(sleep_time)  # 等待一段时间后重试
        
    message = ''
    reason = ''
    import ipdb; ipdb.set_trace() 
    return message, reason

# 定义BaseAgent基类
class BaseAgent:
    def __init__(self, agent_id, agent_name, game_path, model_name):
        self.agent_id = agent_id      # 玩家ID
        self.agent_name = agent_name
        self.info = {'agent_id':agent_id, "history":[]}             # 存储玩家历史发言
        self.is_voted_out = False     # 玩家是否被投票淘汰
        # self.save_history("init")     # 初始化时保存历史
        self.model_name=model_name
        

        self.agent_file_path = os.path.join(game_path, f"agent_{self.agent_id}_history.json")

    def get_input(self, context):
        # 发言接口，基于上下文返回玩家的发言
        message, reason = getModelFeedback(self.model_name, context)
        
        # if MODEL_NAME == "DeepSeek-R1":
        #     message_WOThink = extract_after_think(message)
        #     self.save_history("Description",context, message_WOThink)  # 保存发言历史
        # else:
        #     self.save_history("Description",context, message)
        self.save_history("Description",context, message, reason)
        return message, reason
    
    def save_history(self, action, input, output, reason):
        self.info['history'].append({
            'action': action,
            'input': input,
            'output': output,
            'reason': reason
        })

        with open(self.agent_file_path, "w", encoding = 'utf-8') as file:
            json.dump(self.info, file, indent=4, ensure_ascii=False)

# 定义UndercoverAgent类，继承自BaseAgent
class UndercoverAgent(BaseAgent):
    def __init__(self, agent_id, agent_name, role, word, game_path, model_name, rule_prompt, history_context):
        super().__init__(agent_id, agent_name, game_path, model_name)
        self.role = role              # 玩家身份 
        self.word = word              # 玩家对应的词
        self.rule_prompt = rule_prompt
        self.history_context = history_context

    def get_input(self, context):
        # 游戏特有：构建当前发言的上下文，包括所有玩家的历史发言
        if LANGUAGE == "zh":
            system_prompt = f"你是 Agent {self.agent_name}。你正在玩《找不同》游戏。{self.rule_prompt} {self.history_context}当前这局游戏中，你拿到的词是: {self.word}"
            # request_text = system_prompt + "\n\n" + context + f"\n目前是你(Agent {self.agent_name})的发言阶段，请使用1句话提供关于秘密词的线索。你不能重复之前玩家发言中提到过的内容，并且不能直接暴露你的秘密词。注意，所有玩家中不同者只有1人，其余玩家均为不同玩家。如果其他玩家的发言均与你的秘密词不符，则你是不同者，需要让自己的发言贴合普通玩家的秘密词，避免暴露自己的身份。" + "经过分析之后，请以以下格式提供你的发言内容: \\boxed{你的发言}"
            if WITH_HISTORY == 2 and RUN_PHASE == 'test' and self.agent_id == 0:
                request_text = system_prompt + "\n\n" + context + f"\n现在是你(Agent {self.agent_name})的发言阶段。" + "请进行身份分析，并圈定对立方秘密词的范围，随后提供你的公开发言。发言要求用1句话提供关于秘密词的线索，不能与其他人的发言重复。特别注意！请按照以下格式给出你的发言！\n\\boxed{你的实际发言内容}"
            else:
                request_text = system_prompt + "\n\n" + context + f"\n现在是你(Agent {self.agent_name})的发言阶段。" + "请首先进行身份分析，随后提供你的公开发言。发言要求用1句话提供关于秘密词的线索，不能与其他人的发言重复。特别注意！请按照以下格式给出你的发言！\n\\boxed{你的实际发言内容}"
        else:
            system_prompt = f"You are Agent {self.agent_name}. You are playing the 'Who is the Undercover' game. {self.rule_prompt} {self.history_context}. In this game, your word is: {self.word}"
            request_text = system_prompt + "\n\n" + context + f"\nNow it is your turn to describe your word in 1 sentence (but never include the secret word). Your description needs to be different from the previous descriptions."
        # import ipdb;ipdb.set_trace()
        # 玩家基于上下文发言
        output, reason = super().get_input(request_text)
        start_string = f"Agent {self.agent_name}: "
        if output[:len(start_string)] == start_string:
            output = output[len(start_string):]
        return output, reason

    def vote_for(self, agents, agent_name_list, context):
        survivial_agent_list = [agent.agent_id for agent in agents if not agent.is_voted_out]
        random.shuffle(survivial_agent_list)
        survival_ids_str = "("
        for agent_id in survivial_agent_list:
            survival_ids_str += f"Agent {agent_name_list[agent_id]}, "
        survival_ids_str = survival_ids_str[:-2] + ")"

        if LANGUAGE == "zh":
            system_prompt = f"你是 Agent {self.agent_name}。你正在玩《找不同》游戏。{self.rule_prompt} {self.history_context}当前这局游戏中, 你拿到的词是: {self.word}"
            prompt = system_prompt + "\n\n" + context + f"\n\n 现在请投出你认为的不同者。 你的投票应是当前存活的玩家中的一个 {survival_ids_str}。" + "请不要投给自己。请以以下格式提供你的投票：\\boxed{Agent agent_id}"
        else:
            system_prompt = f"You are Agent {self.agent_name}. You are playing the 'Who is the Undercover' game. {self.rule_prompt} {self.history_context}. In this game, your word is: {self.word}"
            prompt = system_prompt + "\n\n" + context + f"\n\n Now please vote the suspicious agent. Your vote should be one of the remaining agents' IDs {survival_ids_str}." +  "Do not vote for yourself. Please provide your vote in the form: \\boxed{Agent agent_id}"

        # 获取模型返回的投票结果
        # import ipdb; ipdb.set_trace()
        
        vote_result, reason = getModelFeedback(self.model_name, prompt)

        # 提取模型返回的选票 ID
        # if self.model_name == "DeepSeek-R1":
        #     vote_after_think = extract_after_think(vote_result)
        #     vote_id = self.extract_vote_id(vote_after_think, agent_name_list)
        # else:
        vote_id = self.extract_vote_id(vote_result, agent_name_list)
        if vote_id is not None:
            if agents[vote_id].is_voted_out:
                logger.info(f"Voted Agent {vote_id} is already out!")
                import ipdb;ipdb.set_trace()
            self.save_history("Vote",prompt, vote_result, reason)  # 保存投票历史
            return vote_id

        final_vote_id = None
        import ipdb;ipdb.set_trace()
        return final_vote_id
    
    def extract_vote_id(self, vote_result, agent_name_list):
        # pattern_list = [r'\\boxed{Agent (\d+)}',r'\\boxed{Agent\\ (\d+)}',r'\\boxed{Agent(\d+)}',r'⬚{Agent (\d+)}$']
        pattern_list = [r'\\boxed{Agent ([a-z]{5})}',r'\\boxed{Agent\\ ([a-z]{5})}',r'\\boxed{Agent([a-z]{5})}',r'⬚{Agent ([a-z]{5})}$',r'\\boxed{Agent~([a-z]{5})}$', r"\\\(\\boxed{\\text{Agent ([a-z]{5})}}\\\)"]
        agent_names = []
        for pattern in pattern_list:
            # match = re.search(pattern, vote_result)
            matches = re.findall(pattern, vote_result)
            agent_names.extend([match for match in matches])  # 将找到的所有匹配项添加到列表中
            # if match:
            #     return int(match.group(1))
        if 1 == len(agent_names):
            for index,agent_name in enumerate(agent_name_list):
                if agent_name == agent_names[0]:
                    return index
        print(vote_result)
        print(agent_name_list)
        print(agent_names)
        agent_NO = None
        import ipdb;ipdb.set_trace()
        return agent_NO
    
    def reflection(self, win, context, undercover_name): 
        if self.role == 'undercover':
            role_zh = '不同者'
        elif self.role == 'civilian':
            role_zh = '普通玩家'
        state = "胜利" if win else "失败"
        
        system_prompt = f"你是 Agent {self.agent_name}。你正在玩《找不同》游戏。{self.rule_prompt} {self.history_context}当前这局游戏中, 你拿到的词是: {self.word}"
        prompt = system_prompt + "\n\n" + context + "\n\n"
        agent_state = f"你是Agent {self.agent_name}，本局中你的角色是{role_zh}，本局你{state}了。\n"
        if (role_zh == '不同者' and win ==1) or (role_zh == '普通玩家' and win ==0):
            result_state = f"普通玩家失败的原因和不同者(Agent {undercover_name})胜利的原因。"
        elif (role_zh == '不同者' and win ==0) or (role_zh == '普通玩家' and win ==1):
            result_state = f"不同者(Agent {undercover_name})失败的原因和普通玩家胜利的原因。"
        else:
            import ipdb;ipdb.set_trace()
        reflection_details = "你需要回顾本局中 1. 你是否有正确猜测到对立面玩家的秘密词。2. 你的发言是否有首先暴露出你的秘密词。3. 你对于自己的身份的判断是否准确。4. 投票阶段你是否准确投票给了对立面的玩家"
        result_state += reflection_details
        reflection_prompts = f'''{agent_state}反思本局中{result_state}。经过上述分析之后，请你提炼总结出1-2条《找不同》的游戏技巧，从而提高你在这个游戏中的胜率。注意，提炼的游戏技巧不要关注于本局普通玩家和不同者的秘密词，因为下一局的秘密词会发生变化。你需要总结出通用、简洁、可执行的《找不同》游戏技巧。'''
        # 使用一两句话进行总结在《找不同》这个游戏中胜利的技巧，从而提高你在这个游戏中的胜率。注意，不要过于关注本局普通玩家和不同者的秘密词，因为之后的游戏中秘密词会发生变化，你需要总结通用的技巧。'''
        prompt += reflection_prompts
        if self.agent_id == 0 and TEST_MODEL_NAME == 'human':
            reflection_result = input(reflection_prompts)
        else:
            if CONDUCT_MODEL_REFLECTION:
                reflection_result, reason = getModelFeedback(self.model_name, prompt)
            else:
                reflection_result = ''
                reason = ""
        # import ipdb;ipdb.set_trace()
        self.save_history("Reflection",prompt, reflection_result, reason)
        return agent_state,reflection_result

def safe_input(prompt="Enter a number: "):
    while True:
        user_input = input(prompt)
        try:
            return int(user_input)  # 尝试转换为 int
        except ValueError:
            print("Invalid input. Enter a valid integer.")
            import ipdb; ipdb.set_trace()  # 启动调试模式

# 游戏控制类
class UndercoverGame:
    def __init__(self, num_agents, agent_name_list, num_rounds, game_data, game_uid = "", test_agent_id = None, game_number = 0, agent_history_context = []):
        self.num_agents = num_agents  # 玩家数量
        self.num_rounds = num_rounds  # 发言轮数
        self.agents = []              # 所有玩家
        self.agent_name_list = agent_name_list
        self.context = ""             # 游戏的上下文
        self.agent_context = ['' for i in range(num_agents)]
        self.round = 0                # 当前轮次
        # self.game_uid = f"game_{int(time.time())}"  # 基于时间戳生成唯一游戏ID
        if test_agent_id == None and RUN_PHASE == 'train':
            self.game_folder = f"game_log/{game_uid}/{game_number}"  # 基于时间戳生成唯一游戏ID
        elif test_agent_id != None and RUN_PHASE == 'test':
            self.game_folder = f"game_log/{game_uid}/{test_agent_id}/{game_number}"
        else:
            logger.error(f"Error: test_agent_id={test_agent_id}, RUN_PHASE={RUN_PHASE}")
            import ipdb;ipdb.set_trace()
        os.makedirs(self.game_folder, exist_ok=True)
        self.game_file_path = os.path.join(self.game_folder, "game_data.json")
            
        self.rule_prompt = open(f"games/undercover/undercover_rule_prompt.{LANGUAGE}.txt", "r").read()
        self.rule_prompt_rich = open(f"games/undercover/undercover_rule_prompt.{LANGUAGE}_rich.txt", "r").read()
        if TEST_MODEL_NAME == 'deepseek-v3' or TEST_MODEL_NAME == 'deepseek-v3-250324':
            self.rule_prompt_model = open(f"games/undercover/undercover_rule_prompt.{LANGUAGE}_v3.txt", "r").read()
            # self.rule_prompt_model = open(f"games/undercover/undercover_rule_prompt.{LANGUAGE}_zero_deepseek.txt", "r").read()
            # self.rule_prompt_model = open(f"games/undercover/undercover_rule_prompt.{LANGUAGE}_example_deepseek.txt", "r").read()
        elif TEST_MODEL_NAME == "claude-3-5-sonnet-20241022":
            self.rule_prompt_model = open(f"games/undercover/undercover_rule_prompt.{LANGUAGE}_claude3-5_v2.txt", "r").read()
            #  self.rule_prompt_model = open(f"games/undercover/undercover_rule_prompt.{LANGUAGE}_zero_claude.txt", "r").read()
            #   self.rule_prompt_model = open(f"games/undercover/undercover_rule_prompt.{LANGUAGE}_example_claude.txt", "r").read()
        elif TEST_MODEL_NAME == 'gpt-4o-2024-11-20' or TEST_MODEL_NAME == 'o1':
            self.rule_prompt_model = open(f"games/undercover/undercover_rule_prompt.{LANGUAGE}_4o.txt", "r").read()
            # self.rule_prompt_model = open(f"games/undercover/undercover_rule_prompt.{LANGUAGE}_zero_gpt.txt", "r").read()
        else:
            if WITH_HISTORY == 3 or WITH_HISTORY == 33:
                import ipdb;ipdb.set_trace()
        # self.load_game_data()
        self.game_data = game_data
        self.game_number = game_number
        # self.history_context = history_context
        self.final_result = [-1 for i in range(num_agents)]
        self.agent_history_context = agent_history_context
        self.agent_current_context = ["" for i in range(num_agents)]


    def load_game_data(self):
        with open("games/undercover/data.jsonl", "r") as file:
            data = file.readlines()
        self.game_data = [json.loads(line) for line in data]

    def assign_roles(self, specific_role = None):
        # gd = random.choice(self.game_data)
        gd = self.game_data[self.game_number]
        self.undercover_word = gd['undercover_word']
        self.civilian_word = gd['civilian_word']
        if specific_role != None:
            self.undercover_id = specific_role
        else:
            self.undercover_id = random.randint(0, self.num_agents - 1)
        self.undercover_name = self.agent_name_list[self.undercover_id]

        # 随机分配玩家身份和词
        for i in range(self.num_agents):
            if i == self.undercover_id:
                if WITH_HISTORY == 2 and RUN_PHASE == 'test':
                    game_rule = self.rule_prompt_rich
                    if i != 0:
                        import ipdb;ipdb.set_trace()
                elif WITH_HISTORY == 3 and RUN_PHASE == 'test':
                    game_rule = self.rule_prompt_model
                    if i != 0:
                        import ipdb;ipdb.set_trace()
                elif WITH_HISTORY == 33 and RUN_PHASE == 'test':
                    game_rule = self.rule_prompt_model
                    if i != 0:
                        import ipdb;ipdb.set_trace()
                else:
                    game_rule = self.rule_prompt
                agent = UndercoverAgent(agent_id=i, agent_name = self.agent_name_list[i],role='undercover', word=self.undercover_word, game_path=self.game_folder, model_name=TEST_MODEL_NAME, rule_prompt=game_rule, history_context = self.agent_history_context[i])
            else:
                agent = UndercoverAgent(agent_id=i, agent_name = self.agent_name_list[i], role='civilian', word=self.civilian_word, game_path=self.game_folder, model_name=BASELINE_MODEL_NAME, rule_prompt=self.rule_prompt, history_context = self.agent_history_context[i])

            self.agents.append(agent)
        if TEST_MODEL_NAME == 'human':
            logger.info(f"你的词是{self.agents[0].word}")

    def play_game(self, speak_order = None):
        # 游戏开始，玩家按照随机顺序发言
        survival_agent = [agent for agent in self.agents if not agent.is_voted_out]
        while len(survival_agent) > 3 and self.round < 6:
            self.round += 1
            logger.info(f"--- Round {self.round} ---")
            self.context += f"--- Round {self.round} ---\n"
            for i in range(self.num_agents):
                self.agent_context[i] += f"--- Round {self.round} ---\n"
            
            # 玩家发言
            speak_start_context = ''
            if LANGUAGE == "zh":
                speak_start_context += "[发言阶段]\n"
                logger.info("[发言阶段]")
            else:
                speak_start_context += f"[Description Phase]\n"
                logger.info("[Description Phase]")
            if speak_order == None or self.round > 3:
                speak_agent_ids = [aid for aid, agent in enumerate(self.agents) if not agent.is_voted_out]
                random.shuffle(speak_agent_ids)
            else:
                speak_agent_ids = speak_order[self.round-1]
            logger.info(f"[speak id] {str(speak_agent_ids)}")
            speak_user_name = [self.agent_name_list[aid] for aid in speak_agent_ids]
            speak_user = '本轮发言顺序是：' + ', '.join(speak_user_name) + '\n'
            logger.info(speak_user)
            speak_start_context += speak_user

            for i in range(self.num_agents):
                self.agent_context[i] += speak_start_context
            
            local_content = ["" for i in range(self.num_agents)]
            for aid in speak_agent_ids:
                if TEST_MODEL_NAME == 'human' and aid == 0:
                    message_WOThink = input("Please input your speech: ")
                    reason = ''
                else:
                    agent = self.agents[aid]
                    agent_input, reason_think = agent.get_input(self.agent_context[aid])
                    # if agent.model_name in ["DeepSeek-r1", 'o1']:
                    #     message_WOThink = extract_after_think(agent_input)  # 保存发言历史
                    #     reason, message_WOThink = extractSpeech(message_WOThink, aid)
                    # else:
                    reason, message_WOThink = extractSpeech(agent_input, aid, self.agent_name_list[aid])
                    filter_text_list = ['我的发言：','你的发言：', '我的发言:','你的发言:','我的实际发言内容：','你的实际发言内容：','我的实际发言内容:','你的实际发言内容:']
                    for filter_text in filter_text_list:
                        if message_WOThink.startswith(filter_text):
                            message_WOThink = message_WOThink[len(filter_text):].strip()
                    if '\n' in message_WOThink or '（' in message_WOThink or ':' in message_WOThink or '：' in message_WOThink or '发言内容' in message_WOThink:
                        logger.error(message_WOThink)
                        import ipdb;ipdb.set_trace()

                for i in range(self.num_agents):
                    speak_agent_name = self.agent_name_list[aid]
                    local_agent_context = ''
                    if i ==aid:
                        local_agent_context = f"<think>我是Agent {speak_agent_name}。{reason}</think>\n"
                    local_agent_context += f"Agent {speak_agent_name}: {message_WOThink}\n"
                    # local_content[i] += local_agent_context
                    self.agent_context[i] += local_agent_context
                logger.info(f"Agent {speak_agent_name} ({aid}): {message_WOThink}")
                # logger.info(f"Agent {agent.agent_id} thinks: {agent_input}")

            # for i in range(self.num_agents):
            #     self.agent_context[i] += local_content[i]
            
            if self.round >= 3:
                self.vote()
            logger.info("\n")

            survival_agent = [agent for agent in self.agents if not agent.is_voted_out]

            if self.agents[self.undercover_id].is_voted_out:
                self.final_result = [0 if agent.role == 'undercover' else 1 for agent in self.agents]
                result = ""
                if LANGUAGE == "zh":
                    logger.info(f"[游戏结束]\n[胜负结果]\n不同者 Agent {self.undercover_name} 已被淘汰！")
                    result = f'\n[游戏结束]\n[胜负结果]\n不同者 Agent {self.undercover_name} 已被淘汰！'
                else:
                    logger.info(f"[Game Finish]\nCivilians win! Undercover is Agent {self.undercover_name}")
                    result = f'\n[Game Finish]\nCivilians win! Undercover is {self.undercover_name}'
                self.context += result
                for i in range(self.num_agents):
                    self.agent_context[i] += result
                break
        
        if self.agents[self.undercover_id].is_voted_out == False:
            self.final_result = [1 if agent.role == 'undercover' else 0 for agent in self.agents]
            if LANGUAGE == "zh":
                logger.info(f"[游戏结束]\n[胜负结果]\n不同者 Agent {self.undercover_name} 获胜！")
                result = f'\n[游戏结束]\n[胜负结果]\n不同者 Agent {self.undercover_name} 获胜！'
            else:
                logger.info(f"[Game Finish]\n[Game Result]\nUndercover Agent {self.undercover_name} wins!")
                result += f'[Game Finish]\n[Game Result]\nUndercover Agent {self.undercover_name} win!'
            self.context += result
            for i in range(self.num_agents):
                self.agent_context[i] += result
            
        if LANGUAGE == 'zh':
            logger.info(f"不同者的秘密词是 {self.undercover_word}")
            logger.info(f"普通玩家的秘密词是 {self.civilian_word}")
            result = f"不同者的秘密词是 {self.undercover_word}\n普通玩家的秘密词是 {self.civilian_word}\n"
        else:
            logger.info(f"Undercover word is {self.undercover_word}")
            logger.info(f"Civilian word is {self.civilian_word}")
            result = f'Undercover word is {self.undercover_word}\nCivilian word is {self.civilian_word}\n'
        self.context += result
        for i in range(self.num_agents):
            self.agent_context[i] += result

        logger.info("[总结阶段]")
        for agent_id, agent in enumerate(self.agents):
            agent_state, reflection = agent.reflection(self.final_result[agent_id], self.agent_context[agent_id], self.undercover_name)
            logger.info(agent_state[:-1])
            # if agent.model_name == "DeepSeek-R1":
            #     reflection_WOThink = extract_after_think(reflection)
            #     self.agent_current_context[agent_id] = self.agent_context[agent_id] + agent_state + "[你的本局总结]\n" + reflection_WOThink
            #     logger.info(reflection_WOThink)
            # else:
            self.agent_current_context[agent_id] = self.agent_context[agent_id] + agent_state + "[你的本局总结]\n" + reflection
            logger.info(reflection)
            # import ipdb;ipdb.set_trace()

        # survival_agent = [agent for agent in self.agents if not agent.is_voted_out]
        # if len(survival_agent) == 3 or self.round >= 6:
        #     if any(agent.role == 'undercover' and not agent.is_voted_out for agent in survival_agent):
        #         logger.info("Undercover wins!")
        #     else:
        #         logger.info("Civilians win!")


    def vote(self):
        # 进行投票
        if LANGUAGE == "zh":
            phase = f"\n[投票阶段]\n"
            logger.info("\n[投票阶段]")
        else:
            phase = f"\n[Voting Phase]\n"
            logger.info("\n[Voting Phase]")
        self.context += phase
        for i in range(self.num_agents):
            self.agent_context[i] += phase

        # vote_context = self.context # 已经tou玩家的投票对其他玩家不可见
        votes = {}
        vote_agent_ids = [aid for aid, agent in enumerate(self.agents) if not agent.is_voted_out]
        random.shuffle(vote_agent_ids)
        if (TEST_MODEL_NAME == 'human') and (0 in vote_agent_ids):
            vote_agent_ids.remove(0)
            vote_agent_ids.insert(0, 0)

        # for aid,agent in enumerate(self.agents):
         # if not agent.is_voted_out:  # 只有未淘汰的玩家投票
        for aid in vote_agent_ids:
            agent = self.agents[aid]
            if TEST_MODEL_NAME == 'human' and aid == 0:
                vote_id = safe_input("Please Vote, input the number: ")
            else:
                vote_id = agent.vote_for(self.agents, self.agent_name_list, self.agent_context[aid])
            if vote_id is not None:
                if vote_id not in votes:
                    votes[vote_id] = 1
                else:
                    votes[vote_id] += 1
                if LANGUAGE == "zh":
                    logger.info(f"玩家 {agent.agent_name} ({aid}) 投票给 {self.agent_name_list[vote_id]} ({vote_id})")
                    result = f"Agent {agent.agent_name} 投票给 {self.agent_name_list[vote_id]}\n"
                else:
                    logger.info(f"Agent {agent.agent_name} votes for {self.agent_name_list[vote_id]}")
                    result = f"Agent {agent.agent_name} votes for {self.agent_name_list[vote_id]}\n"
                self.context += result
                for i in range(self.num_agents):
                    self.agent_context[i] += result

        if len(votes)>0:
            max_votes = max(votes.values())
            if list(votes.values()).count(max_votes) > 1:
                if LANGUAGE == "zh":
                    logger.info(f"投票存在平票，没有玩家出局！")
                    result = "投票存在平票，没有玩家出局！\n"
                elif LANGUAGE == "en":
                    logger.info(f"Voting is tied, no one is voted out!")
                    result = "Voting is tied, no one is voted out!"
                self.context += result
                for i in range(self.num_agents):
                    self.agent_context[i] += result
                return
            voted_out_agent = max(votes, key=votes.get)
            voted_out_agent_name = self.agent_name_list[voted_out_agent]
            if LANGUAGE == "zh":
                logger.info(f"玩家 {voted_out_agent_name} ({voted_out_agent}) 已出局！")
                result = f"Agent {voted_out_agent_name} 已出局！\n"
            else:
                logger.info(f"Agent {voted_out_agent_name} is voted out.")
                result = f"Agent {voted_out_agent_name} is voted out."
            self.context += result
            for i in range(self.num_agents):
                self.agent_context[i] += result
            # 标记被淘汰的玩家
            self.agents[voted_out_agent].is_voted_out = True
            # flag = 0
            # for agent in self.agents:
            #     if agent.agent_id == voted_out_agent:
            #         agent.is_voted_out = True
            #         flag = 1
            # if flag == 0:
            #     logger.info("Voted Agent Note Found!")
            #     import ipdb;ipdb.set_trace()
            
    
    def save_history(self):
        # 保存游戏历史

        game_data = {
            "num_agents": self.num_agents,
            "num_rounds": self.num_rounds,
            "agents": [agent.__dict__ for agent in self.agents],
            "context": self.context,
            "agent_context": self.agent_context,
            "round": self.round,
        }
        
        with open(self.game_file_path, "w", encoding = 'utf-8') as file:
            json.dump(game_data, file, indent=4, ensure_ascii=False)
        
        # # 分别调用每个玩家的保存历史方法
        # for agent in self.agents:
        #     agent.save_history("game")

def save_summary(GAME_ID, agent_context, agent_name_list):
    result_list = []
    for aid in range(len(agent_context)):
        # history =  f"\n游戏已经进行了{round}局，你赢了{win_situation[aid]}局，输了{round - win_situation[aid]}局。\n[历史游戏记录]\n{agent_context[aid]}"
        result_dict = {"agent_id":aid, 'agent_name': agent_name_list[aid], "history_context":agent_context[aid]}
        result_list.append(result_dict)
    
    with open(f"game_log/{GAME_ID}/summary.jsonl", "w", encoding = 'utf-8') as file:
        for data in result_list:
            file.write(json.dumps(data, ensure_ascii=False) + "\n")
        # json.dump(result_list, file, indent=4, ensure_ascii=False)
    return

import string

def run_train():

    agent_context = ["" for i in range(AGENT_NUM)]
    win_situation = [0 for i in range(AGENT_NUM)]

    logger.info(f"[Model] baseline = {BASELINE_MODEL_NAME}, test = {TEST_MODEL_NAME}")

    with open("games/undercover/data.jsonl", "r") as file:
        data = file.readlines()
        game_data = [json.loads(line) for line in data][:ROUND]
    random.shuffle(game_data)

    random_strings = set()
    while len(random_strings) < AGENT_NUM:
        random_strings.add(''.join(random.choices(string.ascii_lowercase, k=5)))
    random_string_list = list(random_strings)
    logger.info(f"[Agent Name] {'-'.join(random_string_list)}")

    difference_list = random.sample(range(5), 3)
    
    for i in range(ROUND):
        logger.info(f"******\nGame No: {i}")
        input_history = ['' for i in range(AGENT_NUM)]
        if i > 0:
            for aid in [0]:#range(AGENT_NUM):
                input_history[aid] =  f"\n游戏已经进行了{i}局，你赢了{win_situation[aid]}局，输了{i - win_situation[aid]}局。\n[历史游戏记录]\n{agent_context[aid]}\n\n[第{i+1}局]\n"
        game = UndercoverGame(num_agents=AGENT_NUM, agent_name_list = random_string_list, num_rounds=6, game_uid = GAME_ID , game_number = i, game_data = game_data, agent_history_context = input_history)
        if i in difference_list:
            game.assign_roles(specific_role = 0)
        else:
            difference_role = random.randint(1, AGENT_NUM-1)
            game.assign_roles(specific_role = difference_role)
        game.play_game()
        game.save_history()
        result = game.final_result
        context = game.agent_current_context
        for agent_id, win in enumerate(result):
            win_situation[agent_id] += win
            agent_context[agent_id] += f"[第{i+1}局开始]\n{context[agent_id]}\n[第{i+1}局结束]\n\n"
        time.sleep(3)
    
    print("[Summary Info]")
    # logger.info(f"Game Result: {win_situation}\n")
    logger.info(f"Game Context:")
    for i in range(AGENT_NUM):
        logger.info(f"Agent {i}\n{agent_context[i]}\n")
    save_summary(GAME_ID, agent_context, random_string_list)
    

def run_test():

	# history_data_path = "./game_log/game_1741060648_claude/summary.jsonl"
    if TEST_MODEL_NAME == 'azure-o1-preview-2024-09-12':
        # history_data_path = './game_log/game_1741529451-o1/summary.jsonl'
         # no order t = 0.8
        # history_data_path = './game_log/game_1741772857_model=azure-o1-preview-2024-09-12/summary.jsonl'
         # no order t = 1.0
        #  history_data_path = './game_log/game_1741789341_t=1_model=azure-o1-preview-2024-09-12/summary.jsonl'
        # t=1 randomOrder, name change to xxxxx
        history_data_path = './game_log/game_1741932954_t=1_RandomOrder_model=azure-o1-preview-2024-09-12/summary.jsonl'
    elif TEST_MODEL_NAME == 'claude-3-5-sonnet-20241022':
        # history_data_path = "./game_log/game_1741242643-claude3.5/summary.jsonl"\
        # no order t = 0.8
        # history_data_path = './game_log/game_1741773191_model=claude-3-5-sonnet-20241022/summary.jsonl'
        # no order t = 1
        # history_data_path = './game_log/game_1741789332_t=1_model=claude-3-5-sonnet-20241022/summary.jsonl'
        # t=1 randomOrder, name change to xxxxx
        history_data_path = './game_log/game_1741930812_t=1_RandomOrder_model=claude-3-5-sonnet-20241022/summary.jsonl'
    elif TEST_MODEL_NAME == 'o1':
        history_data_path = './game_log/Order/game_1741529451-o1/history.jsonl'
    elif TEST_MODEL_NAME == 'claude-3-7-sonnet-20250219':
        history_data_path = './game_log/Order/game_1741242643-claude3.5/history.jsonl'
    elif TEST_MODEL_NAME == 'deepseek-r1':
        # history_data_path = './game_log/RandomSeqSpeak/gpt-4o__deepseek-r1/t=1_RandomOrder/train/1742455386/summary.jsonl'
        # history_data_path = './game_log/RandomSeqSpeak/deepseek-r1__deepseek-r1/t=1_RandomOrder/train/1742569631/summary.jsonl'
        # new easy rule
        history_data_path = './game_log/RandomSeqSpeak/new_rule_0323/deepseek-r1__deepseek-r1/t=1_RandomOrder/train/1742743815/summary.jsonl'
    elif TEST_MODEL_NAME == 'deepseek-v3':
        history_data_path = './game_log/RandomSeqSpeak/new_rule_0323/deepseek-v3__deepseek-v3/t=1_RandomOrder/train/1742825699/summary.jsonl'
    elif TEST_MODEL_NAME == 'azure-gpt-4o-2024-11-20':
        # history_data_path = './game_log/RandomSeqSpeak/new_rule_0323/azure-gpt-4o-2024-11-20__azure-gpt-4o-2024-11-20/t=1_RandomOrder/train/1743404764/summary.jsonl'
        history_data_path = './game_log/RandomSeqSpeak/new_rule_0323/azure-gpt-4o-2024-11-20__azure-gpt-4o-2024-11-20/t=1_RandomOrder/train/1743411327/summary.jsonl'
    elif TEST_MODEL_NAME == 'human':
        history_data = []
        history_data_path = None
    else:
        history_data_path = None
        history_data = []
        if WITH_HISTORY == 1:
            import ipdb;ipdb.set_trace()


    if history_data_path is not None:
        with open(history_data_path, "r") as file:
            history_data = file.readlines()
        history_data = [json.loads(line) for line in history_data]

    # game_data = read_data()
    # import ipdb;ipdb.set_trace()
    game_data = [{'undercover_word': '雨衣', 'civilian_word': '雨伞'}, {'undercover_word': '短袖', 'civilian_word': '背心'}, {'undercover_word': '芥末', 'civilian_word': '辣椒'}, {'undercover_word': '梨', 'civilian_word': '苹果'}, {'undercover_word': '杂志', 'civilian_word': '书籍'},\
                  {'undercover_word': '狗', 'civilian_word': '狼'}, {'undercover_word': '小提琴', 'civilian_word': '吉他'}, {'undercover_word': '菠萝', 'civilian_word': '西瓜'}, {'undercover_word': '靠枕', 'civilian_word': '枕头'}, {'undercover_word': '游艇', 'civilian_word': '帆船'},\
                  {'undercover_word': '博客', 'civilian_word': '日记'}, {'undercover_word': '被子', 'civilian_word':'床单'}, {'undercover_word': '公主', 'civilian_word': '皇后'}, {'undercover_word': '气球', 'civilian_word': '风筝'}, {'undercover_word': '滑旱冰', 'civilian_word': '滑滑板'},\
                   {'undercover_word': '老虎', 'civilian_word': '猫'}, {'undercover_word': '湖', 'civilian_word': '河'}, {'undercover_word': '汽车', 'civilian_word': '卡车'}, {'undercover_word': '领结', 'civilian_word': '领带'}, {'undercover_word': '月季', 'civilian_word': '玫瑰'}, \
                    {'undercover_word': '口罩', 'civilian_word': '面具'}, {'undercover_word': '病毒', 'civilian_word': '细菌'}, {'undercover_word': '唇膏', 'civilian_word': '口红'}, {'undercover_word': '手铐', 'civilian_word': '手镯'}, {'undercover_word': '白酒', 'civilian_word': '啤酒'}, \
                    {'undercover_word': '苦瓜', 'civilian_word': '黄瓜'}, {'undercover_word': '包子', 'civilian_word': '饺子'}, {'undercover_word': '同桌', 'civilian_word': '同学'}, {'undercover_word': '跳舞', 'civilian_word': '太极'}, {'undercover_word': '扶梯', 'civilian_word': '电梯'}, \
                    {'undercover_word': '油条','civilian_word': '麻花'}, {'undercover_word': '水盆', 'civilian_word': '水桶'}, {'undercover_word': '奖牌', 'civilian_word': '金牌'}, {'undercover_word': '麦克风', 'civilian_word': '扩音器'}, {'undercover_word':'双胞胎', 'civilian_word': '龙凤胎'},\
                    {'undercover_word': '元旦', 'civilian_word': '圣诞'}, {'undercover_word': '纸巾', 'civilian_word': '手绢'}, {'undercover_word': '作家', 'civilian_word': '编剧'}, {'undercover_word': '贴画', 'civilian_word': '壁纸'}, {'undercover_word': '侦探', 'civilian_word': '警察'}, \
                    {'undercover_word': '背心', 'civilian_word': '短袖'}, {'undercover_word': '脚', 'civilian_word': '手'}, {'undercover_word': '莲子粥', 'civilian_word': '八宝粥'}, {'undercover_word': '樱桃', 'civilian_word': '草莓'}, {'undercover_word': '数学课本', 'civilian_word': '语文课本'}, \
                    {'undercover_word': '泡泡糖', 'civilian_word': '棒棒糖'}, {'undercover_word': '琵琶', 'civilian_word':'吉他'}, {'undercover_word': '洗发露', 'civilian_word': '护发素'}, {'undercover_word': '座机', 'civilian_word': '手机'}, {'undercover_word': '童话', 'civilian_word': '神话'}]
    
    # game_data = [{'undercover_word': '雨衣', 'civilian_word': '雨伞'}, {'undercover_word': '短袖', 'civilian_word': '背心'}, {'undercover_word': '辣椒', 'civilian_word': '芥末'},\
    #             {'undercover_word': '苹果', 'civilian_word': '梨'}, {'undercover_word': '杂志', 'civilian_word': '书籍'}, {'undercover_word': '狼', 'civilian_word': '狗'},\
    #             {'undercover_word': '小提琴', 'civilian_word': '吉他'}, {'undercover_word': '西瓜', 'civilian_word': '菠萝'}, {'undercover_word': '靠枕', 'civilian_word': '枕头'}, \
    #             {'undercover_word': '帆船', 'civilian_word': '游艇'}, {'undercover_word': '日记', 'civilian_word': '博客'}, {'undercover_word': '床单', 'civilian_word': '被子'}, \
    #             {'undercover_word': '公主', 'civilian_word': '皇后'}, {'undercover_word': '风筝', 'civilian_word': '气球'}, {'undercover_word': '滑旱冰', 'civilian_word': '滑滑板'},\
    #             {'undercover_word': '老虎', 'civilian_word': '猫'}, {'undercover_word': '河', 'civilian_word': '湖'}, {'undercover_word': '卡车', 'civilian_word': '汽车'}, \
    #             {'undercover_word': '领结', 'civilian_word': '领带'}, {'undercover_word': '月季', 'civilian_word': '玫瑰'}]

    

    logger.info(f"[Model] baseline = {BASELINE_MODEL_NAME}, test = {TEST_MODEL_NAME}")
    logger.info(f"With History: {WITH_HISTORY}")
    logger.info(f"Range: {start}, {end}")

    random_strings = set()
    while len(random_strings) < AGENT_NUM* AGENT_NUM * (end-start):
        random_strings.add(''.join(random.choices(string.ascii_lowercase, k=5)))
    random_string_list = list(random_strings)
    logger.info(f"[Agent Name] {'-'.join(random_string_list)}")

    if TEST_MODEL_NAME == 'human':
        difference_list = random.sample(range(start, start+10), 6)
        if end-start>10:
            logger.info("human evaluation > 10 rounds!")
            return
        
    # agent_id_list = [i for i in range(AGENT_NUM)]
    # game_order_list = [random.sample(agent_id_list, len(agent_id_list)) for _ in range((end - start) * 3)]
    # 10-20
    # game_order_list = [[2, 3, 0, 1, 4], [0, 3, 1, 2, 4], [4, 3, 2, 0, 1], [0, 2, 1, 3, 4], [1, 3, 4, 0, 2], [4, 1, 2, 0, 3], [2, 0, 3, 1, 4], [1, 0, 4, 2, 3], [1, 3, 0, 4, 2], [0, 3, 1, 4, 2], [1, 0, 2, 4, 3], [1, 0, 4, 3, 2], [0, 1, 3, 2, 4], [1, 4, 3, 2, 0], [1, 4, 2, 3, 0], [0, 2, 3, 1, 4], [1, 3, 4, 2, 0], [0, 3, 1, 2, 4], [2, 1, 3, 0, 4], [1, 4, 2, 0, 3], [1, 2, 4, 3, 0], [3, 0, 1, 4, 2], [3, 4, 0, 1, 2], [1, 4, 3, 2, 0], [3, 2, 1, 4, 0], [1, 0, 2, 4, 3], [3, 4, 0, 2, 1], [1, 0, 2, 4, 3], [3, 4, 2, 1, 0], [2, 0, 3, 1, 4]]
    # 13-20
    # game_order_list = [[0, 3, 1, 4, 2], [1, 0, 2, 4, 3], [1, 0, 4, 3, 2], [0, 1, 3, 2, 4], [1, 4, 3, 2, 0], [1, 4, 2, 3, 0], [0, 2, 3, 1, 4], [1, 3, 4, 2, 0], [0, 3, 1, 2, 4], [2, 1, 3, 0, 4], [1, 4, 2, 0, 3], [1, 2, 4, 3, 0], [3, 0, 1, 4, 2], [3, 4, 0, 1, 2], [1, 4, 3, 2, 0], [3, 2, 1, 4, 0], [1, 0, 2, 4, 3], [3, 4, 0, 2, 1], [1, 0, 2, 4, 3], [3, 4, 2, 1, 0], [2, 0, 3, 1, 4]]
    # 0-7 7-10
    game_order_list = [[2, 0, 3, 1, 4], [1, 0, 4, 3, 2], [4, 0, 3, 1, 2],
                        [3, 1, 2, 4, 0], [0, 4, 2, 3, 1], [3, 4, 2, 1, 0],
                        [4, 2, 0, 1, 3], [0, 3, 2, 1, 4], [0, 2, 3, 4, 1],
                        [0, 3, 4, 2, 1], [0, 4, 2, 3, 1], [4, 2, 0, 3, 1],
                        [0, 1, 4, 3, 2], [3, 2, 1, 0, 4], [2, 1, 3, 4, 0],
                        [0, 1, 4, 3, 2], [3, 2, 1, 0, 4], [3, 1, 4, 0, 2],
                        [1, 2, 4, 3, 0], [0, 4, 1, 3, 2], [3, 2, 4, 0, 1],
                        [0, 3, 4, 1, 2], [0, 1, 2, 4, 3], [4, 2, 3, 0, 1], 
                        [4, 1, 0, 2, 3], [0, 2, 4, 3, 1], [2, 3, 1, 0, 4], 
                        [1, 0, 3, 2, 4], [4, 3, 1, 0, 2], [0, 2, 3, 4, 1],
                        [2, 3, 0, 1, 4], [0, 3, 1, 2, 4], [4, 3, 2, 0, 1], 
                        [0, 2, 1, 3, 4], [1, 3, 4, 0, 2], [4, 1, 2, 0, 3], 
                        [2, 0, 3, 1, 4], [1, 0, 4, 2, 3], [1, 3, 0, 4, 2], 
                        [0, 3, 1, 4, 2], [1, 0, 2, 4, 3], [1, 0, 4, 3, 2],
                        [0, 1, 3, 2, 4], [1, 4, 3, 2, 0], [1, 4, 2, 3, 0], 
                        [0, 2, 3, 1, 4], [1, 3, 4, 2, 0], [0, 3, 1, 2, 4], 
                        [2, 1, 3, 0, 4], [1, 4, 2, 0, 3], [1, 2, 4, 3, 0], 
                        [3, 0, 1, 4, 2], [3, 4, 0, 1, 2], [1, 4, 3, 2, 0], 
                        [3, 2, 1, 4, 0], [1, 0, 2, 4, 3], [3, 4, 0, 2, 1], 
                        [1, 0, 2, 4, 3], [3, 4, 2, 1, 0], [2, 0, 3, 1, 4],
                        [0, 1, 4, 2, 3], [1, 3, 2, 0, 4], [3, 1, 0, 4,2], [0, 3, 1, 4, 2], [4, 3, 2, 1, 0], [1, 4, 2, 0, 3], 
                       [1, 4, 0, 3, 2], [4, 2, 1, 0, 3], [0, 3, 2, 4, 1], [0, 1, 3, 2, 4], [1, 4, 0, 3, 2], [4, 0, 1, 3, 2], 
                       [3, 2, 0, 4, 1], [4, 2, 3, 0, 1], [3, 0, 1, 4, 2], [2, 3, 4, 0, 1], [4, 3, 1, 2, 0], [4, 0, 3, 2, 1], 
                       [0, 3, 4, 1, 2], [2, 4, 1, 0, 3], [2, 3, 1, 4, 0], [4, 2, 3, 0, 1], [1, 3, 2, 0, 4], [1, 2, 0, 3, 4], 
                       [4, 0, 1, 2, 3], [1, 3, 2, 4, 0], [3, 2, 4, 0, 1], [0, 1, 3, 2, 4], [2, 3, 4, 1, 0], [3, 2, 1, 0, 4], 
                       [2, 3, 0, 1, 4], [1, 3, 2, 0, 4], [2, 1, 4, 3, 0], [3, 4, 2, 1, 0], [4, 0, 1, 3, 2],  [4, 3, 2, 0, 1]]
    # 7-10
    # game_order_list = [[0, 3, 4, 1, 2], [0, 1, 2, 4, 3], [4, 2, 3, 0, 1], [4, 1, 0, 2, 3], [0, 2, 4, 3, 1], [2, 3, 1, 0, 4], [1, 0, 3, 2, 4], [4, 3, 1, 0, 2], [0, 2, 3, 4, 1]]
    # 20-32
    # game_order_list = [[0, 1, 4, 2, 3], [1, 3, 2, 0, 4], [3, 1, 0, 4,2], [0, 3, 1, 4, 2], [4, 3, 2, 1, 0], [1, 4, 2, 0, 3], 
    #                    [1, 4, 0, 3, 2], [4, 2, 1, 0, 3], [0, 3, 2, 4, 1], [0, 1, 3, 2, 4], [1, 4, 0, 3, 2], [4, 0, 1, 3, 2], 
    #                    [3, 2, 0, 4, 1], [4, 2, 3, 0, 1], [3, 0, 1, 4, 2], [2, 3, 4, 0, 1], [4, 3, 1, 2, 0], [4, 0, 3, 2, 1], 
    #                    [0, 3, 4, 1, 2], [2, 4, 1, 0, 3], [2, 3, 1, 4, 0], [4, 2, 3, 0, 1], [1, 3, 2, 0, 4], [1, 2, 0, 3, 4], 
    #                    [4, 0, 1, 2, 3], [1, 3, 2, 4, 0], [3, 2, 4, 0, 1], [0, 1, 3, 2, 4], [2, 3, 4, 1, 0], [3, 2, 1, 0, 4], 
    #                    [2, 3, 0, 1, 4], [1, 3, 2, 0, 4], [2, 1, 4, 3, 0], [3, 4, 2, 1, 0], [4, 0, 1, 3, 2],  [4, 3, 2, 0, 1]]
    if len(game_order_list) != 3*32:
        import ipdb;ipdb.set_trace()

    for i in [0]:
        for j in range(start, end):
            agent_context = ["" for i in range(AGENT_NUM)]
            start_index = (i*(end-start)+j)*AGENT_NUM
            agent_name_list = random_string_list[start_index:start_index+AGENT_NUM]
            if WITH_HISTORY == 1:
                history_context = history_data[i]["history_context"] + f"\n\n[第{ROUND+1}局]\n"
                replace_name_dict = {history_data[x]['agent_name']:agent_name_list[x] for x in range(AGENT_NUM)}
                logger.info(f"Replace Agent Names\n{str(replace_name_dict)}")
                pattern = re.compile('|'.join(map(re.escape, replace_name_dict.keys())))
                agent_context[i] = pattern.sub(lambda match: replace_name_dict[match.group(0)], history_context)
            logger.info(f"******\nAgent No: {i}. Game {j}")
            game = UndercoverGame(num_agents=AGENT_NUM, agent_name_list = agent_name_list, num_rounds=6, game_data = game_data, game_uid = GAME_ID, test_agent_id = i, game_number = j, agent_history_context = agent_context)
            if TEST_MODEL_NAME == 'human':
                if j in difference_list:
                    game.assign_roles(specific_role=i)
                else:
                    difference_role = random.randint(1, AGENT_NUM-1)
                    game.assign_roles(specific_role=difference_role)
            else:
                game.assign_roles(specific_role=i)
            # game_counter = j-start
            game.play_game(game_order_list[j*3:(j+1)*3])
            game.save_history()


if __name__ == '__main__':
    if RUN_PHASE == 'train':
        run_train()
    elif RUN_PHASE == 'test':
        run_test()