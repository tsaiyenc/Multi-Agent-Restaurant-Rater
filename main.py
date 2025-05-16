from __future__ import annotations
from autogen import ConversableAgent, register_function
import os, sys, re, ast
from typing import Dict, List, get_type_hints
import logging
from datetime import datetime

# 設定 logging
log_filename = f"logs/logger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
os.makedirs("logs", exist_ok=True)

# 設定檔案處理器
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)

# 設定根記錄器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# 關閉其他處理器
logging.getLogger().handlers = []

SCORE_KEYWORDS: dict[int, list[str]] = {
    1: ["awful", "horrible", "disgusting"],
    2: ["bad", "unpleasant", "offensive"],
    3: ["average", "uninspiring", "forgettable"],
    4: ["good", "enjoyable", "satisfying"],
    5: ["awesome", "incredible", "amazing"]
}

# ────────────────────────────────────────────────────────────────
# 0. OpenAI API key setup ── *Do **not** modify this block.*
# ────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ Set the OPENAI_API_KEY environment variable first.")
LLM_CFG = {"config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]}

# ────────────────────────────────────────────────────────────────
# 1. Utility data structures & helper functions
# ────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()

def fetch_restaurant_data(restaurant_name: str) -> dict[str, list[str]]:
    data = {}
    target = normalize(restaurant_name)
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            name, review = line.split('.', 1)
            if normalize(name) == target:
                data.setdefault(name.strip(), []).append(review.strip())
    return data

def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> dict[str, str]:
    """Geometric-mean rating rounded to 3 dp."""
    n = len(food_scores)
    if n == 0 or n != len(customer_service_scores):
        raise ValueError("food_scores and customer_service_scores must be non-empty and same length")
    total = sum(((f**2 * s)**0.5) * (1 / (n * (125**0.5))) * 10 for f, s in zip(food_scores, customer_service_scores))
    return {restaurant_name: f"{total:.3f}"}

# register functions
fetch_restaurant_data.__annotations__ = get_type_hints(fetch_restaurant_data)
calculate_overall_score.__annotations__ = get_type_hints(calculate_overall_score)

# ──────────────────────────────────────────────
# 2. Agent setup
# ──────────────────────────────────────────────

def build_agent(name, msg):
    return ConversableAgent(name=name, system_message=msg, llm_config=LLM_CFG)

DATA_FETCH = build_agent(
    "fetch_agent",
    'Return JSON {"call":"fetch_restaurant_data","args":{"restaurant_name":"<name>"}}'
)
ANALYZER = build_agent(
    "review_analyzer_agent",
    """You are a professional review analyst. Your task is:
    1. Analyze the given review for food and service related adjectives
    2. Convert adjectives to scores (1-5) based on the following criteria:
    {SCORE_KEYWORDS}
    3. For words not exactly matching the keywords:
       - Consider similar words and synonyms
       - Handle common typos and variations
       - Use context to determine the appropriate score
    4. Response format must strictly follow this JSON format:
    {
        "food_score": <score>,
        "service_score": <score>
    }
    5. If you cannot determine a score, use -1
    6. Only return the JSON, no additional text
    7. If a word is ambiguous, use the most appropriate score based on context"""
)
SCORER = build_agent(
    "scoring_agent",
    "Given name + two lists. Reply only: calculate_overall_score(...)"
)
ENTRY = build_agent("entry", "Coordinator")

# register functions
register_function(
    fetch_restaurant_data,
    caller=DATA_FETCH,
    executor=ENTRY,
    name="fetch_restaurant_data",
    description="Fetch reviews from specified data file by name.",
)
register_function(
    calculate_overall_score,
    caller=SCORER,
    executor=ENTRY,
    name="calculate_overall_score",
    description="Compute final rating via geometric mean.",
)


# ────────────────────────────────────────────────────────────────
# 3. Conversation helpers
# ────────────────────────────────────────────────────────────────

def parse_scores(chat_summary: str, logger) -> dict:
    """解析聊天摘要中的分數"""
    try:
        return ast.literal_eval(chat_summary)
    except:
        logger.error(f"無法解析分數: {chat_summary}")
        return None

def parse_data_fetch_response(chat_history: list, logger) -> dict:
    """解析 DATA_FETCH 的回應"""
    for past in reversed(chat_history):
        try:
            data = ast.literal_eval(past["content"])
            if isinstance(data, dict) and data and not ("call" in data):
                logger.debug(f"找到餐廳資料: {data}")
                return data
        except:
            continue
    return None

def run_chat_sequence(entry: ConversableAgent, sequence: list[dict]) -> str:
    ctx = {**getattr(entry, "_initiate_chats_ctx", {})}
    for step in sequence:
        logger.debug(f"{'='*50}")
        logger.debug(f"正在與 {step['recipient'].name} 進行對話...")
        
        if step["recipient"] is ANALYZER and "reviews_dict" in ctx:
            # 處理每條評論
            all_scores = []
            restaurant_name = next(iter(ctx["reviews_dict"]))
            reviews = ctx["reviews_dict"][restaurant_name]
            
            for review in reviews:
                logger.debug(f"分析評論: {review}")
                msg = f"Analyze this review: {review}"
                retry_count = 0
                scores = None
                
                while retry_count < 3 and scores is None:
                    chat = entry.initiate_chat(
                        step["recipient"], message=msg,
                        summary_method=step.get("summary_method", "last_msg"),
                        max_turns=step.get("max_turns", 1),
                    )
                    scores = parse_scores(chat.summary, logger)
                    
                    if scores is None:
                        retry_count += 1
                        if retry_count < 3:
                            logger.error(f"無法解析分數 (嘗試 {retry_count}/3): {chat.summary}")
                            msg = f"Analyze this review again: {review}"
                        else:
                            logger.error(f"無法解析分數，已達最大重試次數")
                
                if scores is not None:
                    all_scores.append(scores)
                    logger.debug(f"分數: {scores} 已添加")
            
            # 整理所有分數
            food_scores = [score["food_score"] for score in all_scores]
            service_scores = [score["service_score"] for score in all_scores]
            ctx["analyzer_output"] = f"food_scores={food_scores}\ncustomer_service_scores={service_scores}"
            continue
            
        msg = step["message"].format(**ctx)
        logger.debug(f"發送訊息: {msg}")
        logger.debug(f"{'='*50}")
        
        chat = entry.initiate_chat(
            step["recipient"], message=msg,
            summary_method=step.get("summary_method", "last_msg"),
            max_turns=step.get("max_turns", 2),
        )
        out = chat.summary
        
        logger.debug(f"{step['recipient'].name} 的回應:")
        logger.debug(f"{'-'*30}")
        logger.debug(out)
        logger.debug(f"{'-'*30}")
        
        # Data fetch output
        if step["recipient"] is DATA_FETCH:
            logger.debug("正在處理 DATA_FETCH 的輸出...")
            retry_count = 0
            data = None
            
            while retry_count < 3 and data is None:
                data = parse_data_fetch_response(chat.chat_history, logger)
                
                if data is None:
                    retry_count += 1
                    if retry_count < 3:
                        logger.error(f"無法解析餐廳資料 (嘗試 {retry_count}/3)")
                        chat = entry.initiate_chat(
                            step["recipient"], message=msg,
                            summary_method=step.get("summary_method", "last_msg"),
                            max_turns=step.get("max_turns", 2),
                        )
                    else:
                        logger.error("無法解析餐廳資料，已達最大重試次數")
            
            if data is not None:
                ctx.update({"reviews_dict": data, "restaurant_name": next(iter(data))})
    return out

ConversableAgent.initiate_chats = lambda self, seq: run_chat_sequence(self, seq)

# ──────────────────────────────────────────────
# 4. Main entry
# ──────────────────────────────────────────────

def main(user_query: str, data_path: str = "restaurant-data.txt"):
    global DATA_PATH
    DATA_PATH = data_path
    logger.info(f"初始化系統...")
    logger.info(f"使用資料檔案: {data_path}")
    logger.info(f"使用者查詢: {user_query}")
    
    agents = {"data_fetch": DATA_FETCH, "analyzer": ANALYZER, "scorer": SCORER}
    chat_sequence = [
        {"recipient": agents["data_fetch"], 
         "message": "Find reviews for this query: {user_query}", 
         "summary_method": "last_msg", 
         "max_turns": 2},

        {"recipient": agents["analyzer"], 
         "message": "Analyze reviews", 
         "summary_method": "last_msg", 
         "max_turns": 1},

        {"recipient": agents["scorer"], 
         "message": "{analyzer_output}", 
         "summary_method": "last_msg", 
         "max_turns": 2},
    ]
    ENTRY._initiate_chats_ctx = {"user_query": user_query}
    logger.info("開始執行對話序列...")
    result = ENTRY.initiate_chats(chat_sequence)
    logger.info("最終結果:")
    logger.info(f"{'='*50}")
    logger.info(result)
    logger.info(f"{'='*50}")
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python main.py path/to/data.txt "How good is Subway?" ')
        sys.exit(1)

    path = sys.argv[1]
    query = sys.argv[2]
    main(query, path)
