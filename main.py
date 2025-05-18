from __future__ import annotations
from autogen import ConversableAgent, register_function
import os, sys, re, ast
from typing import Dict, List, get_type_hints
import logging
from datetime import datetime
from difflib import SequenceMatcher

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
    logger.debug(f"準備計算總分: {restaurant_name}, {food_scores}, {customer_service_scores}")
    n = len(food_scores)
    if n == 0 or n != len(customer_service_scores):
        raise ValueError("food_scores and customer_service_scores must be non-empty and same length")
    total = sum(((f**2 * s)**0.5) * (1 / (n * (125**0.5))) * 10 for f, s in zip(food_scores, customer_service_scores))
    logger.debug(f"計算出總分: {total:.3f}")
    return {restaurant_name: f"{total:.3f}"}

def string_similarity(a: str, b: str) -> float:
    """計算兩個字串的相似度
    
    Args:
        a: 第一個字串
        b: 第二個字串
    
    Returns:
        相似度 (0-1 之間的浮點數)
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

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

SIMILARITY_AGENT = build_agent(
    "similarity_agent",
    """You are a professional word similarity analyzer. Your task is:
    1. Given a list of adjectives and a list of keywords, find the most similar keyword from the keywords list that best matches ANY of the given adjectives
    2. Consider semantic meaning, context, and common usage for each adjective
    3. Compare all adjectives with all keywords and choose the best overall match
    4. Response format must strictly follow this JSON format:
    {
        "most_similar_keyword": "<keyword>"
    }
    5. Only return the JSON, no additional text
    6. If no similar keyword is found, return the closest match based on context"""
)

REVIEW_EXTRACTOR = build_agent(
    "review_extractor_agent",
    """You are a professional review extractor. Your task is:
    1. Extract descriptive elements related to food and customer service from the review, including:
       - Adjectives (e.g., "delicious", "friendly")
       - Adverbs (e.g., "consistently", "quickly")
       - Descriptive phrases (e.g., "above and beyond", "out of this world")
       - Short descriptive fragments (e.g., "made with care", "worth the wait")
    2. DO NOT generate or infer any descriptions that are not directly present in the review
    3. IMPORTANT: Pay special attention to negative modifiers before adjectives (e.g., "not delicious", "never friendly")
       - If an adjective is preceded by a negative word, include the negative word as part of the description
       - This helps maintain the true sentiment of the review
    4. Response format must strictly follow this JSON format:
    {
        "food_descriptions": ["<description1>", "<description2>", ...],
        "service_descriptions": ["<description1>", "<description2>", ...]
    }
    5. Only return the JSON, no additional text
    6. If no descriptions are found for a category, return an empty list
    7. IMPORTANT: Only extract descriptions that are explicitly present in the review text"""
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
    caller=REVIEW_EXTRACTOR,
    executor=ENTRY,
    name="calculate_overall_score",
    description="Compute final rating via geometric mean.",
)


# ────────────────────────────────────────────────────────────────
# 3. Conversation helpers
# ────────────────────────────────────────────────────────────────


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

def get_score_from_adjectives(adjectives: list[str], entry: ConversableAgent, logger) -> int | None:
    """從描述詞列表中獲取評分
    
    Args:
        adjectives: 描述詞列表
        entry: 對話代理
        logger: 日誌記錄器
    
    Returns:
        評分 (1-5) 或 None
    """
    # 直接檢查關鍵字
    for adj in adjectives:
        if adj in [word for words in SCORE_KEYWORDS.values() for word in words]:
            return next(score for score, words in SCORE_KEYWORDS.items() if adj in words)
    
    # 如果沒有直接匹配，使用相似度 agent
    if adjectives:
        retry_count = 0
        while retry_count < 3:
            similarity_msg = f"Find the most similar keyword from these adjectives: {adjectives} to match with these keywords: {[word for words in SCORE_KEYWORDS.values() for word in words]}"
            similarity_chat = entry.initiate_chat(
                SIMILARITY_AGENT, message=similarity_msg,
                summary_method="last_msg",
                max_turns=1,
            )
            try:
                similarity_result = ast.literal_eval(similarity_chat.summary)
                logger.debug(f"從 adjectives: {adjectives} 獲得相似關鍵字: {similarity_result}")
                if "most_similar_keyword" in similarity_result:
                    similar_keyword = similarity_result["most_similar_keyword"]
                    return next(score for score, words in SCORE_KEYWORDS.items() if similar_keyword in words)
                else:
                    logger.error(f"相似度結果中沒有找到 most_similar_keyword: {similarity_chat.summary}")
            except:
                logger.error(f"無法解析相似度結果 (嘗試 {retry_count + 1}/3): {similarity_chat.summary}")
            
            retry_count += 1
            if retry_count < 3:
                logger.info(f"正在重試相似度分析 (第 {retry_count + 1} 次)")
        logger.error(f"無法找到最相似的關鍵字，已達最大重試次數")
    
    return None

def run_chat_sequence(entry: ConversableAgent, sequence: list[dict]) -> str:
    ctx = {**getattr(entry, "_initiate_chats_ctx", {})}
    for step in sequence:
        logger.debug(f"{'='*50}")
        logger.debug(f"正在與 {step['recipient'].name} 進行對話...")
        
        if step["recipient"] is REVIEW_EXTRACTOR and "reviews_dict" in ctx:
            # 處理每條評論
            all_scores = []
            restaurant_name = next(iter(ctx["reviews_dict"]))
            reviews = ctx["reviews_dict"][restaurant_name]
            
            for review in reviews:
                logger.debug(f"分析評論: {review}")
                msg = f"Extract all descriptive elements from this review: {review}"
                retry_count = 0
                adjectives = None
                
                while retry_count < 3 and adjectives is None:
                    chat = entry.initiate_chat(
                        step["recipient"], message=msg,
                        summary_method=step.get("summary_method", "last_msg"),
                        max_turns=step.get("max_turns", 1),
                    )
                    try:
                        adjectives = ast.literal_eval(chat.summary)
                        logger.debug(f"獲得描述詞: {adjectives}")
                        if not isinstance(adjectives, dict) or "food_descriptions" not in adjectives or "service_descriptions" not in adjectives:
                            raise ValueError("Invalid response format")
                        
                        # 檢查每個描述詞是否在原始評論中有相似匹配
                        review_text = review.lower()
                        all_adjectives = adjectives["food_descriptions"] + adjectives["service_descriptions"]
                        
                        for adj in all_adjectives:
                            # 將描述詞轉為小寫
                            adj_lower = adj.lower()
                            
                            # 檢查描述詞是否直接出現在評論中
                            if adj_lower in review_text:
                                continue
                            
                            # 如果描述詞包含多個字，檢查是否有足夠相似的部分
                            adj_words = adj_lower.split()
                            if len(adj_words) > 1:
                                # 計算描述詞中每個字在評論中的相似度
                                word_similarities = []
                                for word in adj_words:
                                    # 在評論中尋找最相似的單字
                                    max_similarity = max(
                                        (string_similarity(word, review_word) for review_word in review_text.split()),
                                        default=0
                                    )
                                    word_similarities.append(max_similarity)
                                
                                # 如果所有字的平均相似度超過 80%，則認為匹配成功
                                if sum(word_similarities) / len(word_similarities) >= 0.8:
                                    continue
                            
                            # 如果單字描述詞，檢查是否有任何單字與描述詞的相似度超過 80%
                            if len(adj_words) == 1:
                                has_match = any(string_similarity(adj_lower, word) >= 0.8 for word in review_text.split())
                                if has_match:
                                    continue
                            
                            # 如果以上檢查都失敗，記錄錯誤並重試
                            logger.error(f"描述詞 '{adj}' 在評論中找不到相似匹配")
                            if retry_count < 2:
                                adjectives = None
                            raise ValueError("Adjective not found in review")
                        
                        logger.debug("所有描述詞都在評論中找到相似匹配")
                    except:
                        retry_count += 1
                        if retry_count < 3:
                            logger.error(f"無法解析描述詞 (嘗試 {retry_count}/3): {chat.summary}")
                            # msg = f"Extract all descriptive elements from this review again: {review}"
                        else:
                            logger.error(f"無法解析描述詞，已達最大重試次數")
                            continue
                
                if adjectives is not None:
                    # 處理食物和服務描述詞
                    food_score = get_score_from_adjectives(adjectives["food_descriptions"], entry, logger)
                    service_score = get_score_from_adjectives(adjectives["service_descriptions"], entry, logger)
                    
                    if food_score is not None and service_score is not None:
                        all_scores.append({"food_score": food_score, "service_score": service_score})
                        logger.debug(f"分數- food: {food_score}, service: {service_score} 已添加")
            
            # 整理所有分數並直接計算總分
            food_scores = [score["food_score"] for score in all_scores]
            service_scores = [score["service_score"] for score in all_scores]
            final_score = calculate_overall_score(restaurant_name, food_scores, service_scores)
            ctx["final_score"] = str(final_score)
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
    return ctx.get("final_score", out)

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
    
    agents = {"data_fetch": DATA_FETCH, "extractor": REVIEW_EXTRACTOR, "similarity": SIMILARITY_AGENT}
    chat_sequence = [
        {"recipient": agents["data_fetch"], 
         "message": "Find reviews for this query: {user_query}", 
         "summary_method": "last_msg", 
         "max_turns": 2},

        {"recipient": agents["extractor"], 
         "message": "Analyze reviews", 
         "summary_method": "last_msg", 
         "max_turns": 1},
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
