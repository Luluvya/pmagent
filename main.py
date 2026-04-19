from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, httpx, io, base64
from datetime import datetime
from typing import TypedDict, Annotated
from pypdf import PdfReader

import chromadb
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

app = FastAPI(title="PMAgent API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SILICONFLOW_KEY = os.getenv("SILICONFLOW_KEY", "")
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
LLM_MODEL = "deepseek-ai/DeepSeek-V3"
VL_MODEL = "Qwen/Qwen2-VL-72B-Instruct"
EMBED_MODEL = "BAAI/bge-m3"

# ── ChromaDB ──────────────────────────────────────────────
chroma_client = chromadb.Client()
pm_collection = chroma_client.get_or_create_collection("pm_knowledge")
kb_version = {"version": 1, "updated_at": datetime.now().isoformat()}

# ── Embedding ─────────────────────────────────────────────
async def get_embedding(text: str) -> list:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            EMBED_URL,
            headers={"Authorization": f"Bearer {SILICONFLOW_KEY}"},
            json={"model": EMBED_MODEL, "input": text[:500]}
        )
        return resp.json()["data"][0]["embedding"]

# ── LLM ───────────────────────────────────────────────────
async def llm(system: str, user: str, max_tokens: int = 2000) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            API_URL,
            headers={"Authorization": f"Bearer {SILICONFLOW_KEY}", "Content-Type": "application/json"},
            json={
                "model": LLM_MODEL,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            }
        )
        return resp.json()["choices"][0]["message"]["content"]

# ── 多模态图片识别 ─────────────────────────────────────────
async def extract_text_from_image(image_bytes: bytes, media_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            API_URL,
            headers={"Authorization": f"Bearer {SILICONFLOW_KEY}", "Content-Type": "application/json"},
            json={
                "model": VL_MODEL,
                "max_tokens": 2000,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64}"}
                        },
                        {
                            "type": "text",
                            "text": "请提取这张图片中的所有文字内容，保持原有格式和结构，不要添加任何解释。"
                        }
                    ]
                }]
            }
        )
        return resp.json()["choices"][0]["message"]["content"]

# ── RAG ───────────────────────────────────────────────────
async def retrieve(query: str, n: int = 3) -> str:
    if pm_collection.count() == 0:
        return ""
    emb = await get_embedding(query)
    results = pm_collection.query(
        query_embeddings=[emb],
        n_results=min(n, pm_collection.count())
    )
    docs = results.get("documents", [[]])[0]
    return "\n\n---\n\n".join(docs) if docs else ""

# ── Init RAG ──────────────────────────────────────────────
async def init_rag():
    data_dir = "data"
    if not os.path.exists(data_dir):
        return
    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    if not files or pm_collection.count() > 0:
        print(f"RAG ready: {pm_collection.count()} chunks")
        return
    idx = 0
    for fname in files:
        with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
            content = f.read()
        chunks = [c.strip() for c in content.split("---") if len(c.strip()) > 50]
        for chunk in chunks:
            emb = await get_embedding(chunk)
            pm_collection.add(documents=[chunk], embeddings=[emb], ids=[f"doc_{idx}"])
            idx += 1
    kb_version["updated_at"] = datetime.now().isoformat()
    print(f"RAG initialized: {pm_collection.count()} chunks")

# ── Agent State ───────────────────────────────────────────
class PMState(TypedDict):
    messages: Annotated[list, add_messages]
    mode: str
    user_input: str
    jd_text: str
    resume_text: str
    history: list
    jd_analysis_result: str
    match_result: str
    mock_result: str
    feedback_result: str
    case_result: str
    plan_result: str
    chat_result: str
    final_response: str
    has_rag: bool

# ── Supervisor ────────────────────────────────────────────
async def supervisor_node(state: PMState) -> PMState:
    history_str = "\n".join([
        f"{'用户' if m['role']=='user' else '助手'}：{m['content'][:80]}"
        for m in state.get("history", [])[-4:]
    ])
    has_jd = bool(state.get("jd_text", "").strip())
    has_resume = bool(state.get("resume_text", "").strip())
    if not has_jd and len(state.get("user_input", "")) > 80:
        has_jd = True

    mode = await llm(
        f"""你是AI产品经理面试备考助手的任务调度器。
用户是否提供了JD：{'是' if has_jd else '否'}
用户是否提供了简历：{'是' if has_resume else '否'}

判断用户意图，只返回以下之一，不要其他文字：
analyze  - 分析JD，提炼岗位核心能力要求（用户直接发来JD内容，或要求分析JD）
match    - 简历与JD匹配度分析
mock     - 模拟面试，出题考察
feedback - 对用户的回答进行评分和改进建议
case     - 产品案例拆解或产品设计题解析
plan     - 制定个性化备考计划
chat     - 打招呼、闲聊、备考咨询等其他问题""",
        f"历史：{history_str}\n用户：{state['user_input']}"
    )
    mode = mode.strip().lower()
    if mode not in ["analyze", "match", "mock", "feedback", "case", "plan"]:
        mode = "chat"
    return {**state, "mode": mode}

# ── JD Analyzer ───────────────────────────────────────────
async def analyze_node(state: PMState) -> PMState:
    jd = state.get("jd_text", "").strip() or state.get("user_input", "")
    if not jd:
        return {**state, "jd_analysis_result": "请提供JD内容。", "has_rag": False}
    context = await retrieve(jd[:300] + " 技术PM岗位能力要求 面试考察 各大厂特点")
    result = await llm(
        f"""你是资深互联网产品经理，擅长解读大厂JD。
{'参考知识库：\n' + context if context else ''}

请对以下JD进行深度分析，输出格式如下：

## 岗位核心能力要求
列出3-5个最关键的能力维度，每条说明为什么重要

## 隐藏要求解读
从字里行间挖掘JD没有明说但实际考察的能力和态度

## 面试重点预测
基于JD推测面试官最可能考察的3-5类问题，每类给出示例问题

## 该公司/岗位特点
结合公司背景，说明这个岗位的特殊性和差异化要求

## 备考行动建议
给出3-5条具体可执行的备考建议

用专业、实用的中文回答。""",
        f"JD内容：\n{jd}"
    )
    return {**state, "jd_analysis_result": result, "has_rag": bool(context)}

# ── Resume Matcher ────────────────────────────────────────
async def match_node(state: PMState) -> PMState:
    jd = state.get("jd_text", "")
    resume = state.get("resume_text", "")
    if not jd or not resume:
        return {**state, "match_result": "请提供JD和简历内容后再进行匹配分析。", "has_rag": False}
    context = await retrieve("简历匹配 岗位能力 面试优势 简历优化建议")
    result = await llm(
        f"""你是专业的求职顾问，擅长简历与岗位匹配分析。
{'参考知识库：\n' + context if context else ''}

请对简历与JD进行深度匹配分析，输出格式如下：

## 综合匹配度
评分：X/10分
核心判断：一句话说明整体匹配情况

## 优势亮点（强匹配项）
列出简历中与JD高度契合的3-5个点，并说明面试中如何突出展示

## 待补强项（弱匹配项）
列出JD要求但简历较弱的2-3个方向，给出具体补救建议

## 简历优化建议
针对该JD，给出3条具体的简历修改建议（哪里改、怎么改、改成什么样）

## 面试策略建议
基于匹配情况，给出面试准备的重点方向和话术建议

用专业、直接的中文回答，不要客套。""",
        f"JD内容：\n{jd}\n\n简历内容：\n{resume}"
    )
    return {**state, "match_result": result, "has_rag": bool(context)}

# ── Mock Interview ────────────────────────────────────────
async def mock_node(state: PMState) -> PMState:
    jd = state.get("jd_text", "")
    resume = state.get("resume_text", "")
    history_str = "\n".join([
        f"{'用户' if m['role']=='user' else '助手'}：{m['content'][:200]}"
        for m in state.get("history", [])[-8:]
    ])
    context = await retrieve(state["user_input"] + " 面试题 考察维度 追问方式")
    result = await llm(
        f"""你是严格但友好的互联网大厂产品经理面试官（字节/阿里/腾讯/美团/京东风格）。
{'候选人简历：\n' + resume[:800] if resume else ''}
{'参考题库：\n' + context if context else ''}

面试规则：
1. 每次只问一个问题，等候选人回答后再追问或换题
2. 问题覆盖：自我介绍、产品设计、数据分析、跨团队协作、行为题
3. 根据候选人回答深度决定是追问还是换新话题
4. 对话历史为空时，先请候选人自我介绍
5. 只输出面试官的问题，语气自然，像真实面试

不要解释，不要给答案提示，直接问问题。""",
        f"JD：{jd[:500] if jd else '互联网AI产品经理通用岗位'}\n\n对话历史：\n{history_str}\n\n候选人最新回答：{state['user_input']}"
    )
    return {**state, "mock_result": result, "has_rag": bool(context)}

# ── Feedback Agent ────────────────────────────────────────
async def feedback_node(state: PMState) -> PMState:
    history_str = "\n".join([
        f"{'用户' if m['role']=='user' else '助手'}：{m['content'][:300]}"
        for m in state.get("history", [])[-6:]
    ])
    context = await retrieve(state["user_input"] + " STAR法则 回答框架 面试技巧 评分标准")
    result = await llm(
        f"""你是资深互联网PM面试教练，擅长精准点评候选人回答。
{'参考知识库：\n' + context if context else ''}

请对候选人的回答进行全面点评，输出格式：

## 回答评分
结构清晰度：X/10 · 内容深度：X/10 · 亮点展示：X/10

## 做得好的地方
具体指出1-2个亮点

## 需要改进的地方
具体指出2-3个问题，说明为什么这样回答不够好

## STAR法则重构建议
用STAR框架给出更好的回答结构（给方向，不要代写完整答案）

## 面试官视角
这个回答在面试官眼里的印象，以及可能的追问方向

专业严格，帮候选人真正提升，不要客套鼓励。""",
        f"对话历史：\n{history_str}\n\n候选人最新回答：{state['user_input']}"
    )
    return {**state, "feedback_result": result, "has_rag": bool(context)}

# ── Case Analysis Agent ───────────────────────────────────
async def case_node(state: PMState) -> PMState:
    context = await retrieve(state["user_input"] + " 产品案例 竞品分析 产品设计框架 拆解方法")
    result = await llm(
        f"""你是资深互联网产品经理，擅长产品案例分析和设计题解析。
{'参考知识库：\n' + context if context else ''}

请对用户提出的产品案例或设计题进行深度拆解，输出格式：

## 问题理解
明确题目考察的核心能力是什么

## 分析框架
选择最合适的分析框架，说明为什么用这个框架

## 深度分析
按框架逐步拆解，每个维度给出具体观点

## 核心结论
提炼2-3个最重要的洞察或建议

## 面试答题建议
如果这是面试题，给出作答思路和注意事项

用产品经理的思维方式回答，有深度、有结构、有观点。""",
        f"用户问题：{state['user_input']}\n\n补充背景（JD）：{state.get('jd_text','')[:300]}"
    )
    return {**state, "case_result": result, "has_rag": bool(context)}

# ── Plan Agent ────────────────────────────────────────────
async def plan_node(state: PMState) -> PMState:
    jd = state.get("jd_text", "")
    resume = state.get("resume_text", "")
    context = await retrieve("备考计划 面试准备 学习路径 产品经理备考建议")
    result = await llm(
        f"""你是专业的求职备考规划师，擅长为候选人制定个性化备考计划。
{'参考知识库：\n' + context if context else ''}

请根据用户情况制定个性化备考计划，输出格式：

## 当前情况评估
基于简历和JD，分析候选人的优势和主要差距

## 备考重点方向
列出3-5个最需要加强的方向，说明优先级

## 每日备考计划
{'给出1周速成计划（每天具体任务，上午/下午/晚上分配）' if not resume else '给出2周深度备考计划（分阶段，每天具体任务）'}

## 推荐练习题
针对目标岗位，列出5道最值得优先练习的面试题

## 避坑提示
3条备考过程中最容易踩的坑

计划要具体可执行，不要泛泛而谈。""",
        f"目标岗位JD：{jd[:500] if jd else '互联网AI产品经理通用岗位'}\n\n候选人简历：{resume[:500] if resume else '未提供'}\n\n用户需求：{state['user_input']}"
    )
    return {**state, "plan_result": result, "has_rag": bool(context)}

# ── Chat ──────────────────────────────────────────────────
async def chat_node(state: PMState) -> PMState:
    history_str = "\n".join([
        f"{'用户' if m['role']=='user' else '助手'}：{m['content'][:100]}"
        for m in state.get("history", [])[-6:]
    ])
    context = await retrieve(state["user_input"])
    result = await llm(
        f"""你是PMAgent，专注互联网AI产品经理求职备考的助手。
{'参考知识库：\n' + context if context else ''}
能帮用户：分析JD、匹配简历、模拟面试、点评回答、拆解产品案例、制定备考计划。
友好专业，适当用emoji，引导用户上传JD或简历开始备考。不超过200字。""",
        f"历史：\n{history_str}\n用户：{state['user_input']}"
    )
    return {**state, "chat_result": result, "has_rag": bool(context)}

# ── Synthesis ─────────────────────────────────────────────
async def synthesis_node(state: PMState) -> PMState:
    response = (
        state.get("jd_analysis_result") or
        state.get("match_result") or
        state.get("mock_result") or
        state.get("feedback_result") or
        state.get("case_result") or
        state.get("plan_result") or
        state.get("chat_result") or
        "抱歉，我暂时无法回答这个问题。"
    )
    return {**state, "final_response": response}

# ── Build Graph ───────────────────────────────────────────
def build_graph():
    g = StateGraph(PMState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("analyze",   analyze_node)
    g.add_node("match",     match_node)
    g.add_node("mock",      mock_node)
    g.add_node("feedback",  feedback_node)
    g.add_node("case",      case_node)
    g.add_node("plan",      plan_node)
    g.add_node("chat",      chat_node)
    g.add_node("synthesis", synthesis_node)
    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor",
        lambda s: s.get("mode", "chat"),
        {"analyze":"analyze","match":"match","mock":"mock",
         "feedback":"feedback","case":"case","plan":"plan","chat":"chat"}
    )
    for node in ["analyze","match","mock","feedback","case","plan","chat"]:
        g.add_edge(node, "synthesis")
    g.add_edge("synthesis", END)
    return g.compile()

graph = build_graph()

MODE_LABELS = {
    "analyze":  "🔍 JD分析",
    "match":    "📊 简历匹配",
    "mock":     "🎤 模拟面试",
    "feedback": "📝 回答点评",
    "case":     "🧩 案例拆解",
    "plan":     "📅 备考计划",
    "chat":     "🤖 助手",
}

# ── Startup ───────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    await init_rag()

# ── Endpoints ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    jd_text: str = ""
    resume_text: str = ""
    history: list = []

@app.post("/chat")
async def chat(req: ChatRequest):
    state: PMState = {
        "messages": [HumanMessage(content=req.message)],
        "mode": "",
        "user_input": req.message,
        "jd_text": req.jd_text,
        "resume_text": req.resume_text,
        "history": req.history,
        "jd_analysis_result": "",
        "match_result": "",
        "mock_result": "",
        "feedback_result": "",
        "case_result": "",
        "plan_result": "",
        "chat_result": "",
        "final_response": "",
        "has_rag": False,
    }
    result = await graph.ainvoke(state)
    mode = result.get("mode", "chat")
    return {
        "response": result.get("final_response", ""),
        "mode": mode,
        "agent": MODE_LABELS.get(mode, "🤖"),
        "has_rag": result.get("has_rag", False),
    }

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), type: str = Form("resume")):
    content = await file.read()
    reader = PdfReader(io.BytesIO(content))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return {"text": text.strip(), "type": type, "pages": len(reader.pages)}

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...), type: str = Form("resume")):
    content = await file.read()
    media_type = file.content_type or "image/jpeg"
    text = await extract_text_from_image(content, media_type)
    return {"text": text.strip(), "type": type}

@app.post("/upload-knowledge")
async def upload_knowledge(file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8")
    chunks = [c.strip() for c in content.split("---") if len(c.strip()) > 50]
    count = pm_collection.count()
    for i, chunk in enumerate(chunks):
        emb = await get_embedding(chunk)
        pm_collection.add(documents=[chunk], embeddings=[emb], ids=[f"doc_{count+i}"])
    kb_version["version"] += 1
    kb_version["updated_at"] = datetime.now().isoformat()
    return {"added": len(chunks), "total": pm_collection.count()}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "knowledge_chunks": pm_collection.count(),
        "kb_version": kb_version["version"],
        "model": LLM_MODEL,
        "vl_model": VL_MODEL,
    }
