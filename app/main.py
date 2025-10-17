import os, json, time, asyncio
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
# ---------- 配置 ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "*")
MODEL_PLANNER  = "gemini-2.5-flash"
MODEL_WORKER   = "gemini-2.5-flash"

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

client = genai.Client(api_key=GEMINI_API_KEY)
GOOGLE_SEARCH_TOOL = types.Tool(google_search=types.GoogleSearch())

# ---------- 工具 ----------
def safe_json(txt: str) -> Dict[str, Any]:
    """尽力把模型文本解析为 dict；若为 list 则包到 items 字段。"""
    try:
        data = json.loads(txt)
        return data if isinstance(data, dict) else {"items": data}
    except Exception:
        try:
            s, e = txt.find("{"), txt.rfind("}")
            if s != -1 and e != -1 and e > s:
                data = json.loads(txt[s:e+1])
                return data if isinstance(data, dict) else {"items": data}
        except Exception:
            pass
    return {"_raw": txt, "_error": "json_parse_failed"}

def to_str(x):
    return x if isinstance(x, str) or x is None else str(x)

def as_dict(x) -> Dict[str, Any]:
    """不是 dict 就兜底成 dict，避免 .get 抛错；把原值放在 _raw 里以便排查。"""
    return x if isinstance(x, dict) else {"_raw": x}

def as_list(x) -> List[Any]:
    """不是 list 就兜底成空 list。"""
    return x if isinstance(x, list) else []

def to_str_list(x) -> List[str]:
    """把任意列表元素安全地转成字符串列表；支持 dict 提取常见字段。"""
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for item in x:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            for k in ("name", "title", "person", "value"):
                v = item.get(k)
                if isinstance(v, str):
                    out.append(v)
                    break
            else:
                try:
                    out.append(json.dumps(item, ensure_ascii=False))
                except Exception:
                    out.append(str(item))
        else:
            out.append(str(item))
    return out

async def gen_async(*, model: str, contents, config: types.GenerateContentConfig):
    """把阻塞的 generate_content 放线程，不阻塞事件循环。"""
    return await asyncio.to_thread(client.models.generate_content, model=model, contents=contents, config=config)

# ---------- Schemas（含 search_queries / sources / step_conclusion） ----------
SOURCE_ITEM = types.Schema(
    type="OBJECT",
    properties={
        "site": types.Schema(type="STRING"),
        "url": types.Schema(type="STRING"),
        "note": types.Schema(type="STRING"),
    },
    required=["site","url"]
)

RESOLVER_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "search_queries": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
        "sources": types.Schema(type="ARRAY", items=SOURCE_ITEM),
        "candidates": types.Schema(type="ARRAY", items=types.Schema(
            type="OBJECT",
            properties={
                "title": types.Schema(type="STRING"),
                "year":  types.Schema(type="STRING"),
                "confidence": types.Schema(type="NUMBER"),
                "urls": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
            },
            required=["title","confidence"]
        )),
        "chosen": types.Schema(type="OBJECT", properties={
            "title": types.Schema(type="STRING"),
            "year":  types.Schema(type="STRING"),
            "confidence": types.Schema(type="NUMBER"),
            "reason": types.Schema(type="STRING"),
        }, required=["title","confidence"]),
        "step_conclusion": types.Schema(type="STRING"),
    }, required=["chosen"]
)

RATING_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "search_queries": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
        "sources": types.Schema(type="ARRAY", items=SOURCE_ITEM),
        "sites": types.Schema(type="ARRAY", items=types.Schema(
            type="OBJECT",
            properties={
                "site": types.Schema(type="STRING"),
                "rating": types.Schema(type="NUMBER"),
                "raw": types.Schema(type="STRING"),
                "votes": types.Schema(type="INTEGER"),
                "url": types.Schema(type="STRING"),
            }, required=["site","rating","url"]
        )),
        "step_conclusion": types.Schema(type="STRING"),
    },
    required=["sites"]
)

TEAM_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "search_queries": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
        "sources": types.Schema(type="ARRAY", items=SOURCE_ITEM),
        "team": types.Schema(type="OBJECT", properties={
            "director": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
            "writer": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
            "cast": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
            "prod_companies": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
        }),
        "verdict": types.Schema(type="STRING"),
        "step_conclusion": types.Schema(type="STRING"),
    },
    required=["verdict"]
)

REVIEW_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "search_queries": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
        "sources": types.Schema(type="ARRAY", items=SOURCE_ITEM),
        "sources_detail": types.Schema(type="ARRAY", items=types.Schema(
            type="OBJECT",
            properties={
                "site": types.Schema(type="STRING"),
                "url":  types.Schema(type="STRING"),
                "pos_ratio": types.Schema(type="NUMBER"),
                "neg_ratio": types.Schema(type="NUMBER"),
                "top_pros": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                "top_cons": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
            }, required=["site","url"]
        )),
        "overall_sentiment": types.Schema(type="STRING"),
        "step_conclusion": types.Schema(type="STRING"),
    },
    required=["overall_sentiment"]
)

def worker_config(schema: Optional[types.Schema] = None) -> types.GenerateContentConfig:
    if schema is None:
        return types.GenerateContentConfig(tools=[GOOGLE_SEARCH_TOOL])
    return types.GenerateContentConfig(tools=[GOOGLE_SEARCH_TOOL], response_schema=schema)

# ---------- 子 Agent（阻塞实现，在线程里跑） ----------
def run_resolver_blocking(user_query: str) -> Dict[str, Any]:
    prompt = f"""
用户输入：{user_query}
任务：解析为具体影片。请严格输出 JSON，字段包括：
- search_queries: 本步用到的检索关键词数组
- sources: [{'{'}site,url,note{'}'}...] 关键参考站点及其简要说明
- candidates: 若干候选（title/year/confidence/urls）
- chosen: 最终选择（title/year/confidence/reason）
- step_conclusion: 用一两句话总结你为何选择该影片
使用 Google Search 工具验证。
"""
    resp = client.models.generate_content(model=MODEL_WORKER, contents=prompt, config=worker_config(RESOLVER_SCHEMA))
    return safe_json(resp.text)

def run_rating_blocking(title: str, year: Optional[str]) -> Dict[str, Any]:
    prompt = f"""
目标：查询《{title}》{f"（{year}）" if year else ""}在 IMDb/TMDb/RottenTomatoes/Metacritic/豆瓣 的评分并统一到 0-10。
严格输出 JSON，字段：
- search_queries: 使用过的检索关键词数组
- sources: 关键参考站点列表（site,url,note）。可与 sites 重复，但 note 需说明“该源为何有效”
- sites: 评分明细（site/rating/raw/votes/url）
- step_conclusion: 一句话概括评分形势（如“普遍高分/口碑分化/褒贬不一”）
使用 Google Search。
"""
    resp = client.models.generate_content(model=MODEL_WORKER, contents=prompt, config=worker_config(RATING_SCHEMA))
    return safe_json(resp.text)

def run_team_blocking(title: str, year: Optional[str]) -> Dict[str, Any]:
    prompt = f"""
目标：分析《{title}》{f"（{year}）" if year else ""}导演/编剧/主演/制作公司履历与风险。
严格输出 JSON，字段：
- search_queries: 使用过的检索关键词数组
- sources: 关键参考站点（site,url,note）
- team: director/writer/cast/prod_companies（请输出为字符串数组，不要对象）
- verdict: 总体判断（正/中/负 or 简洁语句）
- step_conclusion: 一句话概括阵容亮点与潜在风险
使用 Google Search。
"""
    resp = client.models.generate_content(model=MODEL_WORKER, contents=prompt, config=worker_config(TEAM_SCHEMA))
    return safe_json(resp.text)

def run_review_blocking(title: str, year: Optional[str]) -> Dict[str, Any]:
    prompt = f"""
目标：汇总《{title}》{f"（{year}）" if year else ""}各大评论站与媒体舆情，估算正/负面占比与高频优缺点。
严格输出 JSON，字段：
- search_queries: 使用过的检索关键词数组
- sources: 关键参考站点（site,url,note）
- sources_detail: [{'{'}site,url,pos_ratio,neg_ratio,top_pros,top_cons{'}'}...]
- overall_sentiment: 总体情绪（正/中/负 或 简短描述）
- step_conclusion: 用一两句话概括舆情结论
使用 Google Search。
"""
    resp = client.models.generate_content(model=MODEL_WORKER, contents=prompt, config=worker_config(REVIEW_SCHEMA))
    return safe_json(resp.text)

# ---------- 规划：父Agent给出 agents 列表 ----------
PLAN_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "objective": types.Schema(type="STRING"),
        "agents": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
        "notes": types.Schema(type="STRING"),
    },
    required=["agents"]
)

async def plan_agents(user_query: str) -> List[str]:
    prompt = f"""
用户输入：{user_query}
只输出 JSON：
- agents: 需要调用的 agent 名称数组，候选有 resolver, rating, team, review
- objective: 简述目标
- notes: （可选）简短理由
通常先 resolver，再并行 rating/team/review。
"""
    resp = await gen_async(
        model=MODEL_PLANNER,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=PLAN_SCHEMA),
    )
    data = safe_json(resp.text)
    agents = [a for a in as_list(as_dict(data).get("agents")) if a in {"resolver","rating","team","review"}]
    if "resolver" not in agents:
        agents = ["resolver"] + [a for a in agents if a != "resolver"]
    if agents == ["resolver"]:
        agents.extend(["rating","team","review"])
    return agents

# ---------- 叙述器 ----------
def narrate(agent: str, payload: Dict[str, Any]) -> str:
    payload_d = as_dict(payload)

    if agent == "resolver":
        ch = as_dict(payload_d.get("chosen"))
        year = ch.get("year")
        year_part = f" ({year})" if year else ""
        return f"我确定目标影片是《{ch.get('title','?')}》{year_part}，置信度 {ch.get('confidence','?')}。"

    if agent == "rating":
        sites = as_list(payload_d.get("sites"))
        if not sites:
            return "未获取到评分来源。"
        parts = []
        for s in sites[:6]:
            if isinstance(s, dict):
                parts.append(f"{s.get('site')}:{s.get('rating')}/10")
            else:
                parts.append(str(s))
        return "评分来源汇总：" + "；".join(parts) + "。"

    if agent == "team":
        team = as_dict(payload_d.get("team"))
        dirs = "、".join(to_str_list(team.get("director")))
        cast = "、".join(to_str_list(team.get("cast")))
        verdict = payload_d.get("verdict","")
        return f"主创：导演 {dirs or '—'}；主演 {cast or '—'}。结论：{verdict}。"

    if agent == "review":
        srcs = as_list(payload_d.get("sources_detail"))
        pos = [s.get("pos_ratio", 0.0) for s in srcs if isinstance(s, dict) and isinstance(s.get("pos_ratio"), (int,float))]
        neg = [s.get("neg_ratio", 0.0) for s in srcs if isinstance(s, dict) and isinstance(s.get("neg_ratio"), (int,float))]
        pos_avg = int(100*sum(pos)/max(1,len(pos))) if pos else 0
        neg_avg = int(100*sum(neg)/max(1,len(neg))) if neg else 0
        return f"评论抽样显示：正面约 {pos_avg}% ，负面约 {neg_avg}% 。"

    return "完成。"

# ---------- WS：全量流式 + 强制并行 + agent_step ----------
@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    await ws.accept()

    async def send(obj: Dict[str, Any]):
        await ws.send_json(obj)

    async def stream_text(event_type: str, text: str, delay: float = 0.0):
        for ch in text:
            await send({"type": event_type, "data": ch})
            if delay > 0:
                await asyncio.sleep(delay)

    async def stream_agent_text(agent: str, text: str, delay: float = 0.0):
        for ch in text:
            await send({"type": "agent_delta", "data": {"agent": agent, "text": ch}})
            if delay > 0:
                await asyncio.sleep(delay)

    try:
        init = await ws.receive_json()
        user_query = init.get("user_query") or ""
        if not user_query:
            await send({"type":"error", "data":"missing user_query"})
            await ws.close(); return

        # 0) 规划
        await send({"type":"plan_start", "data":"规划启动…"})
        agents_list = await plan_agents(user_query)
        await send({"type":"plan_delta", "data": json.dumps({"agents": agents_list}, ensure_ascii=False)})
        await send({"type":"plan_end", "data":"规划完成"})

        # 1) resolver
        await send({"type":"agent_start", "data": {"agent": "resolver", "args": {"user_query": user_query}}})
        await send({"type":"agent_delta", "data": {"agent":"resolver","text":"正在解析影片实体…","stage":"calling","percent":40}})

        resolver_payload = await asyncio.to_thread(run_resolver_blocking, user_query)
        resolver_payload_d = as_dict(resolver_payload)

        await send({"type":"agent_step", "data": {
            "agent": "resolver",
            "search_queries": as_list(resolver_payload_d.get("search_queries")),
            "sources": as_list(resolver_payload_d.get("sources")),
            "conclusion": resolver_payload_d.get("step_conclusion") or narrate("resolver", resolver_payload_d),
        }})
        await send({"type":"agent_delta", "data": {"agent":"resolver","text": narrate("resolver", resolver_payload_d),"stage":"parsed","percent":90}})
        await send({"type":"agent_end", "data": {"agent":"resolver","ok": True, "payload": resolver_payload}})

        chosen = as_dict(resolver_payload_d.get("chosen"))
        title, year = chosen.get("title"), to_str(chosen.get("year"))

        # 2) 并行跑其余 agent
        want = set(agents_list); want.update(["rating","team","review"]); want.discard("resolver")

        async def run_agent(agent: str):
            await send({"type":"agent_start", "data": {"agent": agent, "args": {"title": title, "year": year}}})
            await send({"type":"agent_delta", "data": {"agent": agent, "text": "准备检索与分析…", "stage":"queued", "percent":10}})
            await send({"type":"agent_delta", "data": {"agent": agent, "text": "正在调用 Gemini 子模型 + Google Search…", "stage":"calling", "percent":40}})

            def _call():
                if agent == "rating": return run_rating_blocking(title, year)
                if agent == "team":   return run_team_blocking(title, year)
                if agent == "review": return run_review_blocking(title, year)
                return {"error": f"unknown agent {agent}"}

            payload = await asyncio.to_thread(_call)
            payload_d = as_dict(payload)

            await send({"type":"agent_step", "data": {
                "agent": agent,
                "search_queries": as_list(payload_d.get("search_queries")),
                "sources": as_list(payload_d.get("sources")),
                "conclusion": payload_d.get("step_conclusion") or narrate(agent, payload_d),
            }})

            await send({"type":"agent_delta", "data": {"agent": agent, "text": "信息汇总中…", "stage":"aggregate", "percent":80}})
            await stream_agent_text(agent, narrate(agent, payload_d), delay=0.02)
            await send({"type":"agent_end", "data": {"agent": agent, "ok": True, "payload": payload}})
            return (agent, payload)

        tasks = [asyncio.create_task(run_agent(a)) for a in sorted(want)]
        results: Dict[str, Any] = {"resolver": resolver_payload}
        for t in asyncio.as_completed(tasks):
            agent, payload = await t
            results[agent] = payload

        # 3) 最终报告
        final_prompt = f"""
你是父Agent。下面是本次多Agent的结构化结果(JSON)：
{json.dumps(results, ensure_ascii=False, default=str)}

请用中文写一个结论性报告，包含：
- 影片标题/年份
- 评分概览与比较
- 阵容亮点与潜在风险
- 评论/舆情的整体倾向与高频优缺点
- 是否值得看 & 适合人群
用段落与小标题清晰呈现，语言简练。
"""
        await send({"type":"final_start", "data":"开始生成最终报告…"})
        final_resp = await gen_async(model=MODEL_PLANNER, contents=final_prompt, config=types.GenerateContentConfig())
        final_text = final_resp.text or "(无输出)"
        await stream_text("final_delta", final_text, delay=0.02)  # 想打字机就设 0.01
        await send({"type":"final_end", "data":"报告完成"})
        await ws.close(); return

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_json({"type":"error","data":str(e)})
        finally:
            await ws.close()
