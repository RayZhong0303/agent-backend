import os, json, asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
# ---------- 基础配置 ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY")
MODEL_PLANNER  = "gemini-2.5-flash"
MODEL_WORKER   = "gemini-2.5-flash"

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

client = genai.Client(api_key=GEMINI_API_KEY)
GOOGLE_SEARCH_TOOL = types.Tool(google_search=types.GoogleSearch())

# ---------- 工具函数 ----------
def human_line(agent: str, action: str, why: str = "", hint: str = "") -> str:
        """
        生成更口语化的一行叙述；避免机械术语。
        action: 正在做什么
        why: 为什么（可空）
        hint: 预期产出/下一步（可空）
        """
        parts = [f"[{agent}] {action}"]
        if why: parts.append(f"（原因：{why}）")
        if hint: parts.append(f"→ {hint}")
        return " ".join(parts)
def safe_json(txt: str) -> Dict[str, Any]:
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
    return x if isinstance(x, dict) else {"_raw": x}

def as_list(x) -> List[Any]:
    return x if isinstance(x, list) else []

def to_str_list(x) -> List[str]:
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
    return await asyncio.to_thread(client.models.generate_content, model=model, contents=contents, config=config)

# ---------- Schemas ----------
SOURCE_ITEM = types.Schema(
    type="OBJECT",
    properties={"site": types.Schema(type="STRING"), "url": types.Schema(type="STRING"), "note": types.Schema(type="STRING")},
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
            properties={"site": types.Schema(type="STRING"), "rating": types.Schema(type="NUMBER"),
                        "raw": types.Schema(type="STRING"), "votes": types.Schema(type="INTEGER"), "url": types.Schema(type="STRING")},
            required=["site","rating","url"]
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
                "site": types.Schema(type="STRING"), "url":  types.Schema(type="STRING"),
                "pos_ratio": types.Schema(type="NUMBER"), "neg_ratio": types.Schema(type="NUMBER"),
                "top_pros": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                "top_cons": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
            }, required=["site","url"]
        )),
        "overall_sentiment": types.Schema(type="STRING"),
        "step_conclusion": types.Schema(type="STRING"),
    },
    required=["overall_sentiment"]
)

PLAN_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "objective": types.Schema(type="STRING"),
        "agents": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
        "notes": types.Schema(type="STRING"),
    }, required=["agents"]
)

# 跟进多轮的规划 Schema
FOLLOWUP_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "change_title": types.Schema(type="BOOLEAN"),
        "new_title": types.Schema(type="STRING"),
        "new_year": types.Schema(type="STRING"),
        "agents": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),  # 需要重跑的子agent，如 ["rating","review"]
        "answer_only": types.Schema(type="BOOLEAN"),  # 仅基于已有结果作答，不调用子agent
        "notes": types.Schema(type="STRING"),
    }, required=[]
)

def worker_config(schema: Optional[types.Schema] = None) -> types.GenerateContentConfig:
    if schema is None:
        return types.GenerateContentConfig(tools=[GOOGLE_SEARCH_TOOL])
    return types.GenerateContentConfig(tools=[GOOGLE_SEARCH_TOOL], response_schema=schema)

# ---------- 子Agent ----------
def run_resolver_blocking(user_query: str) -> Dict[str, Any]:
    prompt = f"""
用户输入：{user_query}
任务：解析为具体影片。请严格输出 JSON，字段包括：
- search_queries
- sources: [{{site,url,note}}...]
- candidates: [{'{'}title,year,confidence,urls{'}'}...]
- chosen: {{title,year,confidence,reason}}
- step_conclusion
使用 Google Search 验证。
"""
    resp = client.models.generate_content(model=MODEL_WORKER, contents=prompt, config=worker_config(RESOLVER_SCHEMA))
    return safe_json(resp.text)

async def run_rating_blocking(title: str, year: Optional[str], agent_name: str, say_fn, step_fn) -> Dict[str, Any]:
    """
    拆成多步：plan_sites -> per-site 抓取 -> 汇总
    参数：
      - agent_name: 用于 WS 标记（通常传 "rating"）
      - say_fn: 上面的 say
      - step_fn: 上面的 step_note
    """
    # 0) 规划要查的站点 & 检索词
    await say_fn(agent_name, human_line(agent_name, "正在规划要查哪些评分站点", "覆盖主流平台", "准备生成检索关键词…"), stage="plan", percent=12)

    plan_prompt = f"""
仅输出 JSON：
- search_queries: 针对《{title}》{f"（{year}）" if year else ""}抓分的关键词，越短越好
- sites: 需要查询的平台列表，限定在 ["IMDb","TMDb","RottenTomatoes","Metacritic","豆瓣"]，按重要性排序
"""
    plan_resp = await gen_async(
        model=MODEL_WORKER,
        contents=plan_prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    plan = safe_json(plan_resp.text)
    queries = as_list(plan.get("search_queries"))
    sites_plan = as_list(plan.get("sites")) or ["IMDb","TMDb","RottenTomatoes","Metacritic","豆瓣"]

    partial: Dict[str, Any] = {
        "search_queries": queries,
        "sources": [],
        "sites": [],
        "step_conclusion": ""
    }
    await step_fn(agent_name, partial, f"准备在 { '、'.join(sites_plan) } 查询评分…")
    await say_fn(agent_name, human_line(agent_name, f"将查询：{'、'.join(sites_plan)}", "", "开始逐站点抓取"), stage="plan", percent=18)

    # 1) 逐站点抓取
    total = len(sites_plan)
    for i, site in enumerate(sites_plan, start=1):
        p = 18 + int(60 * i / max(1,total))  # 18% -> ~78%
        await say_fn(agent_name, human_line(agent_name, f"去 {site} 上找该片评分", "以影片官方条目为准", "完成后会立刻汇报"), stage=f"fetch:{site}", percent=p)

        site_schema = types.Schema(
            type="OBJECT",
            properties={
                "site": types.Schema(type="STRING"),
                "rating": types.Schema(type="NUMBER"),
                "raw": types.Schema(type="STRING"),
                "votes": types.Schema(type="INTEGER"),
                "url": types.Schema(type="STRING"),
                "source_item": SOURCE_ITEM,  # 便于直接加入 sources
                "step_conclusion": types.Schema(type="STRING"),
            },
            required=["site","url"]
        )

        site_prompt = f"""
目标：只抓取 {site} 上《{title}》{f"（{year}）" if year else ""}的评分（如有）。
要求：
- rating 统一到 0-10，不确定也返回 raw
- url 指向该片在 {site} 的详情页
- source_item 形如 {{site,url,note}}，note 简述可信度
"""
        site_resp = await gen_async(
            model=MODEL_WORKER,
            contents=site_prompt,
            config=types.GenerateContentConfig(tools=[GOOGLE_SEARCH_TOOL], response_schema=site_schema)
        )
        site_data = safe_json(site_resp.text)
        site_row = {
            "site": site_data.get("site") or site,
            "rating": site_data.get("rating"),
            "raw": site_data.get("raw"),
            "votes": site_data.get("votes"),
            "url": site_data.get("url"),
        }
        if site_row["url"]:
            partial["sites"].append(site_row)
            si = as_dict(site_data.get("source_item"))
            if si.get("url"):
                partial["sources"].append(si)

        # 立刻把该站点的结果叙述给前端
        await step_fn(agent_name, partial, f"{site} 抓取完成")
        pretty = site_row.get('rating') or site_row.get('raw') or '暂无'
        await say_fn(agent_name, human_line(agent_name, f"{site} 抓取完成", "", f"评分：{pretty}"), stage=f"fetched:{site}", percent=min(78, p))

    # 2) 汇总结论
    await say_fn(agent_name, human_line(agent_name, "正在汇总各平台评分", "", "形成本阶段结论"), stage="aggregate", percent=88)

    parts = []
    for s in partial["sites"]:
        if isinstance(s, dict):
            parts.append(f"{s.get('site')}:{s.get('rating') or s.get('raw') or '?'}")
    partial["step_conclusion"] = "；".join(parts) or "未获取到评分来源"

    await say_fn(agent_name, human_line(agent_name, "评分阶段完成", "", "进入后续分析"), stage="done", percent=95)
    return partial

async def run_team_blocking(title: str, year: Optional[str], agent_name: str, say_fn, step_fn) -> Dict[str, Any]:
    await say_fn(agent_name, human_line(agent_name, "梳理导演/编剧/主演/制作公司", "看履历与过往口碑", "随后给出总体判断"), stage="plan", percent=15)

    prompt = f"""
严格输出 JSON：
- search_queries
- sources: [{{site,url,note}}...]
- team: director/writer/cast/prod_companies（字符串数组）
- verdict
- step_conclusion
目标影片：《{title}》{f"（{year}）" if year else ""}。
"""
    resp = await gen_async(model=MODEL_WORKER, contents=prompt, config=worker_config(TEAM_SCHEMA))
    data = safe_json(resp.text)
    await step_fn(agent_name, data, "已初步整理阵容")
    await say_fn(agent_name, human_line(agent_name, "检查关键主创的代表作与奖项", "", "评估稳定输出与潜在风险"), stage="verify", percent=55)

    # 可追加：针对导演/主演逐个验证的子步
    await say_fn(agent_name, human_line(agent_name, "阵容阶段完成", "", "准备出结论"), stage="done", percent=95)
    return data

async def run_review_blocking(title: str, year: Optional[str], agent_name: str, say_fn, step_fn) -> Dict[str, Any]:
    await say_fn(agent_name, human_line(agent_name, "收集评论网站与媒体测评", "覆盖好评与差评", "提取高频优缺点"), stage="plan", percent=20)

    prompt = f"""
严格输出 JSON：
- search_queries
- sources: [{{site,url,note}}...]
- sources_detail: [{{site,url,pos_ratio,neg_ratio,top_pros,top_cons}}...]
- overall_sentiment
- step_conclusion
目标影片：《{title}》{f"（{year}）" if year else ""}。
"""
    resp = await gen_async(model=MODEL_WORKER, contents=prompt, config=worker_config(REVIEW_SCHEMA))
    data = safe_json(resp.text)
    await step_fn(agent_name, data, "已汇总多源评价")
    await say_fn(agent_name, human_line(agent_name, "统计情绪占比并归纳关键词", "", "给出总体倾向"), stage="aggregate", percent=75)

    await say_fn(agent_name, human_line(agent_name, "评论阶段完成", "", "准备汇总"), stage="done", percent=95)
    return data
# ---------- 父Agent规划 ----------
async def plan_agents(user_query: str) -> List[str]:
    prompt = f"""
用户输入：{user_query}
只输出 JSON：
- agents: 需要调用的 agent 名称数组，候选有 resolver, rating, team, review
- objective, notes（可选）
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

async def plan_followup(user_text: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    返回：
    - change_title: 是否换片
    - new_title/new_year: 如果换片
    - agents: 需要重跑的子agent
    - answer_only: 仅用已有结果作答
    """
    prompt = f"""
你是多轮对话的调度器。
给定之前的结构化结果 JSON（可能很长）与用户的追问，请判断是否需要更换影片、是否需要重跑某些子agent，或仅基于现有信息作答。
严格输出 JSON：
- change_title: boolean
- new_title: string
- new_year: string
- agents: ["resolver","rating","team","review"] 子集
- answer_only: boolean
- notes: string（可选）

[之前的结果JSON]
{json.dumps(ctx, ensure_ascii=False, default=str)}

[用户追问]
{user_text}
"""
    resp = await gen_async(
        model=MODEL_PLANNER,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=FOLLOWUP_SCHEMA),
    )
    return safe_json(resp.text)

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
        if not sites: return "未获取到评分来源。"
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

# ---------- WebSocket：多轮对话 ----------
@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    await ws.accept()
    
    async def say(agent: str, text: str, stage: Optional[str] = None, percent: Optional[int] = None):
        """实时发送一条自然语言进度。"""
        await send({"type": "agent_delta", "data": {"agent": agent, "text": text, "stage": stage, "percent": percent}})

    async def step_note(agent: str, payload: Dict[str, Any], fallback_narrate: Optional[str] = None):
        """把当前小步的结构化信息（search_queries/sources/step_conclusion）也同步给前端。"""
        pd = as_dict(payload)
        await send({"type":"agent_step", "data":{
            "agent": agent,
            "search_queries": as_list(pd.get("search_queries")),
            "sources": as_list(pd.get("sources")),
            "conclusion": pd.get("step_conclusion") or (fallback_narrate or ""),
        }})
    async def send(obj: Dict[str, Any]): await ws.send_json(obj)

    async def stream_text(event_type: str, text: str, delay: float = 0.0):
        for ch in text:
            await send({"type": event_type, "data": ch})
            if delay > 0: await asyncio.sleep(delay)

    async def stream_agent_text(agent: str, text: str, delay: float = 0.0):
        for ch in text:
            await send({"type": "agent_delta", "data": {"agent": agent, "text": ch}})
            if delay > 0: await asyncio.sleep(delay)

    # —— 会话内存态（仅存活在本次连接中）——
    session_ctx: Dict[str, Any] = {
        "results": None,           # 上一次完整结构化结果（包含 resolver/rating/team/review）
        "title": None,
        "year": None,
    }

    async def run_full_pipeline(user_query: str):
        """首轮完整运行：规划 -> resolver -> 并行子agent -> 最终报告"""
        await send({"type":"plan_start", "data":"规划启动…"})
        agents_list = await plan_agents(user_query)
        await send({"type":"plan_delta", "data": json.dumps({"agents": agents_list}, ensure_ascii=False)})
        await send({"type":"plan_end", "data":"规划完成"})

        # resolver
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

        # 并行子agent
        want = set(agents_list); want.update(["rating","team","review"]); want.discard("resolver")

        async def run_agent(agent: str):
            await send({"type":"agent_start", "data": {"agent": agent, "args": {"title": title, "year": year}}})
            await send({"type":"agent_delta", "data": {"agent": agent, "text": "准备检索与分析…", "stage":"queued", "percent":10}})
            await send({"type":"agent_delta", "data": {"agent": agent, "text": "正在调用 Gemini 子模型 + Google Search…", "stage":"calling", "percent":40}})
            async def _call_async():
                if agent == "rating": return await run_rating_blocking(title, year, "rating", say, step_note)
                if agent == "team":   return await run_team_blocking(title, year, "team", say, step_note)
                if agent == "review": return await run_review_blocking(title, year, "review", say, step_note)
                return {"error": f"unknown agent {agent}"}

            payload = await _call_async()
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

        # 保存上下文
        session_ctx["results"] = results
        session_ctx["title"]   = title
        session_ctx["year"]    = year

        # 最终报告（Markdown 可直接渲染）
        final_prompt = f"""
你是父Agent。下面是本次多Agent的结构化结果(JSON)：
{json.dumps(results, ensure_ascii=False, default=str)}

请用中文写一个结论性报告（Markdown），包含：
- 影片标题/年份
- 评分概览与比较
- 阵容亮点与潜在风险
- 评论/舆情的整体倾向与高频优缺点
- 是否值得看 & 适合人群
"""
        await send({"type":"final_start", "data":"开始生成最终报告…"})
        final_resp = await gen_async(model=MODEL_PLANNER, contents=final_prompt, config=types.GenerateContentConfig())
        final_text = final_resp.text or "(无输出)"
        await stream_text("final_delta", final_text, delay=0.02)
        await send({"type":"final_end", "data":"报告完成"})

    async def answer_followup(user_text: str):
        """基于上一轮上下文的追问：必要时重跑相关子agent，或直接作答。"""
        ctx = {
            "title": session_ctx.get("title"),
            "year": session_ctx.get("year"),
            "results": session_ctx.get("results"),
        }
        await send({"type":"chat_start", "data":{"role":"assistant","text":"正在理解你的追问并规划动作…"}})
        plan = await plan_followup(user_text, ctx)
        plan_d = as_dict(plan)
        change = bool(plan_d.get("change_title"))
        agents = [a for a in as_list(plan_d.get("agents")) if a in {"resolver","rating","team","review"}]
        answer_only = bool(plan_d.get("answer_only"))

        # 若需要换片，先用 resolver
        if change:
            await send({"type":"agent_start", "data": {"agent": "resolver", "args": {"user_query": user_text}}})
            await send({"type":"agent_delta", "data": {"agent":"resolver","text":"正在解析新的影片实体…","stage":"calling","percent":40}})
            resolver_payload = await asyncio.to_thread(run_resolver_blocking, user_text)
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
            session_ctx["title"] = chosen.get("title")
            session_ctx["year"]  = to_str(chosen.get("year"))
            # 切换后建议重跑三类子agent
            if not agents: agents = ["rating","team","review"]

        title, year = session_ctx.get("title"), session_ctx.get("year")

        # 如需重跑子agent（并行）
        sub_results: Dict[str, Any] = {}
        if agents:
            async def run_agent(agent: str):
                await send({"type":"agent_start", "data": {"agent": agent, "args": {"title": title, "year": year}}})
                await send({"type":"agent_delta", "data": {"agent": agent, "text": "准备检索与分析…", "stage":"queued", "percent":10}})
                await send({"type":"agent_delta", "data": {"agent": agent, "text": "正在调用 Gemini 子模型 + Google Search…", "stage":"calling", "percent":40}})
                async def _call_async():
                    if agent == "rating": return await run_rating_blocking(title, year, "rating", say, step_note)
                    if agent == "team":   return await run_team_blocking(title, year, "team", say, step_note)
                    if agent == "review": return await run_review_blocking(title, year, "review", say, step_note)
                    return {"error": f"unknown agent {agent}"}

                payload = await _call_async()
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
            tasks = [asyncio.create_task(run_agent(a)) for a in sorted(set(agents))]
            for t in asyncio.as_completed(tasks):
                a, p = await t
                sub_results[a] = p

        # 汇总上下文：上一轮 results + 这轮增量 sub_results
        base_results = as_dict(session_ctx.get("results")).copy() if session_ctx.get("results") else {}
        base_results.update(sub_results)
        session_ctx["results"] = base_results  # 刷新上下文

        # 生成回答（Markdown），使用最新的 title/year + results
        answer_prompt = f"""
你是上下文对话助手。以下是最新的结构化结果(JSON)与用户问题，请基于它们输出简洁、直接可读的 Markdown 回答。
- 如果用户问对比/解释/推荐，给出明确结论与理由。
- 如涉及不确定，请说明来源局限。

[标题] {title or '—'} （{year or '—'}）
[结构化JSON]
{json.dumps(base_results, ensure_ascii=False, default=str)}

[用户问题]
{user_text}
"""
        await send({"type":"chat_start", "data":{"role":"assistant","text":"开始撰写回答…"}})
        answer_resp = await gen_async(model=MODEL_PLANNER, contents=answer_prompt, config=types.GenerateContentConfig())
        answer_text = answer_resp.text or "(无输出)"
        # 流式逐字符
        await stream_text("chat_delta", answer_text, delay=0.02)
        await send({"type":"chat_end", "data":{"ok": True}})

    try:
        # 第一帧应包含 { user_query }
        init = await ws.receive_json()
        user_query = init.get("user_query") or ""
        if not user_query:
            await send({"type":"error","data":"missing user_query"}); await ws.close(); return

        # 先跑完整一轮
        await run_full_pipeline(user_query)

        # 然后进入多轮循环：等待用户继续发问
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")
            if mtype == "user_turn":
                text = msg.get("text") or ""
                if not text.strip():
                    await send({"type":"error","data":"empty follow-up"}); continue
                await answer_followup(text)
            elif mtype == "close":
                await ws.close(); return
            else:
                # 兼容“直接发字符串”的客户端
                if "text" in msg and not mtype:
                    await answer_followup(msg["text"])
                else:
                    await send({"type":"error","data":f"unknown message: {msg}"})

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_json({"type":"error","data":str(e)})
        finally:
            await ws.close()
