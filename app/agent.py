"""LangGraph tool-calling agent for multi-document legal queries — the Phase-3 upgrade over the
single-shot v1 chain. ReAct loop: the agent node (LLM with the 2 tools bound) cycles with a
ToolNode until it has enough to answer. Multi-turn memory via MessagesState + MemorySaver, keyed
by thread_id. Model swaps gemma4 <-> Bedrock <-> Gemini in config — nothing here changes."""
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable

from app.llm import get_chat_model
from app.tools import list_contracts, search_contracts
from app.config import LLM_PROVIDER

TOOLS = [list_contracts, search_contracts]

AGENT_SYSTEM = """You are a legal-contract research assistant. You answer using ONLY your tools.

Always work out WHICH contract the question is about, then SCOPE the search to it — scoping is what
makes specific clauses (governing law, dates, renewal/termination terms, liability caps) findable.

Resolving the contract:
1. Take the contract reference straight from the question — whatever the user calls it: a party
   name, a program name, or a title ("the Chase affiliate agreement" -> "Chase"). If they say
   "it"/"that contract", use the contract from earlier in this conversation.
2. Call search_contracts(query, contract_name=<that reference>). You do NOT need the exact title —
   the tool resolves party/program names to the right document on its own.
3. Only if the tool replies that the name points at SEVERAL different contracts, relay its question
   and ask the user which one they mean. Don't pre-emptively ask when you have a usable reference.
4. If the user names NO contract and none is implied, ASK which contract (a bare "what's the
   governing law?" has no answer without knowing the contract).

Other patterns:
- COMPARE a clause across contracts: call search_contracts once PER contract, then synthesize.
- "WHICH contracts ..." / sweeps: call list_contracts, then search the relevant ones.
- OPEN-ENDED questions not tied to one contract: search_contracts with contract_name empty.

Answering:
- Answer ONLY from what the tools return — never guess or use outside knowledge. If the tools find
  nothing relevant, say so plainly.
- Write a complete, helpful answer in plain language: state the answer directly, then give the
  context that makes it useful (the relevant terms, conditions, or exact wording the clause turns
  on). Quote the key contract language where it matters. This is a conversation with a user, not a
  benchmark — don't truncate to a single bare line, but don't pad with filler either.
- Name the contract you're drawing from by its title so the user can trace it."""

# Bind tools once. With gemma4/Bedrock/Gemini this uses each provider's native function-calling.
_llm = get_chat_model().bind_tools(TOOLS)


def _agent_node(state: MessagesState) -> dict:
    """The reasoning turn: inject the system prompt once, then let the model think / call tools."""
    msgs = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in msgs):
        msgs = [SystemMessage(content=AGENT_SYSTEM)] + msgs
    return {"messages": [_llm.invoke(msgs)]}


def build_agent():
    g = StateGraph(MessagesState)
    g.add_node("agent", _agent_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)   # tool calls? -> "tools" ; else -> END
    g.add_edge("tools", "agent")                         # the cycle: results go back to reason
    return g.compile(checkpointer=MemorySaver())


agent = build_agent()


def _as_text(content) -> str:
    """Coerce a message's content to a plain string. Bedrock returns a list of content blocks
    (e.g. [{'type':'text','text':...}]) rather than a string; Ollama/Gemini return a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content
                       if isinstance(b, dict) and b.get("type") == "text").strip()
    return str(content)


@traceable(name="answer_question", metadata={"provider": LLM_PROVIDER})
def ask(question: str, thread_id: str = "default") -> str:
    """One turn. Same thread_id reuses prior history (multi-turn memory).

    @traceable makes each call a named 'answer_question' root run in LangSmith, with the whole
    LangGraph loop (agent node, tool calls, LLM calls) nested underneath as a waterfall span
    tree. It's a transparent no-op when tracing is off (no key /
    LANGCHAIN_TRACING_V2 unset), so eval and offline runs are unaffected. `question` and `thread_id`
    are auto-captured as run inputs."""
    return ask_detailed(question, thread_id)["answer"]


def _steps_from(messages) -> list[dict]:
    """Pull the agent's tool-call trace out of the message list: which tool, with what args,
    in order. Lets the UI show WHAT the agent did (scoped to X, searched corpus-wide, listed)."""
    out = []
    for m in messages:
        for tc in (getattr(m, "tool_calls", None) or []):
            args = tc.get("args", {}) or {}
            out.append({"name": tc["name"],
                        "contract": args.get("contract_name", "").strip(),
                        "query": args.get("query", "").strip()})
    return out


@traceable(name="answer_question", metadata={"provider": LLM_PROVIDER})
def ask_detailed(question: str, thread_id: str = "default") -> dict:
    """Like ask() but also returns the tool-call trace, so the UI can show what the agent did
    (which contracts it scoped to, how many searches). Returns {"answer": str, "steps": [...]}."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 16}
    out = agent.invoke({"messages": [HumanMessage(content=question)]}, config)
    msgs = out["messages"]
    return {"answer": _as_text(msgs[-1].content), "steps": _steps_from(msgs)}
