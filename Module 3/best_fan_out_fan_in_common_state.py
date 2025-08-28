"""
Best approach: keep my reducer’s readiness guard (to truly wait for both) and your merger LLM prompt (to hide internal branches and ignore an unhelpful channel). That combination is optimal for channel sharing and synchronization while producing a single, polished answer.

Code solution (pinned: langgraph==0.2.34, langchain-core==0.2.39, langchain==0.2.14)

Adds an anchor length at fanout, hard “wait-for-both” gating in reducer, and an aggregator prompt that hides internal branches. Replace ... with your concrete runnables and tools.

"""


# Versions: langgraph==0.2.34, langchain-core==0.2.39, langchain==0.2.14
from __future__ import annotations
from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from datetime import datetime

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage

# -----------------------
# State
# -----------------------
class OverallState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    messages_assistant_1: Annotated[List[BaseMessage], add_messages]
    messages_assistant_2: Annotated[List[BaseMessage], add_messages]
    # Anchor length of the shared conversation at fanout time
    messages_anchor_len: int
    confidence: Annotated[List[int], add]

# -----------------------
# Placeholders (wire your actual runnables/tools)
# -----------------------
llm_with_sql_tools1 = ...  # Runnable that supports .invoke(List[BaseMessage]) -> BaseMessage
llm_with_sql_tools2 = ...
aggregator_llm = ...       # Runnable for merging responses
sql_tools1 = ...
sql_tools2 = ...
rag_tools = ...

sys_msg_sql = SystemMessage(content="You are a hotel reservation SQL assistant. Use tools when necessary.")
merger_sys_msg = SystemMessage(content=(
    "You see multiple internal assistants' drafts answering the user's reservation request. "
    "Produce ONE final user-facing answer:\n"
    "- Do NOT reveal internal assistants or their count.\n"
    "- If any draft says it cannot help or lacks data, ignore it and use the helpful draft.\n"
    "- Be concise and accurate."
))

# -----------------------
# Routing helpers
# -----------------------
def tools_condition(state: OverallState, *, messages_key: str = "messages") -> str:
    msgs = state.get(messages_key, [])
    if not msgs:
        return "__end__"
    last = msgs[-1]
    if getattr(last, "additional_kwargs", {}).get("tool_calls"):
        return "tools"
    return "__end__"

# -----------------------
# Nodes
# -----------------------
def conv_assistant(state: OverallState) -> Dict[str, Any]:
    # Your top-level conversational node; update 'messages' as needed.
    return {}

def fanout_to_sql(state: OverallState) -> Dict[str, Any]:
    base_msgs = state.get("messages", [])
    return {
        "messages_assistant_1": base_msgs,
        "messages_assistant_2": base_msgs,
        "messages_anchor_len": len(base_msgs),
        "confidence": state.get("confidence", []),
    }

def sql_assistant1(state: OverallState) -> Dict[str, Any]:
    msgs = state.get("messages_assistant_1", [])
    _ = datetime.now().strftime("%Y-%m-%d")
    resp = llm_with_sql_tools1.invoke([sys_msg_sql] + msgs)
    if not isinstance(resp, BaseMessage):
        resp = AIMessage(content=str(resp))
    return {"messages_assistant_1": [resp]}

def sql_assistant2(state: OverallState) -> Dict[str, Any]:
    msgs = state.get("messages_assistant_2", [])
    _ = datetime.now().strftime("%Y-%m-%d")
    resp = llm_with_sql_tools2.invoke([sys_msg_sql] + msgs)
    if not isinstance(resp, BaseMessage):
        resp = AIMessage(content=str(resp))
    return {"messages_assistant_2": [resp]}

def reduce_sql_results(state: OverallState) -> Dict[str, Any]:
    """
    Hard gate: only aggregate when both branches produced at least one new message
    beyond the fanout anchor. Otherwise, return {} to wait.
    """
    anchor = state.get("messages_anchor_len", 0)
    msgs1 = state.get("messages_assistant_1", [])
    msgs2 = state.get("messages_assistant_2", [])
    new1 = msgs1[anchor:] if len(msgs1) > anchor else []
    new2 = msgs2[anchor:] if len(msgs2) > anchor else []

    # Wait until both channels have produced at least one message
    if not new1 or not new2:
        return {}

    # Merge intelligently without exposing the assistants
    merged = aggregator_llm.invoke(
        [merger_sys_msg] + state.get("messages", []) + new1 + new2
    )
    if not isinstance(merged, BaseMessage):
        merged = AIMessage(content=str(merged))

    # Append only the merged user-facing message to the shared conversation
    return {"messages": [merged]}

def rag_assistant(state: OverallState) -> Dict[str, Any]:
    return {}

def generate(state: OverallState) -> Dict[str, Any]:
    return {}

# -----------------------
# Graph assembly
# -----------------------
from functools import partial

builder = StateGraph(OverallState)

builder.add_node("conv_assistant", conv_assistant)
builder.add_node("fanout_reservation", fanout_to_sql)
builder.add_node("reservation_assistant1", sql_assistant1)
builder.add_node("reservation_assistant2", sql_assistant2)
builder.add_node("reservation_reducer", reduce_sql_results)
builder.add_node("compliance_checker", rag_assistant)
builder.add_node("retriever", generate)

builder.add_node("sql_tools1", ToolNode(sql_tools1, messages_key="messages_assistant_1"))
builder.add_node("sql_tools2", ToolNode(sql_tools2, messages_key="messages_assistant_2"))
builder.add_node("rag_tools", ToolNode(rag_tools))

builder.add_edge(START, "conv_assistant")

def choose_next_node(state: OverallState) -> str:
    # Implement your own routing; default to reservations path
    return "fanout_reservation"

builder.add_conditional_edges(
    "conv_assistant",
    choose_next_node,
    path_map=["fanout_reservation", "compliance_checker", "__end__"],
)

# Fan-out
builder.add_edge("fanout_reservation", "reservation_assistant1")
builder.add_edge("fanout_reservation", "reservation_assistant2")

# Tools routing per branch, then go to reducer; reducer will wait until both are ready
tc1 = partial(tools_condition, messages_key="messages_assistant_1")
tc2 = partial(tools_condition, messages_key="messages_assistant_2")
builder.add_conditional_edges("reservation_assistant1", tc1, path_map={"tools": "sql_tools1", "__end__": "reservation_reducer"})
builder.add_conditional_edges("reservation_assistant2", tc2, path_map={"tools": "sql_tools2", "__end__": "reservation_reducer"})
builder.add_edge("sql_tools1", "reservation_assistant1")
builder.add_edge("sql_tools2", "reservation_assistant2")

# Compliance branch
builder.add_conditional_edges("compliance_checker", lambda s: "tools", path_map={"tools": "rag_tools", "__end__": "conv_assistant"})
builder.add_edge("rag_tools", "retriever")
builder.add_edge("retriever", "conv_assistant")

builder.add_edge("reservation_reducer", END)

# Compile with your checkpointer
memory = ...  # e.g., SqliteSaver(...)
react_graph_with_memory = builder.compile(checkpointer=memory)
