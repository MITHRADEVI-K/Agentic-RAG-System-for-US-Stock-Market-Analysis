"""LangGraph workflow: route -> agent <-> tools -> validate.

    START -> route -> agent --(tool calls?)--> tools -> agent ...
                          +--(final answer)--> validate -> END

Claude is called through the official Anthropic SDK inside the `agent` node;
LangGraph owns the control flow, state and loop limits.
"""
from __future__ import annotations

import json
from typing import Any, TypedDict

import anthropic
from langgraph.graph import END, START, StateGraph

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.router import route
from src.agent.tool_registry import TOOL_FUNCTIONS, TOOL_SCHEMAS
from src.config import CLAUDE_EFFORT, CLAUDE_MODEL
from src.validation.answer_validator import validate_numbers

MAX_TOOL_ROUNDS = 6

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


class AgentState(TypedDict, total=False):
    question: str
    routing: dict
    messages: list          # Anthropic-format conversation
    tool_log: list          # [{name, input, output}] for UI + validation
    rounds: int
    answer: str
    validation: dict
    error: str


def route_node(state: AgentState) -> AgentState:
    routing = route(state["question"])
    hint = (f"\n\n[Router] Detected tickers: {routing['tickers'] or 'none'}; "
            f"intents: {routing['intents']}.")
    return {
        "routing": routing,
        "messages": [{"role": "user", "content": state["question"] + hint}],
        "tool_log": [],
        "rounds": 0,
    }


def agent_node(state: AgentState) -> AgentState:
    tools = [TOOL_SCHEMAS[n] for n in state["routing"]["tools"]]
    try:
        response = _get_client().beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=state["messages"],
            thinking={"type": "adaptive"},
            output_config={"effort": CLAUDE_EFFORT},
            # If a safety classifier declines, the API retries on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError:
        return {"error": "Anthropic API key missing or invalid - set ANTHROPIC_API_KEY in .env"}
    except anthropic.RateLimitError:
        return {"error": "Anthropic rate limit hit - wait a moment and retry."}
    except anthropic.APIStatusError as exc:
        return {"error": f"Claude API error {exc.status_code}: {exc.message}"}
    except anthropic.APIConnectionError:
        return {"error": "Could not reach the Claude API - check your internet connection."}

    # Append the full content (incl. thinking blocks) so the next turn stays valid.
    messages = state["messages"] + [{"role": "assistant", "content": response.content}]
    if response.stop_reason == "refusal":
        return {"messages": messages, "error": "The model declined this request."}
    answer = "".join(b.text for b in response.content if b.type == "text")
    return {"messages": messages, "answer": answer}


def tools_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]["content"]
    results, log = [], list(state["tool_log"])
    for block in last:
        if block.type != "tool_use":
            continue
        fn = TOOL_FUNCTIONS.get(block.name)
        try:
            if fn is None or block.name not in state["routing"]["tools"]:
                raise ValueError(f"Tool {block.name} is not available for this question")
            output: Any = fn(**block.input)
            is_error = False
        except Exception as exc:  # report failures to the model instead of crashing
            output, is_error = {"error": str(exc)}, True
        log.append({"name": block.name, "input": block.input, "output": output})
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": json.dumps(output, default=str),
            "is_error": is_error,
        })
    return {
        "messages": state["messages"] + [{"role": "user", "content": results}],
        "tool_log": log,
        "rounds": state["rounds"] + 1,
    }


def validate_node(state: AgentState) -> AgentState:
    outputs = [t["output"] for t in state.get("tool_log", [])]
    return {"validation": validate_numbers(state.get("answer", ""), outputs)}


def _after_agent(state: AgentState) -> str:
    if state.get("error"):
        return END
    last = state["messages"][-1]["content"]
    wants_tools = any(b.type == "tool_use" for b in last)
    if wants_tools and state["rounds"] < MAX_TOOL_ROUNDS:
        return "tools"
    return "validate"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", route_node)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("validate", validate_node)
    g.add_edge(START, "route")
    g.add_edge("route", "agent")
    g.add_conditional_edges("agent", _after_agent, ["tools", "validate", END])
    g.add_edge("tools", "agent")
    g.add_edge("validate", END)
    return g.compile()


_graph = None


def run_analysis(question: str) -> AgentState:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph.invoke({"question": question})


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Analyze AAPL: technical trend, recent news and major risks."
    result = run_analysis(q)
    print(result.get("error") or result["answer"])
    print("\nTools used:", [t["name"] for t in result.get("tool_log", [])])
    print("Validation:", result.get("validation"))
