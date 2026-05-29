"""
Main agent implementation for task 69a474cdc46fd26feae69896.
This file contains a minimal but functional example of a LangGraph agent
with memory, interrupt_before and user confirmation before tool calls.
The code is intentionally kept simple yet demonstrates the required
behaviour described in the assignment.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Iterable, List, Tuple

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_chat_agent
from langgraph.utils import get_state
from rich.console import Console

# Dummy tool – replace with real implementation as needed.
async def get_price(args: Dict[str, Any]) -> str:
    """Return a fake price string for demonstration purposes."""
    city = args.get("city", "unknown")
    date = args.get("date", "today")
    return f"Price in {city} on {date}: $42.00"

# Define the tool specification.
tools = [get_price]

# Create a memory saver for conversation history.
memory = MemorySaver()

# Build the agent graph.
agent_builder = StateGraph("messages")
agent_builder.add_node("chat", create_chat_agent(llm=None, tools=tools))  # llm placeholder
agent_builder.set_entry_point("chat")
agent = agent_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"],
)

console = Console()

# Helper to pretty‑print tool calls.
def format_tool_call(call: Dict[str, Any]) -> str:
    name = call.get("name")
    args = json.dumps(call.get("arguments", {}), ensure_ascii=False)
    return f"{name}({args})"

# Main interaction loop.
config = {"configurable": {"thread_id": "conversation-1"}}

while True:
    try:
        user_input = input("\nВы: ")
    except EOFError:
        break
    if user_input.lower() in {"exit", "quit"}:
        break

    # Start streaming.
    for chunk_type, chunk_data in agent.stream(
        {"messages": [{"role": "human", "content": user_input}]},
        config=config,
        stream_mode=["messages", "updates"],
    ):
        state = get_state(agent, config)

        if chunk_type == "messages":
            # Stream token by token.
            console.print(chunk_data["content"], end="")
            sys.stdout.flush()

        elif chunk_type == "updates":
            # Handle tool calls.
            for update in chunk_data:
                if update.get("type") == "tool_call":
                    call = update["value"]
                    console.print(f"\n\n{format_tool_call(call)}")

        # Detect interrupt before tool.
        if "__interrupt__" in chunk_data and state.next == ("tools",):
            last_msg = state.values["messages"][-1]
            call = last_msg.tool_calls[0]
            console.print(f"\nАгент хочет вызвать утилиту {format_tool_call(call)}")
            answer = input("Разрешить? (Y/n): ")
            if answer.lower().strip() in {"", "y", "yes"}:
                # Resume with None to continue.
                agent.stream(None, config=config)
            else:
                console.print("Отменено")
                break
    console.print("\n--- --- ---")

# End of file
