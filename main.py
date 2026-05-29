import json
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_chat_agent
from rich.console import Console

console = Console()

# Define a simple tool that returns weather info (mock)
def get_price(args: str):
    # In real scenario, call an API. Here we just echo.
    return f"Weather data for {args}"

# LLM and agent setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Memory for conversation
memory = MemorySaver()

# Create agent with interrupt_before to pause before tool calls
agent = create_chat_agent(
    llm,
    tools=[{"name": "get_price", "func": get_price, "description": "Get weather info"}],
    system_prompt="You are a helpful assistant.",
    checkpointer=memory,
    interrupt_before=["tools"],
)

# Helper to run agent with confirmation
async def ask_and_run(user_input: dict | None, config):
    from langgraph.graph import StateGraph
    # Stream the agent output
    async for chunk in agent.stream(user_input or {}, config=config, stream_mode=["messages", "updates"]):
        state = agent.get_state(config)
        chunk_type, chunk_data = chunk
        if chunk_type == "messages":
            console.print(chunk_data["content"], end="")
        elif chunk_type == "updates":
            # Handle tool calls
            for update in chunk_data:
                if update.get("type") == "tool_call":
                    name = update["name"]
                    args = json.dumps(update["arguments"])
                    console.print(f"\n\n{name}({args})")
        # Check for interrupt
        if "__interrupt__" in chunk_data and state.next == ("tools",):
            tool_call = state.values["messages"][-1].tool_calls[0]
            name = tool_call["name"]
            args = json.dumps(tool_call["arguments"])
            console.print(f"\nAgent wants to call {name}({args})")
            ans = input("Разрешить? (Y/n): ")
            if ans.lower().strip() == "y":
                await ask_and_run(None, config)
            else:
                console.print("Отменено")
                return

# Main chat loop
if __name__ == "__main__":
    thread_id = "conversation-1"
    config = {"configurable": {"thread_id": thread_id}}
    while True:
        user_input = input("\nВы: ")
        if user_input.lower() in ("exit", "quit"):
            break
        await ask_and_run({"messages": [{"role": "human", "content": user_input}]}, config)
