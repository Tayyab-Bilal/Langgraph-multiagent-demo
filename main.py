import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from graph import build_graph
from rag import get_vectorstore

load_dotenv()

def run_chat():
    print("🔧 Loading RAG vectorstore from policy docs...")
    get_vectorstore()  # pre-load once

    print("Building agent graph...")
    graph = build_graph()

    print("\n" + "="*50)
    print("  TechFlow Electronics - Customer Support")
    print("="*50)
    print("Type 'quit' to exit\n")

    # Initial state
    state = {
        "messages": [],
        "intent": "",
        "cancellation_reason": "",
        "customer_email": "",
        "customer_data": {},
        "outcome": "",
        "retention_action": "",
        "processed": False,
    }

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        # Add user message
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Run the graph
        try:
            result = graph.invoke(state)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        # Update state
        state = {**state, **result}

        ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
        if ai_messages:
            print(f"\nAgent: {ai_messages[-1].content}\n")

        # Reset after completed conversation
        if state.get("processed"):
            print("-"*50)
            print("Conversation complete. Starting fresh...\n")
            state = {
                "messages": [],
                "intent": "",
                "cancellation_reason": "",
                "customer_email": "",
                "customer_data": {},
                "outcome": "",
                "retention_action": "",
                "processed": False,
            }

if __name__ == "__main__":
    run_chat()