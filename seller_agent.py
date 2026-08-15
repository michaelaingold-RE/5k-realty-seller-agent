from typing import Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import streamlit as st

# -----------------------------
# Savings Calculator Tool
# -----------------------------
@tool
def calculate_savings(sale_price: float) -> str:
    """Calculate estimated seller savings using 5K Realty's fee ($5,000 or 1%, whichever is greater) compared to a traditional 5.5% commission."""
    if sale_price <= 0:
        return "Please provide a valid sale price greater than zero."

    traditional_fee = sale_price * 0.055
    our_fee = max(5000, sale_price * 0.01)
    savings = traditional_fee - our_fee

    return (
        f"At a sale price of ${sale_price:,.0f}:\n"
        f"- Traditional 5.5% commission: ${traditional_fee:,.0f}\n"
        f"- 5K Realty fee ($5,000 or 1%): ${our_fee:,.0f}\n"
        f"- Estimated savings: ${savings:,.0f}"
    )


tools = [calculate_savings]

SYSTEM_PROMPT = """
You are the Seller Intake Agent for 5K Realty, a flat-fee real estate brokerage serving Fort Mill, Lake Wylie, Tega Cay, Rock Hill, Ballantyne, and the greater Charlotte area.

Your job is to have a helpful, professional conversation with potential sellers.

Key points you must cover naturally:
- 5K Realty charges $5,000 or 1% of the sale price (whichever is greater)
- This usually saves sellers thousands compared to traditional 5.5–6% commissions
- You serve the Greater Charlotte area

Guidelines:
- Be warm, clear, and professional
- Ask one or two questions at a time
- When the seller mentions a possible sale price, use the calculate_savings tool
- Collect basic information: name, contact info, property address, timeline, and motivation
- Never invent details
"""

# -----------------------------
# State
# -----------------------------
class State:
    messages: Annotated[List, add_messages]


# -----------------------------
# LLM + Tool binding
# -----------------------------
def get_llm():
    api_key = st.secrets["OPENAI_API_KEY"].strip()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=api_key)
    return llm.bind_tools(tools)


# -----------------------------
# Nodes
# -----------------------------
def chatbot(state):
    llm = get_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def tool_node(state):
    last_message = state["messages"][-1]
    tool_messages = []

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "calculate_savings":
                result = calculate_savings.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )
    return {"messages": tool_messages}


def should_continue(state):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# -----------------------------
# Build the simple graph
# -----------------------------
graph_builder = StateGraph(dict)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("chatbot")
graph_builder.add_conditional_edges("chatbot", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "chatbot")

agent = graph_builder.compile()
