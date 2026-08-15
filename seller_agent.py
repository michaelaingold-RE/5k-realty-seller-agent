from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
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

def get_agent():
    api_key = st.secrets["OPENAI_API_KEY"].strip()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=api_key)
    
    agent = create_react_agent(
        llm,
        tools=[calculate_savings],
        state_modifier=SystemMessage(content=SYSTEM_PROMPT)
    )
    return agent

# Create the agent
agent = get_agent()
