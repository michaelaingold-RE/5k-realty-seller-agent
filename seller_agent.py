from typing import TypedDict, Annotated, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import operator
import streamlit as st

# -----------------------------
# Savings Calculator Tool
# -----------------------------
@tool
def calculate_savings(sale_price: float) -> str:
    """
    Calculate estimated seller savings using 5K Realty's fee 
    ($5,000 or 1%, whichever is greater) compared to a traditional 5.5% commission.
    """
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

# -----------------------------
# Initialize the LLM
# Reads the API key from Streamlit secrets
# -----------------------------
api_key = st.secrets.get("OPENAI_API_KEY", None)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.4,
    api_key=api_key
)
llm_with_tools = llm.bind_tools(tools)


# -----------------------------
# State
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    seller_info: dict
    ready_for_summary: bool
    summary: Optional[str]


# -----------------------------
# Structured Models
# -----------------------------
class SellerInfo(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    timeline: Optional[str] = None
    motivation: Optional[str] = None
    condition: Optional[str] = None
    expected_price: Optional[str] = None
    working_with_agent: Optional[str] = None
    additional_notes: Optional[str] = None


class RoutingDecision(BaseModel):
    decision: Literal["continue_conversation", "create_summary"]
    reason: str


class FinalSummary(BaseModel):
    is_qualified: bool
    qualification_reason: str
    recommended_next_step: str
    estimated_savings: Optional[str] = None
    seller_summary: str


SYSTEM_PROMPT = """
You are the Seller Intake Agent for 5K Realty, a flat-fee real estate brokerage serving Fort Mill, Lake Wylie, Tega Cay, Rock Hill, Ballantyne, and the greater Charlotte area.

Your role is to have a natural, professional conversation with potential sellers to:
1. Understand their situation
2. Collect key information
3. Clearly explain the $5,000 or 1% fee structure (whichever is greater)
4. Help them see potential savings versus traditional commissions using the calculate_savings tool when a price is discussed
5. Determine if they are a serious lead

Tone: Warm, professional, clear, and confident.

Information to collect naturally:
- Name
- Phone and/or email
- Property address
- Timeline to sell
- Motivation
- Home condition
- Expected sale price
- Whether they are working with another agent

Guidelines:
- Ask one or two questions at a time.
- Use the calculate_savings tool whenever the seller mentions a potential sale price.
- Be clear about the pricing model.
- Never invent details.
"""


# -----------------------------
# Nodes
# -----------------------------
def conversation_node(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tool_node(state: AgentState):
    """Execute any tool calls"""
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


def route_after_conversation(state: AgentState) -> Literal["tool_node", "router"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return "router"


def route_node(state: AgentState):
    router = llm.with_structured_output(RoutingDecision)

    prompt = f"""
    Review the conversation and decide the next step.

    Current seller info: {state.get('seller_info', {})}

    Choose "continue_conversation" if important details are still missing
    (especially name, contact info, address, or timeline).

    Choose "create_summary" only when we have enough information to evaluate the lead.

    Conversation:
    {state['messages']}
    """
    result = router.invoke(prompt)
    return {"ready_for_summary": result.decision == "create_summary"}


def should_continue(state: AgentState) -> Literal["extract_info", "conversation"]:
    if state.get("ready_for_summary"):
        return "extract_info"
    return "conversation"


def extract_info_node(state: AgentState):
    extractor = llm.with_structured_output(SellerInfo)
    prompt = f"""
    Extract all clearly provided seller information.
    Do not invent details.

    Conversation:
    {state['messages']}
    """
    result = extractor.invoke(prompt)
    return {"seller_info": result.model_dump(exclude_none=True)}


def summary_node(state: AgentState):
    summarizer = llm.with_structured_output(FinalSummary)
    prompt = f"""
    Create a clear summary of this seller lead.

    Seller Info: {state.get('seller_info')}
    Conversation: {state['messages']}

    Mark as qualified only if they appear serious and we have solid contact + property details.
    Include estimated savings if available.
    """
    result = summarizer.invoke(prompt)

    formatted = f"""
=== SELLER LEAD SUMMARY ===

Qualified: {"YES" if result.is_qualified else "NO"}
Reason: {result.qualification_reason}
Next Step: {result.recommended_next_step}
Estimated Savings: {result.estimated_savings or "Not enough price information"}

Seller Details:
{result.seller_summary}
"""
    return {"summary": formatted}


# -----------------------------
# Build Graph
# -----------------------------
workflow = StateGraph(AgentState)

workflow.add_node("conversation", conversation_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("router", route_node)
workflow.add_node("extract_info", extract_info_node)
workflow.add_node("summary", summary_node)

workflow.set_entry_point("conversation")

workflow.add_conditional_edges(
    "conversation",
    route_after_conversation,
    {
        "tool_node": "tool_node",
        "router": "router",
    },
)
workflow.add_edge("tool_node", "conversation")
workflow.add_conditional_edges(
    "router",
    should_continue,
    {
        "conversation": "conversation",
        "extract_info": "extract_info",
    },
)
workflow.add_edge("extract_info", "summary")
workflow.add_edge("summary", END)

agent = workflow.compile()
