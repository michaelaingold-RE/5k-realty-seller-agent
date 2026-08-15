import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from seller_agent import agent
import traceback

st.set_page_config(
    page_title="5K Realty – Seller Intake Agent",
    page_icon="🏠",
    layout="centered"
)

# Custom CSS for navy + gold branding
st.markdown("""
<style>
    :root {
        --navy: #0A2540;
        --gold: #C9A227;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    .stButton > button {
        background-color: #0A2540;
        color: white;
        border-radius: 6px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #C9A227;
        color: #0A2540;
    }
    section[data-testid="stSidebar"] {
        background-color: #0A2540;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Branded Header
st.markdown("""
<div style="text-align:center; margin-bottom: 1.5rem;">
    <h1 style="color:#0A2540; margin-bottom:0.2rem; font-size:2rem;">5K Realty</h1>
    <p style="color:#C9A227; font-size:1.15rem; font-weight:500; margin:0;">
        Keep More of Your Equity
    </p>
    <p style="color:#666; font-size:0.95rem; margin-top:0.4rem;">
        Seller Intake Agent
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("Have a normal conversation with the agent as if you are a potential seller.")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build clean messages for the agent
    clean_messages = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            clean_messages.append(HumanMessage(content=msg["content"]))
        else:
            clean_messages.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent.invoke({"messages": clean_messages})

                ai_content = None
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage) and msg.content:
                        ai_content = msg.content
                        break

                if ai_content:
                    st.markdown(ai_content)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": ai_content
                    })
                else:
                    st.warning("No response generated.")

            except Exception as e:
                st.error(f"An error occurred: {type(e).__name__}: {str(e)}")
                st.code(traceback.format_exc())
