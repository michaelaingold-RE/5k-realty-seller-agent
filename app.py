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
    h1 {
        color: #0A2540 !important;
    }
    .stChatMessage {
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #0A2540;
        color: white;
        border-radius: 6px;
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
</style>
""", unsafe_allow_html=True)

st.title("5K Realty – Seller Intake Agent")
st.caption("Internal tool for qualifying seller leads")

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
