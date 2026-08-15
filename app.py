import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from seller_agent import agent
import traceback

st.set_page_config(
    page_title="5K Realty – Seller Intake Agent",
    page_icon="🏠",
    layout="centered"
)

st.title("5K Realty – Seller Intake Agent")
st.caption("Internal tool for qualifying seller leads")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "summary" not in st.session_state:
    st.session_state.summary = None

# Sidebar
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.summary = None
        st.rerun()

    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("Have a normal conversation with the agent as if you are a potential seller. When enough information is collected, a lead summary will appear below.")

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent.invoke({
                    "messages": st.session_state.messages,
                    "seller_info": {},
                    "ready_for_summary": False,
                    "summary": None
                })

                # Update message history
                st.session_state.messages = result["messages"]

                # Show the latest AI response
                last_ai = None
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage) and msg.content:
                        last_ai = msg
                        break

                if last_ai:
                    st.markdown(last_ai.content)

                # Capture summary if available
                if result.get("summary"):
                    st.session_state.summary = result["summary"]

            except Exception as e:
                st.error(f"An error occurred: {type(e).__name__}: {str(e)}")
                st.code(traceback.format_exc())

# Display final summary
if st.session_state.summary:
    st.divider()
    st.subheader("📋 Lead Summary")
    st.code(st.session_state.summary, language="text")
