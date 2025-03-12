import streamlit as st
from agno.agent import Agent
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load API keys from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Ensure API key is set
if not GEMINI_API_KEY:
    st.error("Gemini API key is missing! Please set it in a .env file.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def generate_gemini_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-pro-latest")  # Use the latest supported model
        response = model.generate_content(prompt)
        return response.text if response else "No response."
    except Exception as e:
        return f"Error: {str(e)}"

# Create Simple Agent
simple_agent = Agent(
    name="Simple Agent",
    role="General AI assistant",
    storage=SqliteAgentStorage(table_name="simple_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

# Create Web Search Agent
web_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    tools=[DuckDuckGoTools()],
    storage=SqliteAgentStorage(table_name="web_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

# Create Research Agent
research_agent = Agent(
    name="Research Agent",
    role="Conduct in-depth research",
    storage=SqliteAgentStorage(table_name="research_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

# Create Finance Agent
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Always use tables to display data"],
    storage=SqliteAgentStorage(table_name="finance_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

# Combine Agents into a Team
agent_team = Agent(
    team=[simple_agent, web_agent, research_agent, finance_agent],
    name="Agent Team (Simple + Web + Research + Finance)",
    show_tool_calls=True,
    markdown=True,
)

# Streamlit UI
st.title("🤖 AI Multi-Agent System (Powered by Gemini)")
st.write("Choose an agent and enter your query!")

agent_choice = st.selectbox("Select an Agent", ["Simple Agent", "Web Search Agent", "Research Agent", "Finance Agent", "Gemini AI"])
query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if query:
        if agent_choice == "Gemini AI":
            response = generate_gemini_response(query)
        else:
            selected_agent = {
                "Simple Agent": simple_agent,
                "Web Search Agent": web_agent,
                "Research Agent": research_agent,
                "Finance Agent": finance_agent
            }.get(agent_choice, simple_agent)
            response = selected_agent.run(query)  # Ensure correct agent execution
        
        st.markdown(response)
    else:
        st.warning("Please enter a query!")
