import logging
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.playground import Playground, serve_playground_app

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoggingAgent(Agent):
    def handle_message(self, message):
        logger.info(f"Agent {self.name} received message: {message}")
        response = super().handle_message(message)
        logger.info(f"Agent {self.name} response: {response}")
        return response

web_agent = LoggingAgent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    storage=SqliteAgentStorage(table_name="web_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

finance_agent = LoggingAgent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Always use tables to display data"],
    storage=SqliteAgentStorage(table_name="finance_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

agent_team = LoggingAgent(
    team=[web_agent, finance_agent],
    name="Agent Team (Web+Finance)",
    model=OpenAIChat(id="gpt-4o"),
    show_tool_calls=True,
    markdown=True,
)

app = Playground(agents=[agent_team]).get_app()

if __name__ == "__main__":
    logger.info("Starting the playground app")
    serve_playground_app("finance_agent_team:app", reload=True)