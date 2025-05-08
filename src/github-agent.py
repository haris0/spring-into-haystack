from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.agents import Agent
from haystack_integrations.tools.mcp import MCPTool, StdioServerInfo
from dotenv import load_dotenv
import os

load_dotenv()

github_mcp_server = StdioServerInfo(
    command="docker",
    args=[
        "run",
        "-i",
        "--rm",
        "-e",
        "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server"
    ],
    env={
        "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN'),
    }
)

print("MCP server is created")

tool_1 = MCPTool(
  name="get_file_contents",
  description="Get contents of a file or directory",
  server_info=github_mcp_server,
)

tool_2 = MCPTool(
  name="create_issue",
  description="Create a new issue in a GitHub repository",
  server_info=github_mcp_server,
)

print("MCP tools are created")

generator = HuggingFaceAPIChatGenerator(
  api_type="serverless_inference_api",
  api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
)

## TODO: Create your Agent here:
agent = Agent(
    chat_generator=generator,
    tools=[tool_1, tool_2],
    exit_conditions=["text"]
)

print("Agent created")

## Example query to test your agent
user_input = "Can you find the typo in the README of haris0/spring-into-haystack and open an issue about how to fix it?"

## (OPTIONAL) Feel free to add other example queries that can be resolved with this Agent
agent.warm_up()
response = agent.run(messages=[ChatMessage.from_user(text=user_input)])

## Print the agent thinking process
print(response)
## Print the final response
print(response["messages"][-1].text)
