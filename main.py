from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from todoist_api_python.api import TodoistAPI

load_dotenv()

todoist_api_key = os.getenv("TODOIST_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

todoist = TodoistAPI(todoist_api_key)

# tool is a decorator and also a jargon from llm

@tool
def add_task(task, desc=None):
    """ Add a new task to the user's task list. Use this when user wants to add or create a task"""
    todoist.add_task(content=task,  description=desc)
    # print("Adding a task")
    # print(task)
    # print("Task added")   

# Initialising the LLM

tools = [add_task]

llm = ChatGoogleGenerativeAI(
    model ='gemini-2.5-flash',
    google_api_key=gemini_api_key,
    temperature =0.3
)


system_prompt = "You are a helpful assistant. You will help the user add tasks."
user_input = "add task to get a new laptop also with a description of considering the local market"

# Constructing the prompt:

prompt = ChatPromptTemplate([
    ("system", system_prompt),
    ("user", user_input),
    MessagesPlaceholder('agent_scratchpad')
])

# chain = promt | llm | StrOutputParser()
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# response = chain.invoke({"input":user_input})

response = agent_executor.invoke({"input": user_input})

# print(response)
print(response['output'])