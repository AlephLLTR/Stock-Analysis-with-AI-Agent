
# IMPORTS

import json
import os


import yfinance as yf

from datetime import datetime
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults

from api_settings import fetcher as fetch

#==============================================================================#

# Auxiliares

def FUNC__fetchStock(ticket):
  return yf.download(ticket, start="2023-08-08", end="2024-08-08")

# Importando OPENAI LLM - GPT
os.environ['OPENAI_API_KEY'] = fetch.fetchKey('api_settings/keyring.txt', 'chatgpt')
gpt = ChatOpenAI(model="gpt-4o-mini")

# Importando Google LLM - Gemini 1.5
os.environ['GOOGLE_API_KEY'] = fetch.fetchKey('api_settings/keyring.txt', 'gemini')
gemini = ChatGoogleGenerativeAI(model="gemini-1.0-pro")

#==============================================================================#

# TOOLS

TOOL_yf_fetch = Tool(
  name = "Yahoo Finance Fetch Tool",
  description = "Fetches stock price history for {ticket} from the Yahoo Finance API",
  func = lambda ticket: FUNC__fetchStock(ticket)
)

TOOL_search = DuckDuckGoSearchResults(backend='news', num_results=10)

#==============================================================================#

# AGENTS

AGENT_trendsAnalyst = Agent(
  role = "Senior Market Trends Analyst",
  goal = "Find the {ticket} stock price and analyze market trends",
  backstory = """You're a highly experienced investor and you're an expert in analyzing market trends 
  and make reasonably accurate predictions about future stock price growth""",
  verbose = True,
  llm = gemini,
  max_iter = 5,
  memory = True,
  allow_delegation = False,
  tools = [TOOL_yf_fetch]
)
  
AGENT_newsAnalyst = Agent(
  role = "Senior Market News Analyst",
  goal = """Create a short summary of news regarding the stock {ticket} 
  and specify the current trend based on the context of the news - up, down or sideways. 
  For each requested stock, specify a number between 0 and 100, where 0 indicates extreme fear and 100 indicates extreme greed.""",
  backstory = """You're a senior market analyst with over a decade in experience, you specialize in predicting future trends based on current news.
  You're also a master level analyst in the traditional markets and you've got a deep understanding of human psychology and market trends.
  You're capable of reading, interpreting and understanding current news - their headlines and contents -  however, you still regard them with a healthy dose of skepticism, you also prefer to read articles
  from trustworthy sources""",
  verbose = True,
  llm = gemini,
  max_iter = 10,
  memory = True,
  allow_delegation = False,
  tools = [TOOL_search]
)    

AGENT_writerAnalyst = Agent(
  role = "Senior Stock Analyst",
  goal = "Write an insightful, informative and compelling 3 paragraph-long newsletter report based on stock price trends and news headline trends.",
  backstory = """You're a well-regarded and well-known market analyst, your newsletter is also widely considered to be trustworthy.
  You're able to understand the intricate and complex market behaviors and you're capable to create compelling stories and narratives that resonate with the wider audiences 
  that might not necessarily understand the market as well as you do.
  
  You understand macrofactors and combine multiple theories - eg. cycle theory and fundamental analysis.
  You're capable of holding multiple opinions when analyzing anything""",
  verbose = True,
  llm = gemini,
  max_iter = 5,
  memory = True,
  allow_delegation = True,
)

#==============================================================================#

# TASKS

TASK_getStockPrice = Task(
  description = "Analyze the stock {ticket} price history and create a trend analysis of up, down or sideways",
  expected_output = """ Specify the current trend - up, down or sideways.
  eg. stock='AAPL, price UP'""",
  agent = AGENT_trendsAnalyst
)

TASK_getNews = Task(
  description = f"""Take the stock and always include BTC and ETH to it (if not requested).
  Use the search tool to search each one individually
  
  The current date is {datetime.now()}
  
  Compose the results into a helpful report""",
  expected_output = """ A summary of the overall market and a single-sentenced summary for each requested asset. Include a fear/greed score for each asset based on the current news.
  Use the following format:
  <ASSET NAME>
  <SUMMARY BASED ON NEWS>
  <TREND PREDICTION>
  <FEAR/GREED INDEX>""",
  agent = AGENT_newsAnalyst
)

TASK_writeAnalysis = Task(
  description = """Use the Stock Price Trends and Market News Trends analysis reports to create
  and write a brief newsletter analyzing the {ticket} company featuring the most relevant and important points.
  Focus on the stock price trends, news and the fear/greed index score. What are the near future considerations?
  Also include the previous analysis of the stock trend and news summary.""",
  expected_output = """ An eloquent 3 paragraph newsletter formatted as markdown in an easily readable manner. It should contain:
  - 3 Bullets Executive Summary
  - Introduction - Set the overall picture and spike up the interest of the reader
  - Main Part that provides the actual bulk of the analysis including the summary of the news and the fear/greed index.
  - Summary - key facts and concrete future trends predictions - up, down or sideways.
  """,
  agent = AGENT_writerAnalyst,
  context = [TASK_getStockPrice, TASK_getNews]
)

#==============================================================================#

# CREW

crew = Crew(
  agents = [AGENT_trendsAnalyst, AGENT_newsAnalyst, AGENT_writerAnalyst],
  tasks = [TASK_getStockPrice, TASK_getNews, TASK_writeAnalysis],
  verbose = 2,
  process = Process.hierarchical,
  full_output = True,
  share_crew = False,
  manager_llm = gemini,
  max_iter = 15
)

results = crew.kickoff(inputs={'ticket': 'AAPL'})