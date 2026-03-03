# chain.py
from langchain_core.runnables import RunnableLambda
from translator import translate

# This turns your normal Python function into a LangChain Runnable
translator_chain = RunnableLambda(translate)