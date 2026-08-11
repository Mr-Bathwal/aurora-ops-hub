import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

response = llm.invoke([HumanMessage(content="Say hello in one short sentence.")])
print(response.content)