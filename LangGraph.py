import os
from typing import Literal
from dotenv import load_dotenv
from Bio import Entrez
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from crewai import Crew, Agent
from openai import OpenAI
import networkx as nx
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# Define the tools for the agent to use
@tool
def search(query: str):
    """Call to search PubMed."""
    Entrez.email = "your.email@example.com"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=5)
    record = Entrez.read(handle)
    id_list = record["IdList"]
    ids = ",".join(id_list)
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
    return [handle.read()]

tools = [search]

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_model(state: MessagesState):
    messages = state['messages']
    conversation = [{"role": "system", "content": "You are a helpful assistant."}]
    for message in messages:
        conversation.append({"role": "user", "content": message.content})
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=conversation
    )
    
    reply = response.choices[0].message.content
    return {"messages": [HumanMessage(content=reply)]}

# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the last message indicates a tool call, route to the "tools" node
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # Otherwise, stop (reply to the user)
    return END

# Define a new graph
workflow = StateGraph(MessagesState)

# Define the tool node
tool_node = ToolNode(tools)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="what is the latest research on COVID-19")]},
    config={"configurable": {"thread_id": 42}}
)
print(final_state["messages"][-1].content)

# CrewAI Example
def analyze_abstract(agent, abstract):
    try:
        agent_name = agent.name
    except AttributeError:
        agent_name = "Unknown Agent"
    return f"Agent {agent_name} analyzed abstract: {abstract[:100]}..."

# Initialize agents with required fields
agent1 = Agent(
    name="Agent1", 
    task=analyze_abstract,
    role="Research Analyst",
    goal="Analyze medical abstracts for key insights",
    backstory="Agent1 has extensive experience in medical research."
)

agent2 = Agent(
    name="Agent2", 
    task=analyze_abstract,
    role="Research Analyst",
    goal="Analyze medical abstracts for key insights",
    backstory="Agent2 specializes in virology and epidemiology."
)

# Initialize the crew
crew = Crew(agents=[agent1, agent2])

# Run the crew on each abstract from PubMed data
pubmed_data = final_state["messages"][-1].content.split('\n\n')
results = []
for abstract in pubmed_data:
    for agent in crew.agents:
        results.append(analyze_abstract(agent, abstract))

for result in results:
    print(result)

# # Visualization of the LangGraph using networkx and matplotlib
# G = nx.Graph()

# # Add nodes to the graph
# for i, abstract in enumerate(pubmed_data):
#     G.add_node(i, label=abstract[:30] + "...")  # Add first 30 chars of abstract as label

# # Example: Add edges (assuming some relationships, here simply connecting sequential nodes)
# for i in range(len(pubmed_data) - 1):
#     G.add_edge(i, i + 1)

# # Plot the graph
# pos = nx.spring_layout(G)
# labels = nx.get_node_attributes(G, 'label')
# nx.draw(G, pos, labels=labels, with_labels=True, node_size=5000, node_color="skyblue", font_size=10, font_color="black", font_weight="bold")
# plt.show()

class LangGraph:
    def __init__(self):
        self.nodes = []
        
    def add_node(self, node):
        self.nodes.append(node)

class Node:
    def __init__(self, content):
        self.content = content

# Initialize the graph
graph = LangGraph()

# Split the data into sentences (or use another text processing method)
abstracts = pubmed_data

# Create nodes for each sentence
nodes = [Node(abstract) for abstract in abstracts]