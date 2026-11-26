from parallel_multi_agent.main_graph import flow
from parallel_multi_agent.worker_graph import worker_graph

graph = flow.compile(name="gaia-agent")
worker = worker_graph
