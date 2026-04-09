from typing import Optional

compiled_graph: Optional[object] = None

def set_graph(graph):
    global compiled_graph
    compiled_graph = graph

def get_graph():
    return compiled_graph