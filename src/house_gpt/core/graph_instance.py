from typing import Optional

compiled_graph: Optional[object] = None
_db_pool = None


def set_graph(graph):
    global compiled_graph
    compiled_graph = graph


def get_graph():
    return compiled_graph


def set_pool(pool):
    global _db_pool
    _db_pool = pool


def get_pool():
    return _db_pool