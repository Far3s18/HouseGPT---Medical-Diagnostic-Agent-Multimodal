import pytest
from langchain_core.messages import HumanMessage
from house_gpt.agent.graph.graph import get_graph_builder

@pytest.mark.asyncio
async def test_graph():
    app = get_graph_builder()
    response = await app.ainvoke({"user_id": "001", "messages": [HumanMessage(content="What is the mechanism of action of vancomycin and why is it ineffective against gram-negative bacteria?")]})
    print(response)
    assert response["messages"][-1].content != ""