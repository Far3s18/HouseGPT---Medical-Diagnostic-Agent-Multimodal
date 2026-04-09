from functools import lru_cache
from house_gpt.agent.helpers.model_factory import get_small_model, get_large_model
from house_gpt.agent.helpers.formatter import AsteriskRemovalParser
from house_gpt.states.response import RouterResponse
from house_gpt.agent.prompts import ROUTER_PROMPT, CHARACTER_CARD_PROMPT
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


@lru_cache(maxsize=1)
def get_router_chain():
    model = get_small_model(temperature=0.2).with_structured_output(RouterResponse)

    prompt =ChatPromptTemplate.from_messages([
        ('system', ROUTER_PROMPT),
        ('placeholder', '{messages}')
    ])

    chain = prompt | model

    return chain


def get_character_response_chain(summary: str = ""):
    model = get_small_model()

    system_message = CHARACTER_CARD_PROMPT

    if summary:
        system_message += f"\n\nSummary of conversation earlier between Dr House and the user: {summary}"

    prompt = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{messages}')
    ])

    chain = prompt | model | AsteriskRemovalParser()

    return chain