"""Offline test of the agent loop using a scripted fake LLM (no API key needed).

It proves the wiring: model emits a tool call -> AgentExecutor runs the real
tool -> model receives the result and writes the final answer.
"""

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from agent import ask, build_agent  # noqa: E402


class ScriptedChat(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any):
        return self


def test_agent_calls_emi_tool_then_answers():
    llm = ScriptedChat(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "emi_calculator",
                        "args": {
                            "loan_amount": 1_000_000,
                            "annual_interest_rate": 9,
                            "tenure_years": 20,
                        },
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="Your EMI is about ₹8,997.26 per month."),
        ]
    )
    res = ask(build_agent(llm=llm), "EMI for 10 lakh at 9% for 20 years?")
    assert res["tool_calls"][0]["tool"] == "emi_calculator"
    assert "8,997.26" in res["tool_calls"][0]["output"]
    assert "8,997.26" in res["answer"]
