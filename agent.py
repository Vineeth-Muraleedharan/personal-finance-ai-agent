"""LangChain tool-calling agent for personal finance questions."""

from __future__ import annotations

import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import ALL_TOOLS

load_dotenv()


SYSTEM_PROMPT = """You are a Personal Finance Assistant for users in India. Amounts are in rupees (₹).

Rules:
1. NEVER do financial arithmetic yourself. Always call a tool and base your answer on its output.
2. Pick the tool that matches the question:
   - emi_calculator: loan EMI questions.
   - sip_calculator: "if I invest X per month for Y years at Z%, how much will I have?"
   - required_monthly_savings: "how much should I save/invest monthly to reach a target?"
   - budget_planner: monthly budget or spending plan from an income.
   - calculator: everything else that is plain maths, such as percentages.
3. Convert units before calling a tool: 1 lakh = 100000, 1 crore = 10000000.
4. If a needed number is missing (rate, tenure, income), ask the user for it instead of guessing.
   For a savings goal with no stated return, ask for the expected return or use 0% only if the user agrees.
5. Keep answers short: state the result first, then one or two helpful details such as total interest or total invested.
6. These are estimates based on fixed returns. Remind the user in one short line that real market returns vary, but only for investment questions.
"""


def get_llm(api_key: str | None = None, model: str = "gpt-4o-mini") -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY or enter it in the app.")
    return ChatOpenAI(model=model, temperature=0, api_key=key)


def build_agent(llm: BaseChatModel | None = None, api_key: str | None = None) -> AgentExecutor:
    """Create the agent. The LLM decides which tool to call; tools do the maths."""
    llm = llm or get_llm(api_key)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        return_intermediate_steps=True,
        max_iterations=5,
        handle_parsing_errors=True,
        verbose=False,
    )


def ask(executor: AgentExecutor, question: str, history: list[tuple[str, str]] | None = None) -> dict:
    """Run one question. history is a list of (role, text) with role 'user' or 'assistant'."""
    messages = [
        HumanMessage(content=t) if role == "user" else AIMessage(content=t)
        for role, t in (history or [])
    ]
    result = executor.invoke({"input": question, "chat_history": messages})
    calls = [
        {"tool": action.tool, "inputs": action.tool_input, "output": observation}
        for action, observation in result.get("intermediate_steps", [])
    ]
    return {"answer": result["output"], "tool_calls": calls}
