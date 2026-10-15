"""Run the assignment's test queries against the agent and save a report.

Usage:  OPENAI_API_KEY=sk-... python demo_queries.py
Output: demo_results.md (queries, tool chosen, tool inputs, final answer)
"""

import json
import sys

from agent import ask, build_agent

QUERIES = [
    "Calculate EMI for ₹10 lakh loan at 8% interest for 20 years",
    "If I invest ₹7000 monthly for 15 years at 10% return, how much will I have?",
    "Create a budget plan for income ₹90,000",
    "What is 20% of ₹85,000",
    "How much will I get if I invest ₹3000 per month for 5 years at 8% return",
    # extra queries from the project description
    "If I invest ₹5000 per month for 10 years at 12% return, how much money will I have?",
    "How much should I save monthly to reach ₹20 lakh in 10 years at 12% return?",
    "Calculate EMI for a ₹10 lakh loan at 9% interest for 20 years",
]


def main() -> None:
    executor = build_agent()
    lines = ["# Demo results\n"]
    for i, q in enumerate(QUERIES, 1):
        res = ask(executor, q)
        print(f"\n[{i}] {q}")
        for c in res["tool_calls"]:
            print(f"    tool: {c['tool']}  inputs: {json.dumps(c['inputs'], ensure_ascii=False)}")
        print(f"    answer: {res['answer']}")
        lines.append(f"## {i}. {q}\n")
        for c in res["tool_calls"]:
            lines.append(f"- **Tool used:** `{c['tool']}`")
            lines.append(f"- **Inputs:** `{json.dumps(c['inputs'], ensure_ascii=False)}`")
            lines.append(f"- **Tool output:** {c['output']}")
        lines.append(f"\n**Agent answer:** {res['answer']}\n")
    with open("demo_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\nSaved demo_results.md")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        sys.exit(str(exc))
