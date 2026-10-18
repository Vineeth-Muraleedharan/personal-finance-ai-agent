# Personal Finance AI Agent (LangChain)

An AI assistant that answers personal finance questions by **choosing and calling tools**. The LLM (GPT-4o-mini) decides which tool fits the question; the tools do all the arithmetic, so numbers are exact and never guessed by the model.


## Live app : https://personal-finance-ai-agent-by-vineeth.streamlit.app/

## Tools

| Tool | Purpose | Formula |
|---|---|---|
| `calculator` | General maths, percentages (safe parser, no `eval`) | AST-based evaluation |
| `emi_calculator` | Monthly loan EMI, total interest | EMI = P r (1+r)^n / ((1+r)^n - 1) |
| `sip_calculator` | Future value of a monthly investment | FV = P ((1+r)^n - 1) / r |
| `required_monthly_savings` | Monthly amount needed to reach a goal | P = FV r / ((1+r)^n - 1) |
| `budget_planner` | 50-30-20 budget split | 50% needs, 30% wants, 20% savings |

`r` = annual rate / 12 / 100 and `n` = years x 12. The fourth tool is an addition to the required list, needed for queries like "how much should I save monthly to reach ₹20 lakh in 10 years".

**SIP timing assumption:** the default follows the assignment formula (payment at the end of each month). Passing `start_of_month=True` multiplies the result by (1+r), which matches calculators that assume payment at the start of the month. For example, ₹5000/month for 10 years at 12% gives ₹11,50,193 (end of month) or ₹11,61,695 (start of month).

## Project structure

```
tools.py                  tool logic + LangChain @tool wrappers
agent.py                  tool-calling agent (LangChain AgentExecutor)
app.py                    Streamlit chat UI (shows which tool was used)
demo_queries.py           runs 8 test queries and writes demo_results.md
tests/test_tools.py       unit tests for every formula
tests/test_agent_offline.py   agent loop test with a scripted fake LLM
```

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...          # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
pytest                                 # 14 tests, no API key needed
python demo_queries.py                 # runs the demo queries, saves demo_results.md
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. On share.streamlit.io choose the repo and `app.py` as the main file.
3. Under Advanced settings, Secrets, add: `OPENAI_API_KEY = "sk-..."`. Users can also paste their own key in the sidebar.

## Test queries

1. Calculate EMI for ₹10 lakh loan at 8% interest for 20 years
2. If I invest ₹7000 monthly for 15 years at 10% return
3. Create a budget plan for income ₹90,000
4. What is 20% of ₹85,000
5. How much will I get if I invest ₹3000 per month for 5 years at 8% return
6. How much should I save monthly to reach ₹20 lakh in 10 years at 12% return?

Results with the tool used for each are in `demo_results.md` after running `demo_queries.py`.

## Limitations

Returns are fixed-rate estimates. Real market returns, inflation, taxes and fees are not modelled. This is a learning project, not financial advice.
