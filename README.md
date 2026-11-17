\# MarketForge



A business-aware AI platform for small-business marketing (B.Tech Project Phase I, MITS Kochi).

Existing AI marketing tools ignore stock and sales. MarketForge checks them first.



\## Four stages

Sense (demand forecast + stock risk) → Decide (ReAct agent) → Create (ad copy/images) → Deliver (RedQueen scheduler)



\## What is in this repo (Phase I)

\- `agent/react\_agent.py`: minimal ReAct agent with guardrails (step limit, tool whitelist,

&#x20; owner approval before posting). Run: `python react\_agent.py` (offline demo, no API key).

&#x20; Optional real LLM: `pip install google-genai`, set GEMINI\_API\_KEY, run with `--gemini`.

\- `prototype/index.html`: UI prototype of all five screens using simulated sample data.



\## Status (honest)

\- Prototype forecaster is a simple stand-in; TFT training is in progress.

\- Image studio and scheduler in the prototype are simulations; real versions are Phase II.

\- Not yet tested on a real shop.



\## Team

Muhammad Iyac K C, Navaneeth K Shajil, Ramanujan C, Savin Krishna S

Guide: Dr. Rajesh Cherian Roy

