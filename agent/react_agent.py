"""
MarketForge - minimal ReAct agent (Paper 2: Yao et al., ICLR 2023)

Loop:  Thought -> Action -> Observation -> Thought -> ... -> Finish[answer]

Run offline demo (no API key needed):   python react_agent.py
Run with Gemini:                         pip install google-genai
                                         export GEMINI_API_KEY=...
                                         python react_agent.py --gemini
"""
import json
import os
import re
import sys
from datetime import datetime

# ----------------------------------------------------------------------
# 1. TOOLS  (in the real system these call your FastAPI / backend APIs)
# ----------------------------------------------------------------------
INVENTORY = {
    "cotton-kurta-M": {"category": "kurta", "stock": 14,  "price": 799},
    "linen-kurta-M":  {"category": "kurta", "stock": 220, "price": 1199},
    "silk-saree":     {"category": "saree", "stock": 35,  "price": 4999},
}
FORECAST_P50_WEEK = {"cotton-kurta-M": 30, "linen-kurta-M": 25, "silk-saree": 12}
FORECAST_P90_WEEK = {"cotton-kurta-M": 42, "linen-kurta-M": 34, "silk-saree": 18}


def get_inventory(category: str) -> dict:
    """All products in a category with stock and price."""
    return {sku: v for sku, v in INVENTORY.items() if v["category"] == category}


def get_stock_risk(sku: str) -> dict:
    """Stock vs. forecast demand for one product (TFT output in the real system)."""
    if sku not in INVENTORY:
        return {"error": f"unknown sku {sku}"}
    stock = INVENTORY[sku]["stock"]
    p50, p90 = FORECAST_P50_WEEK[sku], FORECAST_P90_WEEK[sku]
    return {"sku": sku, "stock": stock, "p50_week": p50, "p90_week": p90,
            "days_of_cover": round(stock / (p50 / 7), 1),
            "at_risk": p90 > stock}


def generate_copy(sku: str, aspect: str, tag: str) -> dict:
    """Ad copy generator (Paper 3 module). Stubbed here."""
    return {"sku": sku, "copy": f"[{tag}] Feel the {aspect} of our {sku.replace('-', ' ')}!"}


def schedule_post(sku: str, copy: str) -> dict:
    """Scheduler (Paper 5 module). Needs owner approval - see guardrails."""
    return {"status": "queued", "sku": sku}


TOOLS = {
    "get_inventory":  (get_inventory,  'get_inventory[{"category": "kurta"}]'),
    "get_stock_risk": (get_stock_risk, 'get_stock_risk[{"sku": "cotton-kurta-M"}]'),
    "generate_copy":  (generate_copy,  'generate_copy[{"sku": "...", "aspect": "comfort", "tag": "OVERSTOCK"}]'),
    "schedule_post":  (schedule_post,  'schedule_post[{"sku": "...", "copy": "..."}]'),
}
NEEDS_APPROVAL = {"schedule_post"}          # guardrail: human in the loop

# ----------------------------------------------------------------------
# 2. PROMPT  (instructions + tool list + one few-shot example, as in the paper)
# ----------------------------------------------------------------------
TOOL_TEXT = "\n".join(f"- {name}: {fn.__doc__} Usage: {ex}"
                      for name, (fn, ex) in TOOLS.items())

PROMPT = f"""You are MarketForge's marketing strategist for a small business.
Solve the task by interleaving Thought, Action and Observation steps.
Thought: reason about what you know and what to do next.
Action: exactly one of
{TOOL_TEXT}
- Finish[final recommendation in plain English]
Always check stock risk before recommending a promotion. Never promote an at-risk item.
Write ONE Thought and ONE Action, then stop and wait for the Observation.

Example:
Task: Should we promote silk sarees this week?
Thought 1: I must check whether silk sarees can handle extra demand.
Action 1: get_stock_risk[{{"sku": "silk-saree"}}]
Observation 1: {{"sku": "silk-saree", "stock": 35, "p50_week": 12, "p90_week": 18, "days_of_cover": 20.4, "at_risk": false}}
Thought 2: Stock covers even high demand, so a promotion is safe.
Action 2: Finish[Promote silk sarees; 35 units cover worst-case demand of 18/week.]

Task: {{task}}
"""

ACTION_RE = re.compile(r"Action\s*\d*:\s*(\w+)\[(.*)\]\s*$", re.S)


# ----------------------------------------------------------------------
# 3. LLM BACKENDS
# ----------------------------------------------------------------------
class GeminiLLM:
    """Real LLM. Check the current Gemini model name in Google's docs."""
    def __init__(self, model="gemini-2.5-flash"):
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model
        # stop before the model invents its own Observation
        self.config = types.GenerateContentConfig(stop_sequences=["\nObservation"], temperature=0)

    def __call__(self, prompt: str) -> str:
        r = self.client.models.generate_content(model=self.model, contents=prompt, config=self.config)
        return r.text


class ScriptedLLM:
    """Offline stand-in that replays a plausible trajectory, for testing the loop."""
    def __init__(self):
        self.steps = iter([
            'Thought 1: The owner wants a weekend sale on kurtas. Check the cotton kurta first.\n'
            'Action 1: get_stock_risk[{"sku": "cotton-kurta-M"}]',
            'Thought 2: It is at risk (14 units vs 42 worst-case). Look for another kurta with surplus.\n'
            'Action 2: get_inventory[{"category": "kurta"}]',
            'Thought 3: linen-kurta-M has 220 units. Verify it is safe to promote.\n'
            'Action 3: get_stock_risk[{"sku": "linen-kurta-M"}]',
            'Thought 4: Plenty of cover. Generate overstock copy focused on comfort.\n'
            'Action 4: generate_copy[{"sku": "linen-kurta-M", "aspect": "comfort", "tag": "OVERSTOCK"}]',
            'Thought 5: Copy is ready; queue it for the weekend.\n'
            'Action 5: schedule_post[{"sku": "linen-kurta-M", "copy": "Feel the comfort of linen!"}]',
            'Thought 6: Done.\n'
            'Action 6: Finish[Skip cotton-kurta-M (only 14 units). Promote linen-kurta-M (220 units) '
            'with comfort-focused overstock copy; post awaiting owner approval.]',
        ])

    def __call__(self, prompt: str) -> str:
        return next(self.steps)


# ----------------------------------------------------------------------
# 4. THE ReAct LOOP
# ----------------------------------------------------------------------
def run_agent(task: str, llm, max_steps: int = 8, approve=lambda name, args: False):
    prompt = PROMPT.replace("{task}", task)
    trace = []
    for step in range(1, max_steps + 1):                 # guardrail: step limit
        output = llm(prompt).strip()
        match = ACTION_RE.search(output)
        if not match:                                     # model broke the format
            observation = "Invalid format. Write one Thought then one Action like name[{json}]."
            prompt += f"\n{output}\nObservation {step}: {observation}"
            trace.append({"step": step, "llm": output, "observation": observation})
            continue

        name, raw_args = match.group(1), match.group(2).strip()
        if name == "Finish":
            trace.append({"step": step, "llm": output, "final": raw_args})
            return raw_args, trace

        # --- execute the tool, with guardrails ---
        if name not in TOOLS:                             # guardrail: allowed tools only
            observation = {"error": f"unknown tool {name}"}
        else:
            try:
                args = json.loads(raw_args) if raw_args else {}
                if name in NEEDS_APPROVAL and not approve(name, args):
                    observation = {"status": "pending_owner_approval", "args": args}
                else:
                    observation = TOOLS[name][0](**args)
            except Exception as e:                         # bad JSON / bad arguments
                observation = {"error": str(e)}

        obs_text = json.dumps(observation)
        prompt += f"\n{output}\nObservation {step}: {obs_text}"     # append to context
        trace.append({"step": step, "llm": output, "action": name, "observation": observation})

    return "Stopped: step limit reached without Finish.", trace


# ----------------------------------------------------------------------
if __name__ == "__main__":
    llm = GeminiLLM() if "--gemini" in sys.argv else ScriptedLLM()
    answer, trace = run_agent("Plan a weekend sale for our kurtas.", llm)
    for t in trace:
        print(t["llm"])
        if "observation" in t:
            print(f"Observation {t['step']}: {json.dumps(t['observation'])}")
        print()
    print("FINAL:", answer)
    with open("trace.json", "w") as f:                    # in production: save to MongoDB
        json.dump({"time": datetime.now().isoformat(), "trace": trace}, f, indent=2)
