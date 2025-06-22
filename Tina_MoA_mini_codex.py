#!/usr/bin/env python3
# Tina MoA Minimal Core – Codex-compatible version
import os, json, re, asyncio, subprocess
from typing import Tuple, List
import sys

# Auto-install together if needed
try:
    import together
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "together"], check=True)
    import together

# --- CONFIG ---
API_KEY = "92d5979f5ffc69d344a37cc7b2cbef622b6d51b9a24f984d97a68ece985384fb"
client = together.AsyncClient(api_key=API_KEY)

planner_MODELS = ["mistralai/Mixtral-8x7B-Instruct-v0.1"]
AGGREGATOR_MODEL = "deepseek-ai/DeepSeek-V3"
CRITIC_MODELS = ["Qwen/Qwen3-235B-A22B-fp8-tput"]
REACT_MODEL = "deepseek-ai/DeepSeek-V3"

PLANNER_PROMPT = "You are the planning agent. Given a GOAL, give the best solution for the situation."
AGGREGATOR_PROMPT = "Merge the following candidate sub-tasks into ONE concise, actionable sub-task."
CRITIC_PROMPT = 'Rate the sub-task from 0-10. Return ONLY valid JSON: {"score":N,"reason":"short reason"}'
REACT_PROMPT = """You are the expert executor. Given a sub-task, decide what to do and output a valid JSON.
Only return valid JSON. No explanations, no markdown.
If it requires bash execution: {"action": "execute_bash", "bash": "your_bash_command_here"}
If it is done: {"action": "task_complete", "result": "short summary"}
If reasoning only: {"action": "think", "thought": "your reasoning"}"""

# --- LLM CALL ---
async def llm(model: str, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def parse_score(text: str) -> Tuple[int, str]:
    try:
        match = re.search(r"\{[^}]*\"score\"[^}]*\}", text)
        if match:
            obj = json.loads(match.group(0))
            return int(obj.get("score", -1)), str(obj.get("reason", ""))
    except Exception:
        pass
    return -1, "parse failed"

async def execute_subtask(plan: str) -> bool:
    print(f"[EXEC] {plan}")
    try:
        reaction = await llm(REACT_MODEL, REACT_PROMPT, plan, max_tokens=3000)
        print("[REACT]", reaction)
        match = re.search(r"\{.*?\}", reaction, re.DOTALL)
        if not match:
            print("❌ No valid JSON found.")
            return False
        action_json = json.loads(match.group(0))
        if action_json["action"] == "execute_bash":
            print(f"🧪 Running bash: {action_json['bash']}")
            result = subprocess.run(action_json["bash"], shell=True, capture_output=True, text=True, executable="/bin/bash")
            print(result.stdout)
            if result.returncode != 0:
                print("❌ Error:", result.stderr)
                return False
            return True
        elif action_json["action"] == "task_complete":
            print("✅", action_json["result"])
            return True
        elif action_json["action"] == "think":
            print("🤔", action_json["thought"])
            return False
    except Exception as e:
        print("💥 Exception during execution:", e)
        return False

# --- MAIN ---
async def main() -> None:
    goal = input("Enter your GOAL: ").strip()
    validated = False
    while not validated:
        ideas = await asyncio.gather(
            *[llm(m, PLANNER_PROMPT, f"GOAL: {goal}") for m in planner_MODELS]
        )
        subtask = await llm(AGGREGATOR_MODEL, AGGREGATOR_PROMPT, "\n".join(ideas))
        print(f"\n[SUB-TASK]\n{subtask}\n")
        critic_outputs = await asyncio.gather(*[llm(m, CRITIC_PROMPT, subtask, temperature=0) for m in CRITIC_MODELS])
        scores = [parse_score(o)[0] for o in critic_outputs if parse_score(o)[0] >= 0]
        if not scores:
            print("All critics failed. Regenerating...\n")
            continue
        avg_score = sum(scores) / len(scores)
        print(f"[CRITIC SCORES] {scores} | average = {avg_score:.2f}")
        if avg_score < 7:
            print("Plan rejected. Regenerating...\n")
            continue
        validated = await execute_subtask(subtask)
        if not validated:
            print("Execution failed. Regenerating...\n")
    print("🎉 Task completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
