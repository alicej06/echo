"""
Echo Agent Demo
---------------
A main agent that spawns subagents in parallel to research topics,
then synthesizes their findings.

Run it:
    python3 experiments/agent_demo.py
"""

import anthropic
import concurrent.futures

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"  # fast + cheap for demos


# ── SUBAGENT ──────────────────────────────────────────────────────────────────
# This is the function each subagent runs.
# Each subagent is just a fresh API call with its own focused prompt.

def run_subagent(task: str) -> str:
    """Spin up a subagent to handle one specific task."""
    print(f"  [subagent starting] → {task[:60]}...")

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="You are a focused research assistant. Answer concisely and specifically. 3-5 sentences max.",
        messages=[{"role": "user", "content": task}],
    )

    result = response.content[0].text
    print(f"  [subagent done]     → {task[:60]}...")
    return result


# ── MAIN AGENT ────────────────────────────────────────────────────────────────
# The main agent decides WHAT to research, spawns subagents in parallel,
# then synthesizes everything into a final answer.

def run_agent(user_question: str) -> str:
    """Main agent: plan → delegate → synthesize."""

    print(f"\n{'='*60}")
    print(f"MAIN AGENT received: {user_question}")
    print(f"{'='*60}")

    # ── STEP 1: Main agent plans the subtasks ──
    print("\n[main agent] Planning subtasks...")

    plan_response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system="""You are a task planner. When given a question, break it into
        exactly 3 focused research subtasks. Return ONLY a numbered list like:
        1. Research X
        2. Research Y
        3. Research Z
        Nothing else.""",
        messages=[{"role": "user", "content": user_question}],
    )

    plan_text = plan_response.content[0].text
    print(f"[main agent] Subtasks:\n{plan_text}")

    # Parse the numbered list into individual tasks
    tasks = []
    for line in plan_text.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            # Remove "1. " prefix
            task = line.split(". ", 1)[-1].strip()
            tasks.append(task)

    if not tasks:
        return "Could not parse subtasks."

    # ── STEP 2: Spawn subagents IN PARALLEL ──
    print(f"\n[main agent] Spawning {len(tasks)} subagents in parallel...")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        # Submit all tasks at once — they run simultaneously
        future_to_task = {
            executor.submit(run_subagent, task): task
            for task in tasks
        }
        # Collect results as they finish
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            results[task] = future.result()

    # ── STEP 3: Main agent synthesizes everything ──
    print("\n[main agent] Synthesizing results...")

    synthesis_input = f"Original question: {user_question}\n\nResearch findings:\n"
    for task, result in results.items():
        synthesis_input += f"\n--- {task} ---\n{result}\n"

    synthesis_response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="""You are a synthesis expert. You receive research from multiple
        subagents and write a clear, organized final answer. Use headers if helpful.
        Be direct and useful.""",
        messages=[{"role": "user", "content": synthesis_input}],
    )

    final_answer = synthesis_response.content[0].text

    print("\n" + "="*60)
    print("FINAL ANSWER")
    print("="*60)
    print(final_answer)
    return final_answer


# ── RUN IT ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Try changing this question to anything you want!
    question = "What are the biggest challenges in building an ASL recognition app and how can they be solved?"

    run_agent(question)
