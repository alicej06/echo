"""
BUAD 304 Paper Agent
--------------------
Main agent spawns 4 subagents in parallel, each writing one section
of a 10-page OB paper on organizational culture at Starbucks vs Dutch Bros.
Results are synthesized into a full paper saved as output/buad304_paper.txt

Run:
    python3 experiments/buad304_agent.py
"""

import anthropic
import concurrent.futures
import os

client = anthropic.Anthropic()
MODEL = "claude-opus-4-6"  # best quality for academic writing

RESEARCH_QUESTION = (
    "How do different organizational cultures within the coffee industry "
    "affect overall employee performance and engagement at Starbucks and Dutch Bros?"
)

CONTEXT = """
You are writing a section of a 10-page academic paper for a BUAD 304 (Organizational Behavior)
class at USC. The paper uses an Issue Analysis lens.

Research Question: How do different organizational cultures within the coffee industry affect
overall employee performance and engagement at Starbucks and Dutch Bros?

Key facts to draw on:
- Starbucks: centralized matrix structure, calls employees "partners", recent unionization wave
  (400+ stores unionized by 2024), Glassdoor ~3.7/5, high turnover, Brian Niccol as new CEO
- Dutch Bros: centralized ownership model, 56% turnover rate (below industry average of ~70-80%),
  4/5 Glassdoor rating, "Leadership Pathway" promotion program, high-energy culture, founded on
  brotherhood/community values
- Industry average turnover: ~70-80% for hourly coffee workers

OB concepts to weave in (use these theories specifically):
- Herzberg's Two-Factor Theory (hygiene vs motivator factors)
- Job Characteristics Model (skill variety, task identity, task significance, autonomy, feedback)
- Goal-Setting Theory (SMART goals, accountability)
- Contingency Approach to management
- Structural vs Psychological Empowerment
- Organizational culture frameworks (values, norms, artifacts)
- Motivation theories (intrinsic vs extrinsic)
- Employee engagement and performance literature

Paper format requirements:
- Double-spaced, academic tone
- Use specific data points and examples
- Apply OB theories explicitly by name
- Graded on: quality of analysis, appropriate use of course concepts, logic of conclusions
"""


# ── SUBAGENTS ─────────────────────────────────────────────────────────────────

def write_section(section_name: str, instructions: str) -> tuple[str, str]:
    """Each subagent writes one full section of the paper."""
    print(f"  [subagent] Starting: {section_name}...")

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=f"""{CONTEXT}

You are writing ONLY the following section. Write it fully, with appropriate
subheadings, specific examples, and explicit OB theory application.
Write in clear academic prose. Do not include any other sections.""",
        messages=[{
            "role": "user",
            "content": f"Write the following section of the paper:\n\n{instructions}"
        }],
    )

    result = response.content[0].text
    print(f"  [subagent] Done:     {section_name}")
    return section_name, result


# ── MAIN AGENT ────────────────────────────────────────────────────────────────

def run_paper_agent():
    print("\n" + "="*65)
    print("BUAD 304 PAPER AGENT")
    print(f"Question: {RESEARCH_QUESTION}")
    print("="*65)

    # ── Define the 4 sections each subagent will write ──
    sections = [
        (
            "Introduction & Company Overview",
            """Write the Introduction (p.2) and Summary of Management Topic (p.3).

            Introduction should:
            - Introduce Starbucks and Dutch Bros as subjects
            - State the research question clearly
            - Explain the Issue Analysis lens being used
            - Briefly mention the OB frameworks that will be applied
            - Mention data sources (Glassdoor, annual reports, news articles)

            Summary of Management Topic should:
            - Detail the issue of organizational culture and its effect on performance
            - Introduce key facts about each company (structure, turnover, ratings)
            - Set the stage for the analysis without yet explaining WHY things are happening
            - Keep to facts only in this section

            Total length: ~2 pages (double-spaced)"""
        ),
        (
            "Analysis — Organizational Culture & Employee Motivation",
            """Write Analysis Part 1 (first half of p.4-6): focused on organizational culture
            and motivation theories.

            Cover:
            - How Starbucks' "partner" culture and values play out on the floor vs. on paper
              (apply Herzberg's Two-Factor Theory — what are the hygiene factors and motivators
              at each company?)
            - Dutch Bros' high-energy, brotherhood-based culture and how it drives engagement
              (apply the Job Characteristics Model — score each company on the 5 dimensions)
            - Use specific examples: Starbucks unionization as evidence of unmet hygiene factors,
              Dutch Bros Leadership Pathway as a motivator factor
            - Apply intrinsic vs extrinsic motivation frameworks

            Total length: ~1.5 pages (double-spaced)"""
        ),
        (
            "Analysis — Management Structure & Performance",
            """Write Analysis Part 2 (second half of p.4-6): focused on management structures
            and their effect on performance.

            Cover:
            - Starbucks' centralized matrix structure: how does it affect accountability and
              employee autonomy? Apply Goal-Setting Theory.
            - Dutch Bros' centralized ownership with decentralized culture: apply the
              Contingency Approach — why does this work for Dutch Bros but not Starbucks?
            - Compare turnover rates (Starbucks high, Dutch Bros 56%) as performance outcomes
              and explain them through structural differences
            - Structural vs. Psychological Empowerment: which does each company use and
              how effective is it?
            - Glassdoor ratings as a proxy for engagement data

            Total length: ~1.5 pages (double-spaced)"""
        ),
        (
            "Recommendations & Conclusion",
            """Write the Recommendations/Lessons Learned (p.7-9) and Conclusion (p.10).

            Recommendations section should:
            - Give 3-4 specific, actionable recommendations (mix of short-term and long-term)
            - Ground each recommendation in OB theory (e.g., "Starbucks should implement
              X because Herzberg's Two-Factor Theory shows Y...")
            - Focus primarily on Starbucks since it has more room for improvement
            - Give 1-2 recommendations for Dutch Bros on how to scale its culture
            - Present in a clear format (numbered list with explanation)

            Conclusion should:
            - Summarize key findings about how culture drives performance differently
              at each company
            - Note limitations of the study (e.g., relying on public data, no primary interviews)
            - End with a key takeaway about what managers in the coffee industry can learn

            Total length: ~2.5 pages (double-spaced)"""
        ),
    ]

    # ── Spawn all 4 subagents IN PARALLEL ──
    print(f"\n[main agent] Spawning {len(sections)} subagents in parallel...\n")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(write_section, name, instructions): name
            for name, instructions in sections
        }
        for future in concurrent.futures.as_completed(futures):
            section_name, content = future.result()
            results[section_name] = content

    # ── Main agent writes the Executive Summary last (needs full picture) ──
    print("\n[main agent] All sections complete. Writing Executive Summary...")

    full_draft = "\n\n".join([
        results["Introduction & Company Overview"],
        results["Analysis — Organizational Culture & Employee Motivation"],
        results["Analysis — Management Structure & Performance"],
        results["Recommendations & Conclusion"],
    ])

    exec_summary_response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=CONTEXT,
        messages=[{
            "role": "user",
            "content": f"""Based on the full paper below, write a 1-page Executive Summary.

It should be a standalone document covering:
- What was studied and why
- Key findings (bullet points)
- Key recommendations (bullet points)
- Main takeaway

Use bullet points and bold headers for readability. Max 1 page.

FULL PAPER:
{full_draft}"""
        }],
    )

    exec_summary = exec_summary_response.content[0].text

    # ── Assemble the full paper ──
    full_paper = f"""COFFEE CHAIN INDUSTRY – LABOR & MANAGEMENT ANALYSIS
Starbucks and Dutch Bros: Organizational Culture & Employee Performance

Team ReFrame | Section #20261
Kaitlyn Lee, Sydney Madolora, Charlotte Kwan, Jack Hamel, Joel Martin, Sias Bruinette

Research Question: {RESEARCH_QUESTION}

{"="*65}
EXECUTIVE SUMMARY (p.1)
{"="*65}

{exec_summary}

{"="*65}
INTRODUCTION & COMPANY OVERVIEW (p.2-3)
{"="*65}

{results["Introduction & Company Overview"]}

{"="*65}
ANALYSIS — ORGANIZATIONAL CULTURE & EMPLOYEE MOTIVATION (p.4-5)
{"="*65}

{results["Analysis — Organizational Culture & Employee Motivation"]}

{"="*65}
ANALYSIS — MANAGEMENT STRUCTURE & PERFORMANCE (p.5-6)
{"="*65}

{results["Analysis — Management Structure & Performance"]}

{"="*65}
RECOMMENDATIONS & CONCLUSION (p.7-10)
{"="*65}

{results["Recommendations & Conclusion"]}
"""

    # ── Save to file ──
    os.makedirs("experiments/output", exist_ok=True)
    output_path = "experiments/output/buad304_paper.txt"
    with open(output_path, "w") as f:
        f.write(full_paper)

    print(f"\n{'='*65}")
    print(f"DONE. Paper saved to: {output_path}")
    print(f"{'='*65}\n")

    return full_paper


if __name__ == "__main__":
    run_paper_agent()
