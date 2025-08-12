from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from .app import build_app
from .state import GraphState


def get_llm(model: str = "gpt-5", temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


def load_initial_state(args: argparse.Namespace) -> GraphState:
    state: GraphState = {
        "research_goal": "",
        "research_plan_config": {},
        "hypotheses": [],
        "meta_review": {},
        "meta_review_critique": "",
        "scientific_observations": None,
        "literature_content": None,
        "decision": {},
        "next_task": None,
        "parameters": {},
        "run_metadata": {},
        "errors": [],
    }

    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            from_file: Dict[str, Any] = json.load(f)
        state.update(from_file)
    elif args.goal:
        state["research_goal"] = args.goal

    if args.literature_file:
        try:
            with open(args.literature_file, "r", encoding="utf-8") as f:
                state["literature_content"] = f.read()
        except Exception as e:
            state["literature_content"] = None
            state["errors"].append(f"Failed to read literature file: {e}")

    return state


async def run(args: argparse.Namespace):
    llm = get_llm(model=args.model, temperature=args.temperature)
    app = build_app(llm, max_iterations=args.max_iterations)
    init_state = load_initial_state(args)
    final_state = await app.ainvoke(init_state)
    print(json.dumps(final_state, indent=2))


def main():
    parser = argparse.ArgumentParser(description="AI Co-Scientist (LangGraph)")
    parser.add_argument("--goal", type=str, help="Research goal (natural language).")
    parser.add_argument("--input-json", type=str, help="Path to JSON file with initial state.")
    parser.add_argument("--literature-file", type=str, help="Path to text file with literature content.")
    parser.add_argument("--max-iterations", type=int, default=20, help="Maximum supervisor iterations before termination.")
    parser.add_argument("--model", type=str, default="gpt-5", help="OpenAI chat model name.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature.")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()


