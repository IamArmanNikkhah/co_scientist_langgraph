from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI

from .nodes.bootstrap import make_bootstrap_node
from .nodes.evolution import make_evolution_node
from .nodes.generation import make_generation_node
from .nodes.meta_review import make_meta_review_node
from .nodes.literature import make_literature_node
from .nodes.observation_aggregator import make_observation_aggregator_node
from .nodes.ranking import make_ranking_node
from .nodes.reflection import make_reflection_node
from .nodes.supervisor import make_supervisor_node
from .state import GraphState, safe_append_error


def _print_state_summary(state: GraphState) -> None:
    run_meta = state.get("run_metadata", {}) or {}
    print(
        json.dumps(
            {
                "iteration": run_meta.get("iteration_count", 0),
                "goal": state.get("research_goal", "")[:200],
                "hypotheses": len(state.get("hypotheses", [])),
                "unreviewed": sum(1 for h in state.get("hypotheses", []) if not h.get("is_reviewed")),
                "rank_pending": sum(1 for h in state.get("hypotheses", []) if h.get("is_reviewed") and not h.get("is_ranked")),
                "top_elo": run_meta.get("top_elo_score"),
                "last_task": run_meta.get("last_task"),
                "errors": state.get("errors", [])[-3:],
            },
            indent=2,
        )
    )


def _route_mapping(next_task: str) -> str:
    mapping = {
        "generate": "generate",
        # In the compiled graph, "reflect" path first calls aggregator, then reflection
        "reflect": "aggregate_then_reflect",
        "literature": "literature",
        "rank": "rank",
        "evolve": "evolve",
        "meta_review": "meta_review",
        "terminate": "terminate",
    }
    return mapping.get((next_task or "").strip(), "terminate")


def _strip_circulars(state: GraphState) -> Dict[str, Any]:
    """Return a JSON-safe shallow copy of state (removes circular refs like decision.state)."""
    out: Dict[str, Any] = dict(state)
    decision = dict(out.get("decision", {}) or {})
    if "state" in decision:
        try:
            decision.pop("state", None)
        except Exception:
            pass
        out["decision"] = decision
    return out


def _auto_generate_parameters(task: str, state: GraphState) -> Dict[str, Any]:
    """Auto-generate appropriate parameters for a given task based on current state."""
    hypotheses = state.get("hypotheses", [])
    run_meta = state.get("run_metadata", {}) or {}
    goal = state.get("research_goal", "")
    
    if task == "literature":
        return {
            "search_query": goal[:200] if goal else "research literature",
            "sources": ["pubmed", "arxiv", "scholar"],
            "max_results": 10
        }
    
    elif task == "generate":
        # Mirror supervisor logic for generation mode
        iteration = int(run_meta.get("iteration_count", 0))
        try:
            # Simple heuristic for generation mode (similar to supervisor)
            use_debate = iteration < 5 or len(hypotheses) < 10
        except Exception:
            use_debate = False
        
        params = {
            "quantity": 3,
            "focus_area": goal[:100] if goal else "research focus",
            "generation_mode": "debate" if use_debate else "standard"
        }
        if use_debate:
            params["debate_max_turns"] = 6
        return params
    
    elif task == "reflect":
        # Target unreviewed hypotheses
        unreviewed = [h.get("id") for h in hypotheses if h.get("id") and not h.get("is_reviewed")]
        return {
            "priority_hypothesis_ids": unreviewed[:5],  # Limit to first 5
            "review_depth": "standard"
        }
    
    elif task == "rank":
        # Target newly reviewed hypotheses
        newly_reviewed = [h.get("id") for h in hypotheses if h.get("id") and h.get("is_reviewed") and not h.get("is_ranked")]
        return {
            "newly_reviewed_ids": newly_reviewed
        }
    
    elif task == "evolve":
        # Target high-ELO hypotheses for evolution
        evolution_candidates = [
            h.get("id") for h in hypotheses 
            if h.get("id") and h.get("elo_rating", 1200) > 1300 and h.get("is_reviewed")
        ]
        return {
            "target_hypothesis_ids": evolution_candidates[:3],  # Limit to first 3
            "strategy": "refine"
        }
    
    elif task == "meta_review":
        total_reviews = sum(len(h.get("reviews", [])) for h in hypotheses)
        return {
            "scope": "last_3_iterations" if total_reviews > 20 else "full_history",
            "focus": "suggest_new_directions"
        }
    
    elif task == "terminate":
        # Find best hypothesis
        best_hyp = max(hypotheses, key=lambda h: h.get("elo_rating", 1200), default={}) if hypotheses else {}
        return {
            "reason": "user_requested",
            "final_hypothesis_id": best_hyp.get("id", "")
        }
    
    else:
        return {}


async def _exec_task(task: str, state: GraphState, nodes: Dict[str, Any]) -> None:
    if task == "generate":
        updates = await nodes["generate"](state)
        state.update(updates or {})
        return
    if task == "literature":
        updates = await nodes["literature"](state)
        state.update(updates or {})
        return
    if task == "aggregate_then_reflect":
        updates = await nodes["aggregate"](state)
        state.update(updates or {})
        updates = await nodes["reflect"](state)
        state.update(updates or {})
        return
    if task == "rank":
        updates = await nodes["rank"](state)
        state.update(updates or {})
        return
    if task == "evolve":
        updates = await nodes["evolve"](state)
        state.update(updates or {})
        return
    if task == "meta_review":
        updates = await nodes["meta_review"](state)
        state.update(updates or {})
        return
    if task == "terminate":
        return
    safe_append_error(state, f"Unknown task '{task}' requested.")


async def run_interactive(llm: ChatOpenAI, worker_llm: Optional[ChatOpenAI], init_state: GraphState, max_iterations: int) -> GraphState:
    wllm = worker_llm or llm

    nodes = {
        "bootstrap": make_bootstrap_node(wllm),
        "supervisor": make_supervisor_node(llm, max_iterations=max_iterations),
        "literature": make_literature_node(wllm),
        "generate": make_generation_node(wllm),
        "aggregate": make_observation_aggregator_node(wllm),
        "reflect": make_reflection_node(wllm),
        "rank": make_ranking_node(wllm),
        "evolve": make_evolution_node(wllm),
        "meta_review": make_meta_review_node(wllm),
    }

    # Bootstrap
    state = dict(init_state)
    updates = await nodes["bootstrap"](state)
    state.update(updates or {})

    print("\nInteractive Co-Scientist REPL. Type 'help' for commands.\n")
    _print_state_summary(state)

    proposed_decision: Optional[Dict[str, Any]] = None
    checkpoint_dir: Optional[str] = None

    while True:
        try:
            cmd_line = input("co-sci> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not cmd_line:
            continue

        parts = cmd_line.split(" ", 1)
        cmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if cmd in {"quit", "exit"}:
            break

        if cmd == "help":
            print(
                "\nCommands:\n"
                "  help                           Show this help.\n"
                "  show                           Show state summary and any pending decision.\n"
                "  set-goal <text>                Update research goal.\n"
                "  feedback <text>                Append feedback to meta_review_critique.\n"
                "  upload-lit <path>              Load literature text from file into state.\n"
                "  upload-chronology <path>       Load initial articles_with_reasoning_text from file.\n"
                "  set-chronology <text>          Set/replace articles_with_reasoning_text directly.\n"
                "  next                           Ask Supervisor for next decision (no execution).\n"
                "  approve                        Execute the last proposed decision (from 'next').\n"
                "  override <task> [json_params]  Execute a chosen task. Parameters auto-generated if omitted.\n"
                "  do <task>                      Quick override: execute task with auto-generated parameters.\n"
                "  auto <n>                       Auto-run: iterate next+approve n times (stops on terminate).\n"
                "  checkpoint-dir <path>          Enable checkpointing to a directory (files per iteration).\n"
                "  save                           Save a checkpoint snapshot now (requires checkpoint-dir).\n"
                "  state-json                     Print full state JSON (truncated hypotheses content).\n"
                "  quit/exit                      Exit REPL.\n"
            )
            continue

        if cmd == "show":
            _print_state_summary(state)
            if proposed_decision:
                print("Proposed decision (pending 'approve' or 'override'):")
                print(json.dumps({k: proposed_decision.get(k) for k in ["next_task", "parameters", "rationale"]}, indent=2))
            continue

        if cmd == "set-goal":
            new_goal = rest.strip()
            if not new_goal:
                print("Usage: set-goal <text>")
                continue
            state["research_goal"] = new_goal
            print("Updated goal.")
            continue

        if cmd == "feedback":
            fb = rest.strip()
            if not fb:
                print("Usage: feedback <text>")
                continue
            cur = state.get("meta_review_critique") or ""
            sep = "\n---\n" if cur else ""
            state["meta_review_critique"] = f"{cur}{sep}{fb}"
            print("Feedback appended.")
            continue

        if cmd == "upload-lit":
            path = rest.strip()
            if not path:
                print("Usage: upload-lit <path>")
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    state["literature_content"] = f.read()
                print("Literature loaded.")
            except Exception as e:
                safe_append_error(state, f"Failed to read literature file: {e}")
                print(f"Error: {e}")
            continue

        if cmd == "upload-chronology":
            path = rest.strip()
            if not path:
                print("Usage: upload-chronology <path>")
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    state["articles_with_reasoning_text"] = f.read().strip() or None
                print("Chronology text loaded.")
            except Exception as e:
                safe_append_error(state, f"Failed to read chronology file: {e}")
                print(f"Error: {e}")
            continue

        if cmd == "set-chronology":
            txt = rest.strip()
            if not txt:
                print("Usage: set-chronology <text>")
                continue
            state["articles_with_reasoning_text"] = txt
            print("Chronology text set.")
            continue

        if cmd == "state-json":
            compact_state = dict(state)
            hyps = []
            for h in compact_state.get("hypotheses", [])[:]:
                h_c = dict(h)
                c = str(h_c.get("content", ""))
                if len(c) > 400:
                    h_c["content"] = c[:400] + " ..."
                hyps.append(h_c)
            compact_state["hypotheses"] = hyps
            print(json.dumps(compact_state, indent=2))
            continue

        if cmd == "next":
            try:
                updates = await nodes["supervisor"](state)
                state.update(updates or {})
                proposed_decision = state.get("decision", {}) or {}
                if isinstance(proposed_decision, dict) and "state" in proposed_decision:
                    proposed_decision = dict(proposed_decision)
                    proposed_decision.pop("state", None)
                    state["decision"] = proposed_decision
                print("Supervisor proposed (use 'approve' to execute or 'override <task> <json>' to change):")
                print(json.dumps({k: proposed_decision.get(k) for k in ["next_task", "parameters", "rationale"]}, indent=2))
            except Exception as e:
                safe_append_error(state, f"Supervisor error: {e}")
                print(f"Supervisor error: {e}")
            continue

        if cmd == "approve":
            if not proposed_decision:
                print("No proposed decision. Run 'next' first or use 'override'.")
                continue
            task = _route_mapping(str(proposed_decision.get("next_task")))
            state["parameters"] = dict(proposed_decision.get("parameters", {}))
            await _exec_task(task, state, nodes)
            proposed_decision = None
            _print_state_summary(state)
            if checkpoint_dir:
                try:
                    import os
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    iter_id = (state.get("run_metadata", {}) or {}).get("iteration_count", 0)
                    safe_state = _strip_circulars(state)
                    with open(f"{checkpoint_dir}/iter_{iter_id}.json", "w", encoding="utf-8") as f:
                        json.dump(safe_state, f, indent=2)
                except Exception as e:
                    safe_append_error(state, f"Checkpoint save failed: {e}")
            continue

        if cmd == "do":
            task_name = rest.strip()
            if not task_name:
                print("Usage: do <task>")
                print("Available tasks: literature, generate, reflect, rank, evolve, meta_review, terminate")
                continue
            
            # Auto-generate parameters and execute immediately
            params = _auto_generate_parameters(task_name, state)
            print(f"Executing {task_name} with auto-generated parameters:")
            print(json.dumps(params, indent=2))
            
            state["parameters"] = params
            await _exec_task(_route_mapping(task_name), state, nodes)
            proposed_decision = None
            _print_state_summary(state)
            if checkpoint_dir:
                try:
                    import os
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    iter_id = (state.get("run_metadata", {}) or {}).get("iteration_count", 0)
                    safe_state = _strip_circulars(state)
                    with open(f"{checkpoint_dir}/iter_{iter_id}.json", "w", encoding="utf-8") as f:
                        json.dump(safe_state, f, indent=2)
                except Exception as e:
                    safe_append_error(state, f"Checkpoint save failed: {e}")
            continue

        if cmd == "override":
            if not rest:
                print("Usage: override <task> [json_params]")
                print("  If json_params are omitted, they will be auto-generated based on current state.")
                continue
            sub = rest.split(" ", 1)
            task_name = sub[0].strip()
            params_raw = sub[1].strip() if len(sub) > 1 else ""
            
            try:
                if params_raw:
                    # User provided explicit parameters
                    params = json.loads(params_raw)
                    print(f"Using provided parameters for {task_name}:")
                    print(json.dumps(params, indent=2))
                else:
                    # Auto-generate parameters
                    params = _auto_generate_parameters(task_name, state)
                    print(f"Auto-generated parameters for {task_name}:")
                    print(json.dumps(params, indent=2))
                    
                    # Give user a chance to confirm or modify
                    try:
                        confirm = input("Proceed with these parameters? (y/n/edit): ").strip().lower()
                        if confirm == 'n':
                            print("Override cancelled.")
                            continue
                        elif confirm == 'edit':
                            print("Enter your custom JSON parameters:")
                            try:
                                custom_params_raw = input("Parameters: ").strip()
                                params = json.loads(custom_params_raw)
                                print("Using custom parameters.")
                            except json.JSONDecodeError as e:
                                print(f"Invalid JSON: {e}. Using auto-generated parameters.")
                            except (EOFError, KeyboardInterrupt):
                                print("Edit cancelled. Using auto-generated parameters.")
                    except (EOFError, KeyboardInterrupt):
                        print("Using auto-generated parameters.")
                        
            except json.JSONDecodeError as e:
                print(f"Invalid JSON parameters: {e}")
                continue
                
            state["parameters"] = params
            await _exec_task(_route_mapping(task_name), state, nodes)
            proposed_decision = None
            _print_state_summary(state)
            if checkpoint_dir:
                try:
                    import os
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    iter_id = (state.get("run_metadata", {}) or {}).get("iteration_count", 0)
                    safe_state = _strip_circulars(state)
                    with open(f"{checkpoint_dir}/iter_{iter_id}.json", "w", encoding="utf-8") as f:
                        json.dump(safe_state, f, indent=2)
                except Exception as e:
                    safe_append_error(state, f"Checkpoint save failed: {e}")
            continue

        if cmd == "auto":
            n_str = rest.strip()
            if not n_str or not n_str.isdigit() or int(n_str) <= 0:
                print("Usage: auto <n>")
                continue
            remaining = int(n_str)
            while remaining > 0:
                try:
                    updates = await nodes["supervisor"](state)
                    state.update(updates or {})
                    proposed_decision = state.get("decision", {}) or {}
                except Exception as e:
                    safe_append_error(state, f"Supervisor error during auto: {e}")
                    print(f"Supervisor error during auto: {e}")
                    break
                next_task = str(proposed_decision.get("next_task"))
                if next_task == "terminate":
                    print("Auto: terminate proposed; stopping.")
                    break
                state["parameters"] = dict(proposed_decision.get("parameters", {}))
                await _exec_task(_route_mapping(next_task), state, nodes)
                proposed_decision = None
                if checkpoint_dir:
                    try:
                        import os
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        iter_id = (state.get("run_metadata", {}) or {}).get("iteration_count", 0)
                        safe_state = _strip_circulars(state)
                        with open(f"{checkpoint_dir}/iter_{iter_id}.json", "w", encoding="utf-8") as f:
                            json.dump(safe_state, f, indent=2)
                    except Exception as e:
                        safe_append_error(state, f"Checkpoint save failed: {e}")
                remaining -= 1
            _print_state_summary(state)
            continue

        if cmd == "checkpoint-dir":
            path = rest.strip()
            if not path:
                print("Usage: checkpoint-dir <path>")
                continue
            checkpoint_dir = path
            print(f"Checkpointing enabled: {checkpoint_dir}")
            continue

        if cmd == "save":
            if not checkpoint_dir:
                print("Checkpoint directory not set. Use 'checkpoint-dir <path>' first.")
                continue
            try:
                import os
                os.makedirs(checkpoint_dir, exist_ok=True)
                iter_id = (state.get("run_metadata", {}) or {}).get("iteration_count", 0)
                safe_state = _strip_circulars(state)
                with open(f"{checkpoint_dir}/iter_{iter_id}.json", "w", encoding="utf-8") as f:
                    json.dump(safe_state, f, indent=2)
                print(f"Saved checkpoint iter_{iter_id}.json")
            except Exception as e:
                safe_append_error(state, f"Checkpoint save failed: {e}")
                print(f"Checkpoint save failed: {e}")
            continue

        print(f"Unknown command: {cmd}. Type 'help' for commands.")

    try:
        if checkpoint_dir:
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
            safe_state = _strip_circulars(state)
            with open(f"{checkpoint_dir}/final.json", "w", encoding="utf-8") as f:
                json.dump(safe_state, f, indent=2)
    except Exception as e:
        safe_append_error(state, f"Final checkpoint save failed: {e}")
    return _strip_circulars(state)


