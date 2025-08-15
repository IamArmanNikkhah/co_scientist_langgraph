from __future__ import annotations

from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from typing import Optional

from .nodes.bootstrap import make_bootstrap_node
from .nodes.evolution import make_evolution_node
from .nodes.generation import make_generation_node
from .nodes.literature import make_literature_node
from .nodes.meta_review import make_meta_review_node
from .nodes.observation_aggregator import make_observation_aggregator_node
from .nodes.ranking import make_ranking_node
from .nodes.reflection import make_reflection_node
from .nodes.supervisor import make_supervisor_node
from .nodes.dispatcher import make_dispatcher_node
from .state import GraphState


def build_app(supervisor_llm: ChatOpenAI, worker_llm: Optional[ChatOpenAI] = None, max_iterations: int = 20):
    """Build the LangGraph application with explicit conditional edges based on next_task.

    If worker_llm is provided, it will be used for all non-supervisor nodes to reduce latency/cost.
    """
    graph = StateGraph(GraphState)

    # Choose LLMs
    wllm = worker_llm or supervisor_llm

    # Nodes
    graph.add_node("bootstrap", make_bootstrap_node(wllm))
    graph.add_node("supervisor", make_supervisor_node(supervisor_llm, max_iterations))
    graph.add_node("generate", make_generation_node(wllm))
    graph.add_node("literature", make_literature_node(wllm))
    graph.add_node("aggregate", make_observation_aggregator_node(wllm))
    graph.add_node("reflect", make_reflection_node(wllm))
    graph.add_node("rank", make_ranking_node(wllm))
    graph.add_node("evolve", make_evolution_node(wllm))
    graph.add_node("meta_review", make_meta_review_node(wllm))
    graph.add_node("dispatch", make_dispatcher_node(supervisor_llm, wllm))

    # Flow
    graph.set_entry_point("bootstrap")
    graph.add_edge("bootstrap", "supervisor")

    def route_next(state: GraphState) -> str:
        next_task = (state.get("next_task") or "").strip()
        mapping = {
            # Supervisor first routes to dispatcher for any actionable task
            "literature": "dispatch",
            "generate": "dispatch",
            "reflect": "dispatch",
            "rank": "dispatch",
            "evolve": "dispatch",
            "meta_review": "dispatch",
            "terminate": END,
        }
        return mapping.get(next_task, END)

    graph.add_conditional_edges("supervisor", route_next)

    # Dispatcher then forwards to the actual selected node
    def dispatch_route(state: GraphState) -> str:
        task = (state.get("next_task") or "").strip()
        mapping = {
            "literature": "literature",
            "generate": "generate",
            "reflect": "aggregate",
            "rank": "rank",
            "evolve": "evolve",
            "meta_review": "meta_review",
        }
        return mapping.get(task, "supervisor")

    graph.add_conditional_edges("dispatch", dispatch_route)
    graph.add_edge("generate", "supervisor")
    graph.add_edge("literature", "supervisor")
    graph.add_edge("aggregate", "reflect")
    graph.add_edge("reflect", "supervisor")
    graph.add_edge("rank", "supervisor")
    graph.add_edge("evolve", "supervisor")
    graph.add_edge("meta_review", "supervisor")

    return graph.compile()


