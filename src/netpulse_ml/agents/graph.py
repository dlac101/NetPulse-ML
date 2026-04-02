"""LangGraph StateGraph for the remediation agent workflow.

Graph: ANALYZE -> DIAGNOSE -> PLAN -> (EXECUTE | ESCALATE) -> VERIFY
Deterministic (no LLM calls). LLM intelligence added in Phase 3.
"""

from functools import partial

from langgraph.graph import END, StateGraph

from netpulse_ml.agents.nodes import (
    analyze_node,
    diagnose_node,
    escalate_node,
    execute_node,
    plan_node,
    verify_node,
)
from netpulse_ml.agents.state import RemediationState
from netpulse_ml.config import settings
from netpulse_ml.serving.predictor import Predictor


def _route_after_diagnose(state: RemediationState) -> str:
    """Route after diagnosis: continue to plan if actionable, else end."""
    if state.get("diagnosis", "none") == "none":
        return "end"
    if state.get("status") == "failed":
        return "end"
    return "plan"


def _route_after_plan(state: RemediationState) -> str:
    """Route after planning: execute if auto-executable + high confidence, else escalate."""
    if state.get("recommended_action") is None:
        return "end"

    auto = state.get("auto_executable", False)
    confidence = state.get("action_confidence", 0.0)
    auto_enabled = settings.agent_enable_auto_execute

    if auto and auto_enabled and confidence >= 0.8:
        return "execute"
    return "escalate"


def build_remediation_graph(predictor: Predictor) -> StateGraph:
    """Build and compile the remediation agent graph.

    Args:
        predictor: The model predictor singleton (passed to nodes that need ML inference).

    Returns:
        Compiled LangGraph that can be invoked with RemediationState.
    """
    graph = StateGraph(RemediationState)

    # Bind predictor to nodes that need it
    analyze = partial(analyze_node, predictor=predictor)
    verify = partial(verify_node, predictor=predictor)

    # Add nodes
    graph.add_node("analyze", analyze)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("verify", verify)

    # Add edges
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "diagnose")
    graph.add_conditional_edges(
        "diagnose",
        _route_after_diagnose,
        {"plan": "plan", "end": END},
    )
    graph.add_conditional_edges(
        "plan",
        _route_after_plan,
        {"execute": "execute", "escalate": "escalate", "end": END},
    )
    graph.add_edge("execute", "verify")
    graph.add_edge("escalate", END)
    graph.add_edge("verify", END)

    return graph.compile()
