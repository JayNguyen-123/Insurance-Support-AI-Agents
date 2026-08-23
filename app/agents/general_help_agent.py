from __future__ import annotations

from app.agents.prompts import GENERAL_HELP_PROMPT
from app.agents.state import GraphState
from app.llm.client import run_llm_with_tools
from app.logging_config import get_logger
from app.tracing import trace_agent
from app.vectorstore.faq_store import query_faqs

logger = get_logger(__name__)


@trace_agent
def general_help_agent_node(state: GraphState) -> dict:
    logger.info("General help agent started")

    user_query = state.get("user_input", "")
    conversation_history = state.get("conversation_history", "")
    task = state.get("task", "General insurance support")

    faqs = query_faqs(user_query, n_results=3)

    if faqs:
        logger.info("Found %d relevant FAQs", len(faqs))
        faq_context = "\n\n".join(
            f"FAQ {i + 1} (score: {faq['distance']:.3f})\nQ: {faq['question']}\nA: {faq['answer']}"
            for i, faq in enumerate(faqs)
        )
    else:
        logger.info("No relevant FAQs found")
        faq_context = "No relevant FAQs were found."

    prompt = GENERAL_HELP_PROMPT.format(
        task=task,
        conversation_history=conversation_history,
        faq_context=faq_context,
    )

    final_answer = run_llm_with_tools(prompt)

    logger.info("General help agent completed")
    return {
        "messages": [("assistant", final_answer)],
        "retrieved_faqs": faqs,
        "conversation_history": conversation_history + f"\nGeneral Help Agent: {final_answer}",
    }
