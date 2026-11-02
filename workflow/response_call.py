"""Run conjecture and skeptic stages; requires LangChain, LangGraph, and OPENAI_API_KEY."""

import json
import re
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

MAX_OUTPUT_TOKENS = 200  # Per stage: at most 400 output tokens across two calls.
PROBLEM = (
    "Consider a three-vertex path A-X-D with symmetric edge weights and "
    "q(A)=q(D)=0. Investigate whether r(3,A,D) uniquely determines q(X). "
    "Identify any definitions needed to test this."
)
FORMAT_INSTRUCTIONS = (
    " Keep all fields concise to fit the output limit. Use $...$ for inline "
    "mathematics and $$ on separate lines for display equations within string "
    "fields. Do not invent missing definitions; identify them explicitly."
)


# 1. Define structured answers and the shared state passed between stages.
class Conjecture(BaseModel):
    conjecture: str = Field(description="Precise, testable mathematical claim")
    assumptions: list[str] = Field(description="Explicit mathematical assumptions")
    confidence: Literal["unverified"] = Field(description="Verification status")


class Review(BaseModel):
    concerns: list[str] = Field(description="Missing assumptions or potential weaknesses")
    recommended_test: str = Field(description="A useful test, not a claim of verification")


class ResearchState(TypedDict):
    problem: str
    conjecture: NotRequired[Conjecture]
    review: NotRequired[Review]


def format_math(text: str) -> str:
    """Normalize LaTeX math delimiters for VS Code's Markdown preview."""
    text = re.sub(
        r"\\\((.*?)\\\)", lambda match: "$" + match[1].strip() + "$", text,
        flags=re.DOTALL,
    )
    return re.sub(
        r"\\\[(.*?)\\\]", lambda match: "\n$$\n" + match[1].strip() + "\n$$\n", text,
        flags=re.DOTALL,
    )


def append_report(output_path: Path, text: str) -> None:
    with output_path.open("a", encoding="utf-8") as report:
        report.write(text)


def save_stage(response: dict, schema: type[BaseModel], title: str, output_path: Path):
    """Validate and save each response before the next stage can run."""
    raw = response["raw"]
    result = response["parsed"]
    error = None
    if raw.response_metadata.get("finish_reason") == "length":
        error = f"Response reached the {MAX_OUTPUT_TOKENS}-token limit."
    elif response["parsing_error"] is not None or not isinstance(result, schema):
        error = "Response did not match the required schema (or the model refused)."

    section = f"\n---\n\n## {title}\n\n"
    if error:
        # Preserve incomplete text/tool arguments for inspection, but stop the workflow.
        section += f"> Stage failed: {error}\n\n### Raw response\n\n```json\n"
        section += json.dumps({
            "content": raw.content,
            "tool_calls": raw.tool_calls,
            "additional_kwargs": raw.additional_kwargs,
        }, indent=2, ensure_ascii=False, default=str) + "\n```\n"
    else:
        if isinstance(result, Conjecture):
            section += "### Claim\n\n" + format_math(result.conjecture)
            section += "\n\n### Assumptions\n\n"
            section += "\n".join("- " + format_math(item) for item in result.assumptions) or "None listed."
            section += f"\n\n**Verification status:** {result.confidence}\n"
        elif isinstance(result, Review):
            section += "### Concerns\n\n"
            section += "\n".join("- " + format_math(item) for item in result.concerns) or "None listed."
            section += "\n\n### Recommended test\n\n" + format_math(result.recommended_test) + "\n"
        section += "\n### Structured result\n\n```json\n" + result.model_dump_json(indent=2) + "\n```\n"
    section += "\n### Token usage\n\n```json\n" + json.dumps(raw.usage_metadata, indent=2) + "\n```\n"
    append_report(output_path, section)
    if error:
        raise ValueError(f"{title}: {error} See {output_path}.")
    return result


def build_graph(model, output_path: Path):
    # 2. Give each agent its own output schema, retaining raw token metadata.
    ca_model = model.with_structured_output(Conjecture, include_raw=True)
    sa_model = model.with_structured_output(Review, include_raw=True)

    # 3. Generate a conjecture and save it before passing it to the skeptic.
    def conjecture_agent(state: ResearchState):
        response = ca_model.invoke([
            SystemMessage(
                "You generate testable mathematical conjectures. "
                "State assumptions and never claim a proof." + FORMAT_INSTRUCTIONS
            ),
            HumanMessage(state["problem"]),
        ])
        result = save_stage(response, Conjecture, "Stage 1: Conjecture Agent", output_path)
        return {"conjecture": result}

    # 4. Critique the conjecture and save concerns and a proposed test.
    def skeptic_agent(state: ResearchState):
        response = sa_model.invoke([
            SystemMessage(
                "You are a skeptical mathematical researcher. Identify missing "
                "assumptions, possible weaknesses, and useful numerical tests. "
                "Do not claim mathematical verification without evidence." + FORMAT_INSTRUCTIONS
            ),
            HumanMessage(
                f"Original problem: {state['problem']}\n"
                f"Conjecture: {state['conjecture'].model_dump_json()}"
            ),
        ])
        result = save_stage(response, Review, "Stage 2: Skeptic Agent", output_path)
        return {"review": result}

    # 5. Connect the two stages in order: START -> CA -> SA -> END.
    builder = StateGraph(ResearchState)
    builder.add_node("CA", conjecture_agent)
    builder.add_node("SA", skeptic_agent)
    builder.add_edge(START, "CA")
    builder.add_edge("CA", "SA")
    builder.add_edge("SA", END)
    print(builder.describe())
    return builder.compile()


def main() -> None:
    model = init_chat_model(
        "gpt-4o-mini", model_provider="openai",
        max_tokens=MAX_OUTPUT_TOKENS, max_retries=0, timeout=60,
    )
    output_path = Path(__file__).with_name("response.md")
    graph = build_graph(model, output_path)
    output_path.write_text(
        "# Mathematical research workflow\n\n## Problem\n\n" + format_math(PROBLEM)
        + f"\n\nOutput limit: {MAX_OUTPUT_TOKENS} tokens per stage; no automatic retries.\n"
        + "\n> Both stages are AI-generated. Suggested tests are not executed, and claims remain unverified.\n",
        encoding="utf-8",
    )
    try:
        graph.invoke({"problem": PROBLEM})
    except Exception:
        append_report(output_path, "\n> Workflow stopped before completion. Any completed stage is preserved above. See the terminal for the error.\n")
        raise
    append_report(output_path, "\n---\n\n**Workflow status:** Complete.\n")
    print(f"Saved both stages to {output_path}")
    print("Open that file in VS Code and press Command+Shift+V to preview it.")


if __name__ == "__main__":
    main()
