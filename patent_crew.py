import re
import os
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, model_validator
from typing import Any

import requests

# Use CrewAI and import from crewai.tools
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import BaseTool  # Use CrewAI's own tool system
from langchain_ollama import OllamaLLM

from opensearch_client import get_opensearch_client


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def check_ollama_availability():
    """Check if Ollama is running and return available models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model.get("name") for model in models if model.get("name")]
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return []


def test_model(model_name: str) -> bool:
    """Test if the Ollama model is working properly."""
    try:
        llm = OllamaLLM(
            model=model_name,
            temperature=0.2,
            base_url="http://localhost:11434",
        )
        response = llm.invoke("Reply with the single word: OK")
        return bool(response and str(response).strip())
    except Exception as e:
        print(f"Error testing model '{model_name}': {e}")
        return False


def strip_think_tags(text: str) -> str:
    """
    Remove <think>...</think> blocks produced by DeepSeek-R1 and similar
    chain-of-thought models before CrewAI tries to parse the output.
    """
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Arg unwrapping helper
# ---------------------------------------------------------------------------

def _unwrap(value: Any, expected_type: type) -> Any:
    """
    llama3 (and other small models) sometimes emit tool arguments as metadata
    dicts instead of raw values, e.g.:
        {"description": "Lithium Battery", "type": "str"}  → "Lithium Battery"
        {"YYYY-MM-DD": "2022-12-31"}                       → "2022-12-31"
        {"int": 100}                                       → 100

    This helper detects and unwraps those patterns so Pydantic validation
    never sees the malformed structure.
    """
    if not isinstance(value, dict):
        return value  # already the right type — pass through

    if expected_type is str:
        # Pattern: {"description": "...", "type": "str"}
        if "description" in value:
            return str(value["description"])
        # Pattern: {"YYYY-MM-DD": "2022-12-31"}
        for v in value.values():
            if isinstance(v, str):
                return v
        return str(next(iter(value.values()), ""))

    if expected_type is int:
        # Pattern: {"int": 100}
        if "int" in value:
            return int(value["int"])
        for v in value.values():
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
        return 20  # safe default

    return value


# ---------------------------------------------------------------------------
# Tool schemas  (model_validator unwraps malformed LLM output at schema level)
# ---------------------------------------------------------------------------

class SearchPatentsToolSchema(BaseModel):
    query: str = Field(..., description="Patent search query")
    top_k: int = Field(20, description="Number of patents to retrieve")

    @model_validator(mode="before")
    @classmethod
    def unwrap_llm_args(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "query" in data:
                data["query"] = _unwrap(data["query"], str)
            if "top_k" in data:
                data["top_k"] = _unwrap(data["top_k"], int)
        return data


class SearchPatentsByDateRangeToolSchema(BaseModel):
    query: str = Field(..., description="Patent search query")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    top_k: int = Field(30, description="Number of patents to retrieve")

    @model_validator(mode="before")
    @classmethod
    def unwrap_llm_args(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "query" in data:
                data["query"] = _unwrap(data["query"], str)
            if "start_date" in data:
                data["start_date"] = _unwrap(data["start_date"], str)
            if "end_date" in data:
                data["end_date"] = _unwrap(data["end_date"], str)
            if "top_k" in data:
                data["top_k"] = _unwrap(data["top_k"], int)
        return data


class AnalyzePatentTrendsToolSchema(BaseModel):
    patents_data: str = Field(..., description="Patent data text to analyze for trends")

    @model_validator(mode="before")
    @classmethod
    def unwrap_llm_args(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "patents_data" in data:
                data["patents_data"] = _unwrap(data["patents_data"], str)
        return data


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

# FIX 1: Truncate tool output to avoid overwhelming small LLMs.
# llama3 has a ~4k-8k context window. Returning 30 full abstracts at 200 chars
# each plus titles/dates easily exceeds what the model can reliably process
# and still produce a well-formed "Final Answer". We cap each abstract at 120
# chars and limit total output to MAX_TOOL_OUTPUT_CHARS characters.
MAX_TOOL_OUTPUT_CHARS = 3000
ABSTRACT_PREVIEW_CHARS = 120


def _format_patent_results(results: list) -> str:
    """Format OpenSearch hits into a compact, LLM-friendly string."""
    formatted = []
    for i, hit in enumerate(results):
        src = hit["_source"]
        abstract = src.get("abstract", "N/A")
        if len(abstract) > ABSTRACT_PREVIEW_CHARS:
            abstract = abstract[:ABSTRACT_PREVIEW_CHARS] + "..."
        formatted.append(
            f"{i+1}. Title: {src.get('title', 'N/A')}\n"
            f"   Date: {src.get('publication_date', 'N/A')}\n"
            f"   Patent ID: {src.get('patent_id', 'N/A')}\n"
            f"   Abstract: {abstract}\n"
        )
    output = "\n".join(formatted)
    # Hard cap so the LLM context is never blown out
    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n... (truncated for brevity)"
    return output


class SearchPatentsTool(BaseTool):
    name: str = "search_patents"
    description: str = (
        "Search for patents matching a query. "
        "Returns a list of patents with title, date, patent ID and abstract. "
        "Input: query (string), top_k (integer, optional). "
        "Example: query='lithium battery', top_k=20"
    )
    args_schema: type[BaseModel] = SearchPatentsToolSchema

    def _run(self, query: str, top_k: int = 20) -> str:
        query = _unwrap(query, str) if not isinstance(query, str) else query
        top_k = _unwrap(top_k, int) if not isinstance(top_k, int) else top_k

        try:
            client = get_opensearch_client("localhost", 9200)
            search_query = {
                "size": top_k,
                "query": {"bool": {"must": [{"match": {"abstract": query}}]}},
                "_source": ["title", "abstract", "publication_date", "patent_id"],
            }
            response = client.search(index="patents", body=search_query)
            results = response["hits"]["hits"]
            if not results:
                return "No patents found for the given query."
            return _format_patent_results(results)
        except Exception as e:
            return f"Error searching patents: {str(e)}"


class SearchPatentsByDateRangeTool(BaseTool):
    name: str = "search_patents_by_date_range"
    description: str = (
        "Search for patents published within a specific date range. "
        "Input: query (string), start_date (string YYYY-MM-DD), "
        "end_date (string YYYY-MM-DD), top_k (integer, optional). "
        "Example: query='solid state battery', start_date='2022-01-01', "
        "end_date='2024-12-31', top_k=30"
    )
    args_schema: type[BaseModel] = SearchPatentsByDateRangeToolSchema

    def _run(
        self,
        query: str,
        start_date: str,
        end_date: str,
        top_k: int = 30,
    ) -> str:
        query = _unwrap(query, str) if not isinstance(query, str) else query
        start_date = _unwrap(start_date, str) if not isinstance(start_date, str) else start_date
        end_date = _unwrap(end_date, str) if not isinstance(end_date, str) else end_date
        top_k = _unwrap(top_k, int) if not isinstance(top_k, int) else top_k

        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if not date_pattern.match(str(start_date)):
            start_date = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
        if not date_pattern.match(str(end_date)):
            end_date = datetime.now().strftime("%Y-%m-%d")

        try:
            client = get_opensearch_client("localhost", 9200)
            search_query = {
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [{"match": {"abstract": query}}],
                        "filter": [
                            {
                                "range": {
                                    "publication_date": {
                                        "gte": start_date,
                                        "lte": end_date,
                                    }
                                }
                            }
                        ],
                    }
                },
                "_source": ["title", "abstract", "publication_date", "patent_id"],
            }
            response = client.search(index="patents", body=search_query)
            results = response["hits"]["hits"]
            if not results:
                return f"No patents found between {start_date} and {end_date}."
            return _format_patent_results(results)
        except Exception as e:
            return f"Error searching patents: {str(e)}"


class AnalyzePatentTrendsTool(BaseTool):
    name: str = "analyze_patent_trends"
    description: str = (
        "Analyze trends in patent data. "
        "Pass the raw patent text and receive a trend summary. "
        "Input: patents_data (string containing patent list text). "
        "Example: patents_data='1. Title: Battery Tech...'"
    )
    args_schema: type[BaseModel] = AnalyzePatentTrendsToolSchema

    def _run(self, patents_data: str) -> str:
        patents_data = (
            _unwrap(patents_data, str)
            if not isinstance(patents_data, str)
            else patents_data
        )

        lines = patents_data.strip().splitlines()
        count = sum(1 for line in lines if line.strip().startswith(tuple("0123456789")))
        # FIX 2: Return a shorter analysis snippet so the LLM can reason over it
        preview = patents_data[:800] + ("..." if len(patents_data) > 800 else "")
        return (
            f"Trend analysis over {count} patent entries:\n"
            f"{preview}\n"
            "(Further deep analysis delegated to the Data Analyst agent.)"
        )


# ---------------------------------------------------------------------------
# Crew factory
# ---------------------------------------------------------------------------

def create_patent_analysis_crew(model_name: str = "llama3:latest") -> Crew:
    """
    Create a CrewAI crew for patent analysis using a local Ollama model.
    """
    available_models = check_ollama_availability()
    if not available_models:
        raise RuntimeError(
            "Ollama service is not available. Make sure Ollama is running "
            "('ollama serve')."
        )

    if model_name not in available_models:
        raise RuntimeError(
            f"Model '{model_name}' not found in Ollama.\n"
            f"Available models: {available_models}\n"
            f"Pull it with: ollama pull {model_name}"
        )

    if not test_model(model_name):
        raise RuntimeError(
            f"Model '{model_name}' did not respond correctly. "
            "Check 'ollama logs' for details."
        )

    print(f"✅ Model '{model_name}' found and tested successfully.")

    # FIX 3: Lower num_ctx to a safe value for llama3 (4096 tokens).
    # The default of 2048 is sometimes too small for tool-use chains; 4096
    # gives more headroom without hitting OOM on most hardware.
    llm = LLM(
        model=f"ollama/{model_name}",
        api_base="http://localhost:11434",
        temperature=0.2,
        stream=False,
        extra_headers={},          # avoid accidental header injection
        # Pass Ollama-specific options via the num_ctx key recognised by
        # LiteLLM (CrewAI's underlying router).
        additional_params={"options": {"num_ctx": 4096}},
    )

    tools = [
        SearchPatentsTool(),
        SearchPatentsByDateRangeTool(),
        AnalyzePatentTrendsTool(),
    ]

    # ------------------------------------------------------------------
    # Step callback — strips <think> blocks + guards against None output
    # ------------------------------------------------------------------
    def _clean_step_output(step_output):
        if hasattr(step_output, "output"):
            if step_output.output is None:
                # FIX 4: Prevent NoneType propagation that causes "LLM Failed"
                step_output.output = "Step produced no output. Continuing."
            elif isinstance(step_output.output, str):
                step_output.output = strip_think_tags(step_output.output)
        return step_output

    # ------------------------------------------------------------------
    # Agents
    # FIX 5: Add explicit ReAct-format system prompt snippet so llama3
    # knows it MUST emit "Final Answer:" after using a tool.
    # ------------------------------------------------------------------
    REACT_REMINDER = (
        "\n\nIMPORTANT: After you receive a tool result you MUST write:\n"
        "Thought: I now know the answer.\n"
        "Final Answer: <your complete answer here>\n"
        "Never leave a turn without a Final Answer."
    )

    research_director = Agent(
        role="Research Director",
        goal="Define the scope and plan for patent analysis of {research_area}.",
        backstory=(
            "Expert research director specialising in technology innovation."
            + REACT_REMINDER
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        memory=False,
        max_iter=3,
        max_retry_limit=2,
    )

    patent_retriever = Agent(
        role="Patent Retriever",
        goal="Retrieve relevant patents about {research_area} from the database.",
        backstory=(
            "Patent researcher with deep expertise in information retrieval."
            + REACT_REMINDER
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
        memory=False,
        # FIX 6: Increase max_iter so the agent gets more chances to emit
        # a Final Answer after a tool call before CrewAI marks it Failed.
        max_iter=8,
        max_retry_limit=5,
    )

    data_analyst = Agent(
        role="Patent Data Analyst",
        goal="Identify trends and patterns in {research_area} patent data.",
        backstory=(
            "Data scientist specialising in patent trend analysis."
            + REACT_REMINDER
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
        memory=False,
        max_iter=8,
        max_retry_limit=5,
    )

    innovation_forecaster = Agent(
        role="Innovation Forecaster",
        goal="Predict future innovations in {research_area} based on patent trends.",
        backstory=(
            "Technology forecaster with a strong track record in emerging tech."
            + REACT_REMINDER
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
        memory=False,
        max_iter=5,
        max_retry_limit=3,
    )

    # ------------------------------------------------------------------
    # Tasks
    # FIX 7: Each task description now ends with an explicit instruction
    # to write "Final Answer:" so llama3 knows when to stop iterating.
    # ------------------------------------------------------------------
    task1 = Task(
        description=(
            "Create a concise research plan for {research_area} patents:\n"
            "1. Key technology sub-areas to focus on.\n"
            "2. Relevant time period (last 3 years).\n"
            "3. Specific technological aspects to investigate.\n\n"
            "Do NOT use any tools for this task. "
            "Write your plan directly as your Final Answer."
        ),
        expected_output=(
            "A bullet-point research plan listing focus areas, "
            "time period, and key aspects to investigate."
        ),
        agent=research_director,
    )

    _end_date = datetime.now().strftime("%Y-%m-%d")
    _start_date = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    task2 = Task(
        description=(
            "Retrieve patents about {research_area} published in the last 3 years.\n\n"
            "Step 1 — Call the tool ONCE using this exact format:\n"
            "  Action: search_patents_by_date_range\n"
            f'  Action Input: {{"query": "{"{research_area}".lower()}", '
            f'"start_date": "{_start_date}", "end_date": "{_end_date}", "top_k": 10}}\n\n'
            "Step 2 — After you receive the Observation, immediately write:\n"
            "  Thought: I now have the patent list.\n"
            "  Final Answer: [summarise total patents found, list titles, "
            "group by sub-technology, note key assignees]\n\n"
            "RULES:\n"
            "- Use the tool AT MOST ONCE.\n"
            "- Always pass plain string/integer values — never dicts.\n"
            "- Always end with Final Answer."
        ),
        expected_output=(
            "A structured patent retrieval report with total count, "
            "key patents grouped by sub-technology, top assignees, "
            "and main technological categories."
        ),
        agent=patent_retriever,
        context=[task1],
    )

    task3 = Task(
        description=(
            "Analyse the patent data from the previous task to identify trends "
            "for {research_area}:\n"
            "1. Growing vs declining innovation areas.\n"
            "2. Technology evolution over time.\n"
            "3. Key companies and their focus.\n"
            "4. Emerging sub-technologies.\n\n"
            "You MAY call analyze_patent_trends once if helpful, then write:\n"
            "  Final Answer: [your full trend analysis]\n\n"
            "If no tool call is needed, write your Final Answer directly."
        ),
        expected_output=(
            "A trend analysis report covering growing/declining areas, "
            "technology timeline, company focus, and emerging sub-technologies."
        ),
        agent=data_analyst,
        context=[task2],
    )

    task4 = Task(
        description=(
            "Based on the trend analysis, forecast future innovations "
            "in {research_area} for the next 2–3 years:\n"
            "1. Likely breakthrough technologies.\n"
            "2. Recommended R&D investment areas.\n"
            "3. Companies positioned to lead.\n"
            "4. Potential disruptive technologies.\n\n"
            "Do NOT use any tools. Write your forecast directly as your Final Answer."
        ),
        expected_output=(
            "A forward-looking innovation forecast with predicted breakthroughs, "
            "R&D recommendations, leading companies, and disruptive technologies."
        ),
        agent=innovation_forecaster,
        context=[task3],
    )

    crew = Crew(
        agents=[research_director, patent_retriever, data_analyst, innovation_forecaster],
        tasks=[task1, task2, task3, task4],
        verbose=True,
        process=Process.sequential,
        cache=False,
        step_callback=_clean_step_output,
    )

    return crew


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run_patent_analysis(
    research_area: str = "Lithium Battery",
    model_name: str = "llama3:latest",
) -> str:
    """
    Run the patent analysis crew and return the result as a plain string.
    """
    try:
        crew = create_patent_analysis_crew(model_name)
        crew_output = crew.kickoff(inputs={"research_area": research_area})

        for attr in ("raw", "output", "result"):
            value = getattr(crew_output, attr, None)
            if value and isinstance(value, str) and value.strip():
                return strip_think_tags(value)

        return strip_think_tags(str(crew_output))

    except RuntimeError:
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return (
            f"Analysis failed: {str(e)}\n\n"
            f"Full traceback:\n{error_detail}\n\n"
            "Troubleshooting tips:\n"
            "1. Make sure Ollama is running:  ollama serve\n"
            "2. Pull a compatible model:      ollama pull llama3\n"
            "3. For deepseek-r1 models use at least the 7b variant for tool use.\n"
            "4. Check Ollama logs for errors: journalctl -u ollama  (Linux)\n"
            "5. Try a larger/more capable model if llama3 fails on tool calls.\n"
            "   Recommended: ollama pull llama3.1  or  ollama pull mistral"
        )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    research_area = input(
        "Enter the research area to analyse (default: Lithium Battery): "
    ).strip() or "Lithium Battery"

    model_name = input(
        "Enter the Ollama model to use (default: llama3:latest): "
    ).strip() or "llama3:latest"

    result = run_patent_analysis(research_area, model_name)

    if not isinstance(result, str):
        result = str(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"patent_analysis_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write(result)

    print(f"\nAnalysis completed and saved to {filename}")
    print("\n--- RESULT PREVIEW ---")
    print(result[:1000])