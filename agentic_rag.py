import os
import traceback
from datetime import datetime

import requests
from dotenv import load_dotenv

from opensearch_client import get_opensearch_client
from patent_crew import run_patent_analysis
from patent_search_tools import hybrid_search, iterative_search, semantic_search, keyword_search


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def display_menu() -> str:
    print("\n" + "=" * 60)
    print("  PATENT INNOVATION PREDICTOR - LITHIUM BATTERY TECHNOLOGY  ")
    print("=" * 60)
    print("1. Run complete patent trend analysis and forecasting")
    print("2. Search for specific patents")
    print("3. Iterative patent exploration")
    print("4. View system status")
    print("5. Exit")
    print("-" * 60)
    return input("Select an option (1-5): ").strip()


# ---------------------------------------------------------------------------
# Option 1 — Full CrewAI analysis
# ---------------------------------------------------------------------------

def run_complete_analysis() -> None:
    print("\nRunning comprehensive patent analysis...")
    print("This may take several minutes depending on the data volume.\n")

    research_area = input("Enter research area (default: Lithium Battery): ").strip()
    if not research_area:
        research_area = "Lithium Battery"

    model_name = input(
        "Enter the Ollama model to use (default: llama3:latest): "
    ).strip()
    if not model_name:
        model_name = "llama3:latest"

    print(f"\nAnalysing patents for : {research_area}")
    print(f"Using Ollama model   : {model_name}")
    print("Agents are now processing the data…\n")

    try:
        result = run_patent_analysis(research_area, model_name)
    except RuntimeError as e:
        # Configuration / environment errors — show clearly, no traceback noise
        print(f"\n❌ Configuration error:\n{e}")
        return
    except Exception as e:
        # Unexpected errors — show full traceback for debugging
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        return

    if not isinstance(result, str):
        result = str(result)

    # Guard against empty or trivially short results
    if not result.strip() or result.strip() in ("None", ""):
        print(
            "\n⚠️  The crew returned an empty result.\n"
            "   Possible causes:\n"
            "   • The model is too small for multi-step tool use "
            "(deepseek-r1:1.5b often fails). Try llama3 or mistral.\n"
            "   • Ollama ran out of memory — check 'ollama ps'.\n"
            "   • No patents matched the query in OpenSearch."
        )
        return

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"patent_analysis_{timestamp}.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"\n✅ Analysis completed and saved to: {filename}")
    except OSError as e:
        print(f"\n⚠️  Could not save file: {e}")

    # Display summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY (first 500 chars)")
    print("-" * 60)
    print(result[:500])
    if len(result) > 500:
        print(f"\n… ({len(result) - 500} more characters in the saved file)")


# ---------------------------------------------------------------------------
# Option 2 — Manual patent search
# ---------------------------------------------------------------------------

def search_patents() -> None:
    print("\nPATENT SEARCH")
    print("-" * 60)

    query = input("Enter search query: ").strip()
    if not query:
        print("Search query cannot be empty.")
        return

    search_type = input(
        "Select search type (1: Keyword, 2: Semantic, 3: Hybrid) [3]: "
    ).strip() or "3"

    try:
        if search_type == "1":
            results = keyword_search(query)
        elif search_type == "2":
            results = semantic_search(query)
        else:
            results = hybrid_search(query)
    except Exception as e:
        print(f"Search error: {e}")
        traceback.print_exc()
        return

    if not results:
        print(f"\nNo results found for '{query}'.")
        return

    print(f"\nFound {len(results)} result(s) for '{query}':")
    print("-" * 60)
    for i, hit in enumerate(results):
        source = hit.get("_source", {})
        print(f"{i+1}. {source.get('title', 'N/A')}")
        print(f"   Score      : {hit.get('_score', 'N/A')}")
        print(f"   Date       : {source.get('publication_date', 'N/A')}")
        print(f"   Patent ID  : {source.get('patent_id', 'N/A')}")
        abstract = source.get("abstract", "")
        print(f"   Abstract   : {abstract[:150]}{'...' if len(abstract) > 150 else ''}")
        print("-" * 60)


# ---------------------------------------------------------------------------
# Option 3 — Iterative exploration
# ---------------------------------------------------------------------------

def iterative_exploration() -> None:
    print("\nITERATIVE PATENT EXPLORATION")
    print("-" * 60)

    query = input("Enter initial exploration query: ").strip()
    if not query:
        print("Query cannot be empty.")
        return

    steps_raw = input("Number of exploration steps (default: 3): ").strip()
    try:
        steps = int(steps_raw) if steps_raw else 3
    except ValueError:
        steps = 3

    print(f"\nExploring patents related to '{query}' over {steps} refinement steps…")

    try:
        results = iterative_search(query, refinement_steps=steps)
    except Exception as e:
        print(f"Exploration error: {e}")
        traceback.print_exc()
        return

    if not results:
        print(f"\nNo results found for '{query}'.")
        return

    print(f"\nFound {len(results)} result(s) through iterative exploration:")
    print("-" * 60)
    for i, hit in enumerate(results):
        source = hit.get("_source", {})
        print(f"{i+1}. {source.get('title', 'N/A')}")
        print(f"   Date      : {source.get('publication_date', 'N/A')}")
        print(f"   Patent ID : {source.get('patent_id', 'N/A')}")
        abstract = source.get("abstract", "")
        print(f"   Abstract  : {abstract[:150]}{'...' if len(abstract) > 150 else ''}")
        print("-" * 60)


# ---------------------------------------------------------------------------
# Option 4 — System status
# ---------------------------------------------------------------------------

def check_system_status() -> None:
    print("\nSYSTEM STATUS")
    print("-" * 60)

    # OpenSearch
    try:
        client = get_opensearch_client("localhost", 9200)
        indices = client.cat.indices(format="json")
        print("✅ OpenSearch connection : OK")
        print(f"   Found {len(indices)} index/indices:")
        for index in indices:
            print(
                f"   - {index['index']}: "
                f"{index.get('docs.count', '?')} documents"
            )
    except Exception as e:
        print(f"❌ OpenSearch connection : FAILED — {e}")

    # Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            names = ", ".join(m.get("name", "unknown") for m in models)
            print("✅ Ollama connection     : OK")
            print(f"   Available models: {names or '(none pulled yet)'}")
        else:
            print(
                f"❌ Ollama connection     : FAILED "
                f"(HTTP {response.status_code})"
            )
    except Exception as e:
        print(f"❌ Ollama connection     : FAILED — {e}")

    # Embedding model
    try:
        from embedding import get_embedding  # type: ignore
        sample = get_embedding("test")
        print(f"✅ Embedding model       : OK (dimension: {len(sample)})")
    except Exception as e:
        print(f"❌ Embedding model       : FAILED — {e}")

    print("\nStatus check complete.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    while True:
        choice = display_menu()

        if choice == "1":
            run_complete_analysis()
        elif choice == "2":
            search_patents()
        elif choice == "3":
            iterative_exploration()
        elif choice == "4":
            check_system_status()
        elif choice == "5":
            print("\nExiting Patent Innovation Predictor. Goodbye!")
            break
        else:
            print("\nInvalid option. Please select a number between 1 and 5.")

        input("\nPress Enter to continue…")


if __name__ == "__main__":
    main()