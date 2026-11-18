from typing_extensions import TypedDict, Literal
import pandas as pd
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from pathlib import Path
import io
from IPython.display import Image, display

# Load environment variables from .env file
load_dotenv()

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------
# 1. Shared State Definition
# ---------------------------

class DataState(TypedDict):
    csv_path: str
    df: pd.DataFrame
    action: Literal["clean_missing", "remove_outliers", "both", "none"]
    summary: str
    evidence: str


# ---------------------------
# 2. Initialize LLM
# ---------------------------

llm = ChatOpenAI(model="gpt-4o-mini")


# ---------------------------
# 3. Nodes
# ---------------------------

def load_data(state: DataState) -> DataState:
    """Load CSV into DataFrame."""
    state["df"] = pd.read_csv(state["csv_path"])
    return state


def summarize_data(state: DataState) -> DataState:
    """Generate comprehensive summary including missing values."""
    parts = []
    
    # 1. Basic statistics
    parts.append("DATA DESCRIPTION:\n")
    parts.append(state["df"].describe().to_string())
    
    # 2. Dataset info
    parts.append("\n\nDATA INFO:\n")
    buf = io.StringIO()
    state["df"].info(buf=buf)
    parts.append(buf.getvalue())
    
    # 3. Explicit missing value counts
    parts.append("\n\nMISSING VALUE COUNTS:\n")
    missing_counts = state["df"].isnull().sum()
    parts.append(missing_counts.to_string())

    # 4. Outlier statistics (IQR method)
    parts.append("\n\nOUTLIER STATISTICS (IQR METHOD):\n")
    numeric_cols = state["df"].select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        parts.append("No numerical columns available for outlier analysis.")
        state["evidence"] = "No numerical columns available."
    else:
        rows = {}
        for col in numeric_cols:
            s = state["df"][col].dropna()
            n_valid = len(s)
            if n_valid == 0:
                rows[col] = {
                    "valid_obs": 0, "Q1": float("nan"), "Q3": float("nan"),
                    "IQR": float("nan"), "lower_bound": float("nan"),
                    "upper_bound": float("nan"), "outliers": 0,
                    "outliers_%": float("nan"),
                }
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_outliers = int(((s < lower) | (s > upper)).sum())
            rows[col] = {
                "valid_obs": n_valid, "Q1": q1, "Q3": q3, "IQR": iqr,
                "lower_bound": lower, "upper_bound": upper,
                "outliers": n_outliers,
                "outliers_%": round(n_outliers / n_valid * 100, 2),
            }
        parts.append(pd.DataFrame.from_dict(rows, orient="index").to_string())
        state["evidence"] = "\n".join(
            f"{col}: flagged = {r['outliers']} ({r['outliers_%']}%), "
            f"missing = {int(missing_counts[col])}"
            for col, r in rows.items()
        )

    state["summary"] = "\n".join(parts)
    return state


def reasoning_node(state: DataState) -> DataState:
    """Use LLM to decide whether to clean missing values or remove outliers or both or none."""
    prompt = (
        "You are a data science assistant that selects a data-cleaning strategy.\n"
        "Base your decision EXCLUSIVELY on the dataset summary below. Do not assume "
        "that missing values or outliers exist without supporting evidence in the summary, "
        "and do not invent statistics.\n\n"
        "STEP 1 - MISSING VALUES (evaluate independently)\n"
        "- Examine MISSING VALUE COUNTS for each column. Where useful, compute the percentage "
        "of missing values using the non-null counts in DATA INFO or the row count.\n"
        "- Distinguish between the mere presence of missing values and missing values that "
        "actually warrant treatment.\n"
        "- Missing values warrant treatment only when the evidence is meaningful: a 5 percent or more "
        "of missing values for at least one column."
        "If each of the columns have less than 5 percent of missing values, "
        "missing values are NOT JUSTIFIED.\n"
        "- Conclude: missing values are either JUSTIFIED for treatment or NOT JUSTIFIED.\n\n"
        "STEP 2 - OUTLIERS (evaluate independently)\n"
        "- Examine OUTLIER STATISTICS (IQR METHOD) for each numerical variable: valid_obs, "
        "Q1, Q3, IQR, lower_bound, upper_bound, outliers (count) and outliers_% (percentage). "
        "Also use DATA DESCRIPTION (min, max, mean, median, std) to see how far values "
        "lie beyond the IQR boundaries.\n"
        "- Potential outliers are not the same as observations that warrant removal. The IQR "
        "method flags some points in most real distributions. Do not assume every extreme "
        "numerical value is erroneous, and do NOT choose removal solely because "
        "potential outliers were detected.\n"
        "- Outliers warrant removal only when the evidence is meaningful: a 5 percent or more "
        "of flagged observations for at least one numerical column, or a single value lying very far beyond the boundaries "
        "relative to the IQR and the rest of the distribution.  "
        "If there are no numerical columns, or all columns have less than 5 percent of flagged observations, "
        "outliers are NOT JUSTIFIED.\n"
        "- Conclude: outliers are either JUSTIFIED for removal or NOT JUSTIFIED.\n\n"
        "STEP 3 - FINAL DECISION (combine the two conclusions)\n"
        "- clean_missing: missing values JUSTIFIED and outliers NOT JUSTIFIED "
        "(missing values are the only problem that warrants cleaning).\n"
        "- remove_outliers: outliers JUSTIFIED and missing values NOT JUSTIFIED "
        "(absent, negligible, or not requiring treatment; outliers are the only problem "
        "that warrants cleaning).\n"
        "- both: missing values JUSTIFIED and outliers JUSTIFIED, each independently "
        "justifying its own cleaning operation. Do not choose both unless each is justified.\n"
        "- none: missing values NOT JUSTIFIED and outliers NOT JUSTIFIED "
        "(neither cleaning operation is supported by the statistical evidence).\n"
        "If the summary lacks enough information to justify an operation, do not select it.\n\n"
        "DATASET SUMMARY\n"
        f"{state['summary']}\n\n"
        "OUTPUT FORMAT\n"
        "Respond with exactly three short lines, with no Markdown and no extra text:\n"
        "Step 1: JUSTIFIED or NOT JUSTIFIED (missing values)\n"
        "Step 2: JUSTIFIED or NOT JUSTIFIED (outliers)\n"
        "Final: one of these four words, in lowercase: clean_missing, remove_outliers, both, none"
    )
    response = llm.invoke(prompt).content.strip().lower()
    last_line = response.splitlines()[-1] if response else ""
    decision = last_line.split("final:")[-1].strip()
#    if decision not in ["clean_missing", "remove_outliers", "both"]:
#        decision = "none"
    state["action"] = decision
    return state


def handle_missing_values(state: DataState) -> DataState:
    """Fill missing numeric values with the column mean."""
    df = state["df"].copy()
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].mean())
    state["df"] = df
    return state


def remove_outliers(state: DataState) -> DataState:
    """Remove outliers using IQR method."""
    df = state["df"].copy()
    numeric_cols = df.select_dtypes(include="number").columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    state["df"] = df
    return state

def both(state: DataState) -> DataState:#+++++++++++++++++++++++++++++++++++++++++++++++++
    """Fill missing numeric values with the column mean and Remove outliers using IQR method."""
    df = state["df"].copy()
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].mean())
    
    numeric_cols = df.select_dtypes(include="number").columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    state["df"] = df
    return state

def none(state: DataState) -> DataState:#+++++++++++++++++++++++++++++++++++++++++++++++++
    """Do nothing: neither fill missing numeric values nor remove outliers."""
    #df = state["df"].copy()
 
    #state["df"] = df
    return state


def describe_data(state: DataState) -> DataState:
    """Describe numeric columns after any cleaning."""
    state["summary"] = state["df"].describe().to_string()
    return state


def output_results(state: DataState):
    print(f"\n=== ACTION DECIDED: {state['action'].upper()} ===\n")
    print("EVIDENCE (per numerical column):")
    print(state["evidence"] + "\n")
    print(state["summary"])


# ---------------------------
# 4. Router Function
# ---------------------------

def route_action(state: DataState) -> str:
    """Route based on LLM's chosen action."""
    mapping = {
        "clean_missing": "handle_missing_values",
        "remove_outliers": "remove_outliers",
        "both": "both",#+++++++++++++++++++++++++++++++++++++++++++++++++++
        "none": "none",#"describe_data",
    }
    return mapping.get(state["action"], "describe_data")


# ---------------------------
# 5. Build Graph
# ---------------------------

workflow = StateGraph(DataState)

workflow.add_node("load_data", load_data)
workflow.add_node("summarize_data", summarize_data)
workflow.add_node("reasoning_node", reasoning_node)
workflow.add_node("handle_missing_values", handle_missing_values)
workflow.add_node("remove_outliers", remove_outliers)
workflow.add_node("both", both)#+++++++++++++++++++++++++++++
workflow.add_node("none", none)#+++++++++++++++++++++++++++++
workflow.add_node("describe_data", describe_data)
workflow.add_node("output_results", output_results)

workflow.add_edge(START, "load_data")
workflow.add_edge("load_data", "summarize_data")
workflow.add_edge("summarize_data", "reasoning_node")
workflow.add_conditional_edges("reasoning_node", route_action, {
    "handle_missing_values": "handle_missing_values",
    "remove_outliers": "remove_outliers",
    "both": "both",#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    "none": "none",#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    "describe_data": "describe_data",
})
workflow.add_edge("handle_missing_values", "describe_data")
workflow.add_edge("remove_outliers", "describe_data")
workflow.add_edge("both", "describe_data")#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
workflow.add_edge("none", "describe_data")#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
workflow.add_edge("describe_data", "output_results")
workflow.add_edge("output_results", END)

graph = workflow.compile()

# ---------------------------
# 6. Visualize Graph
# ---------------------------

def save_graph_visualization():
    """Save the workflow graph as a PNG image."""
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        output_path = PROJECT_ROOT / "outputs" / "workflow_graph.png"
        with open(output_path, "wb") as f:
            f.write(png_data)
        print(f"Workflow graph saved to outputs/workflow_graph.png\n")
    except Exception as e:
        print(f"Could not generate graph visualization: {e}\n")


# ---------------------------
# 7. Run Example
# ---------------------------

if __name__ == "__main__":
    # Save workflow visualization
    save_graph_visualization()
    
    # Run the workflow
    #csv_path = str(PROJECT_ROOT / "data" / "outliers.csv")
    csv_path = str(PROJECT_ROOT / "data" / "missing_and_outliers.csv")
    #csv_path = str(PROJECT_ROOT / "data" / "missing.csv")
    ##"missing_and_outliers.csv")
    init_state: DataState = {
        "csv_path": csv_path,
        "df": None,
        "action": "none",
        "summary": "",
    }
    graph.invoke(init_state)