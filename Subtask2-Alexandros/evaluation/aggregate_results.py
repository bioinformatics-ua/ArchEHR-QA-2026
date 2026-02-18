import json
import argparse
from pathlib import Path
import pandas as pd
import plotly.express as px


ALL_METRICS = [
    "strict_macro_precision",
    "strict_macro_recall",
    "strict_macro_f1",
    "strict_micro_precision",
    "strict_micro_recall",
    "strict_micro_f1",
    "lenient_macro_precision",
    "lenient_macro_recall",
    "lenient_macro_f1",
    "lenient_micro_precision",
    "lenient_micro_recall",
    "lenient_micro_f1",
    "overall_score",
]


def parse_model_prompt(filename: str):
    name = filename.replace(".json", "")
    parts = name.split("_prompt_")
    if len(parts) == 2:
        return parts[0], int(parts[1])
    return name, None


def load_results(results_dir: Path):
    rows = []

    for file in sorted(results_dir.glob("*.json")):
        if file.name in ("scores.json", "summary.json"):
            continue

        with open(file) as f:
            data = json.load(f)

        model_name, prompt_idx = parse_model_prompt(file.name)

        row = {
            "model": model_name,
            "prompt": prompt_idx,
            "file": file.stem,
        }

        for m in ALL_METRICS:
            row[m] = data.get(m)

        rows.append(row)

    return pd.DataFrame(rows)


def build_dashboard(df, sort_by):
    df = df.sort_values(sort_by, ascending=False)

    fig = px.bar(
        df,
        x="prompt",
        y=sort_by,
        color="model",
        barmode="group",
        hover_data=ALL_METRICS,
        title=f"Model Comparison ({sort_by})",
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Prompt Index",
        yaxis_title=sort_by,
        legend_title="Model",
    )

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sort-by", default="overall_score")
    args = parser.parse_args()

    results_dir = Path("../results") / args.dataset
    df = load_results(results_dir)

    if df.empty:
        print("No result files found.")
        return

    df = df.round(4)

    fig = build_dashboard(df, args.sort_by)

    table_html = df.to_html(
        index=False,
        classes="display",
        border=0,
        justify="center",
    )

    csv_path = results_dir / "summary.csv"
    df.to_csv(csv_path, index=False)

    html_content = f"""
    <html>
    <head>
        <title>Subtask2 Results Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link rel="stylesheet"
              href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    </head>
    <body style="background-color:#111;color:white;font-family:sans-serif">

    <h1>Subtask 2 - Model Dashboard</h1>

    {fig.to_html(full_html=False)}

    <h2>All Results Table</h2>

    {table_html}

    <script>
        $(document).ready(function() {{
            $('table').DataTable();
        }});
    </script>

    </body>
    </html>
    """

    html_path = results_dir / "dashboard.html"

    with open(html_path, "w") as f:
        f.write(html_content)

    # Print top results to terminal
    top = df.sort_values("overall_score", ascending=False).head(10)
    print("\nTop 10 by overall_score:")
    print(top[["model", "prompt", "overall_score", "strict_micro_f1", "lenient_micro_f1"]].to_string(index=False))
    print(f"\nDashboard saved -> {html_path}")
    print(f"CSV saved -> {csv_path}")


if __name__ == "__main__":
    main()
