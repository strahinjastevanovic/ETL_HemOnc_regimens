import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


METRICS = [
    "size",
    "cardinality",
    "schema_drift",
    "lost_key_count",
    "gained_key_count",
    "jaccard_unique",
    "js_divergence",
    "score"
]

def format_exact_diff(diff_obj):
    if not diff_obj or not isinstance(diff_obj, dict):
        return "None"
    
    lines = []
    for k, v in diff_obj.items():
        key_label = k.replace('_', ' ').capitalize()
        if isinstance(v, list):
            if not v:
                lines.append(f"<b>{key_label}:</b> None")
            else:
                lines.append(f"<b>{key_label}:</b>")
                for item in v:
                    if isinstance(item, str) and "," in item:
                        sub_items = [si.strip() for si in item.split(",")]
                        for si in sub_items:
                            lines.append(f"  - {si}")
                    else:
                        lines.append(f"  - {item}")
        elif isinstance(v, dict):
            lines.append(f"<b>{key_label}:</b>")
            for sub_k, sub_v in v.items():
                lines.append(f"  {sub_k}: {sub_v}")
        else:
            lines.append(f"<b>{key_label}:</b> {v}")
    
    return "<br>".join(lines)

def load_table_metadata(metadata_path):
    if not metadata_path.exists():
        return {}
    with open(metadata_path, 'r') as f:
        return json.load(f)

def load_and_preprocess(report_path, metadata_path):
    with open(report_path, 'r') as f:
        data = json.load(f)
    
    metadata = load_table_metadata(metadata_path)
    rows = []
    
    def visit(node, path):
        if not isinstance(node, dict):
            return
        
        has_metrics = any(m in node for m in ["score", "jaccard_unique", "js_divergence", "size_ref"])
        
        if has_metrics:
            col_name = path[-1]
            table_name = path[0] 
            
            table_meta = metadata.get(table_name, {})
            if table_meta.get("skip"):
                return

            if node.get("score") == 1 and node.get("jaccard_unique") == 1.0 and node.get("js_divergence") == 0.0:
                return

            row = {
                "full_path": " / ".join(path),
                "table": table_name,
                "column": col_name,
                "exact_diff_raw": node.get("exact_diff", {})
            }
            
            row["size"] = node.get("size_new", 1) / node.get("size_ref", 1) if node.get("size_ref", 1) > 0 else 1.0
            row["cardinality"] = node.get("cardinality_new", 1) / node.get("cardinality_ref", 1) if node.get("cardinality_ref", 1) > 0 else 1.0
            row["js_divergence"] = round(float(node.get("js_divergence", 0.0)), 1)
            row["score"] = node.get("score", 0)
            row["jaccard_unique"] = node.get("jaccard_unique", 0)
            row["lost_key_count"] = node.get("lost_key_count", 0)
            row["gained_key_count"] = node.get("gained_key_count", 0)
            row["schema_drift"] = 0
            
            rows.append(row)
        else:
            for k, v in node.items():
                if k.startswith("_"): continue
                visit(v, path + [k])

    visit(data, [])
    df = pd.DataFrame(rows)
    if df.empty: return df, metadata

    def get_base(t):
        for s in [".resolved", ".reported"]:
            if t.endswith(s): return t[:-len(s)]
        return t
    
    df['base_table'] = df['table'].apply(get_base)
    df = df[~df['table'].apply(lambda t: metadata.get(t, {}).get("skip", False))]
    
    return df, metadata

def create_interactive_report(report_path=None, output_dir=None):
    """
    Create interactive HTML report from comparison JSON.
    
    Args:
        report_path: Path to comparison_report.json (default: ./comparison_report.json)
        output_dir: Directory to save HTML output (default: ./html/)
    """
    if report_path is None:
        report_path = Path("comparison_report.json")
    else:
        report_path = Path(report_path)
    
    if output_dir is None:
        output_dir = Path("html")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadata always in regression dir
    metadata_path = Path(__file__).parent / "table_names.json"
    
    df, metadata = load_and_preprocess(report_path, metadata_path)
    if df.empty:
        print("No differences found matching threshold.")
        return

    tables = sorted(df['table'].unique())

    def sort_key(t):
        meta = metadata.get(t, {})
        return (meta.get("order", 9999), t)
    
    tables.sort(key=sort_key)
    
    display_titles = [metadata.get(t, {}).get("name", t) for t in tables]
    
    v_spacing = min(0.08, 0.9 / (len(tables) - 1)) if len(tables) > 1 else 0.05
    
    fig = make_subplots(
        rows=len(tables), cols=1, 
        subplot_titles=display_titles,
        vertical_spacing=v_spacing,
        shared_xaxes=False
    )

    # Professional color scale: Blue (low/bad) to Green (high/good)
    custom_colorscale = [
        [0.0, "#440154"], # Dark Purple/Blue (bad/change)
        [0.5, "#21908C"], # Teal
        [1.0, "#5DC963"]  # Green (good/no change)
    ]

    for i, table in enumerate(tables):
        sub_df = df[df['table'] == table]
        unique_cols = sub_df['column'].unique()
        display_cols = [c.replace('_', ' ') for c in unique_cols]
        display_metrics = [m.replace('_', ' ') for m in METRICS]
        
        pivot_data = []
        for metric in METRICS:
            row_vals = [sub_df[sub_df['column'] == col_name][metric].values[0] for col_name in unique_cols]
            pivot_data.append(row_vals)
            
        pivot_df = pd.DataFrame(pivot_data, index=display_metrics, columns=display_cols)
        
        table_meta = metadata.get(table, {})
        desc = table_meta.get("desc", "")
        if desc:
            fig.add_annotation(
                text=f"<i>{desc}</i>",
                xref="paper", yref="paper",
                x=0, y=1.03,
                showarrow=False,
                font=dict(size=11, color="gray"),
                align="left",
                row=i+1, col=1
            )

        hover_matrix = []
        for metric in METRICS:
            hover_row = []
            for col_name in unique_cols:
                col_row = sub_df[sub_df['column'] == col_name].iloc[0]
                pretty_diff = format_exact_diff(col_row['exact_diff_raw'])
                hover_text = (
                    f"<b>Table:</b> {table_meta.get('name', table)}<br>"
                    f"<b>Col:</b> {col_name.replace('_', ' ')}<br>"
                    f"<b>Metric:</b> {metric.replace('_', ' ')}<br>"
                    f"<b>Value:</b> {col_row[metric]:.3f}<br>"
                    f"-----------------------------<br>"
                    f"{pretty_diff}"
                )
                hover_row.append(hover_text)
            hover_matrix.append(hover_row)

        hm = go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale=custom_colorscale,
            zmin=0, zmax=1, 
            text=hover_matrix,
            hoverinfo="text",
            showscale=True, # Show legend for every diagram
            colorbar=dict(
                title=dict(
                    text="Stability Score<br><span style='font-size:10px'>(1.0=Matches Ref, 0.0=Drift)</span>",
                    side="top"
                ),
                tickvals=[0, 0.5, 1],
                ticktext=["Regression", "Warning", "Stable"],
                len=min(0.8 / len(tables), 0.1), # Cap the length
                y=1 - (i / len(tables)) - (0.4 / len(tables)),
                yanchor="middle"
            ),
            name=f"hm_{i}"
        )
        fig.add_trace(hm, row=i+1, col=1)
        fig.update_xaxes(showticklabels=True, tickangle=45, row=i+1, col=1)

    fig.update_layout(
        height=900 * len(tables), 
        title_text="Clustermap Regression Report",
        template="plotly_white",
        margin=dict(t=120, b=50, l=150, r=150)
    )
    
    toc_html = "<div style='font-family: sans-serif; margin: 20px; padding: 20px; background: #f8f9fa; border-radius: 8px;'>"
    toc_html += "<h2>Sections</h2>"
    toc_html += "<h5 style='font-style: italic;'>Note: Columns with detected changes are published.</h5><ul style='list-style-type: none; padding-left: 0;'>"
    
    for i, table in enumerate(tables):
        table_meta = metadata.get(table, {})
        name = table_meta.get("name", table)
        desc_toc = table_meta.get("desc", "")
        
        sub_df = df[df['table'] == table]
        unique_cols = sorted(sub_df['column'].unique())
        
        toc_html += f"<li style='margin-bottom: 20px; border-bottom: 1px solid #dee2e6; padding-bottom: 15px;'>"
        js_scroll = f"window.scrollTo({{top: document.getElementsByClassName('subplot-title')[{i}].getBoundingClientRect().top + window.pageYOffset - 50, behavior: 'smooth'}});"
        toc_html += f"<a href='#' style='font-weight: bold; font-size: 1.25em; text-decoration: none; color: #007bff;' onclick=\"{js_scroll} return false;\">📌 {name}</a>"
        
        if desc_toc:
            toc_html += f"<div style='color: #666; font-style: italic; margin-top: 5px; font-size: 0.95em;'>{desc_toc}</div>"
        
        if unique_cols:
            tags_html = "; ".join([f"<span style='background: #fff; padding: 2px 8px; border-radius: 12px; border: 1px solid #ddd; font-size: 0.85em;'>{c.replace('_', ' ')}</span>" for c in unique_cols])
            toc_html += f"<div style='margin-top: 10px; line-height: 2;'>{tags_html}</div>"
            
        toc_html += "</li>"
    
    toc_html += "</ul></div>"

    output_file = output_dir / "final_output.html"
    full_body = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    html_template = f"""
    <html>
    <head><title>Regression Report</title></head>
    <body style='margin: 0; padding: 0;'>
        {toc_html}
        <div id='plotly-div'>
            {full_body}
        </div>
        <script>
            document.querySelectorAll('.g-gtitle').forEach((el, i) => {{
                el.classList.add('subplot-title');
            }});
        </script>
    </body>
    </html>
    """
    
    with open(output_file, "w") as f:
        f.write(html_template)
        
    print(f"* Interactive report generated: {output_file}")

if __name__ == "__main__":
    create_interactive_report()
