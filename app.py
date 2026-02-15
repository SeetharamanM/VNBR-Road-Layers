"""
Streamlit Dashboard - VNBR Road Layers
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import re

# Page config
st.set_page_config(
    page_title="VNBR-Road Layers",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load data
@st.cache_data
def load_data():
    csv_path = Path(__file__).parent / "Pavement Layers.csv"
    df = pd.read_csv(csv_path)
    # Parse Length as numeric (handle any edge cases)
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce").fillna(0).astype(int)
    # Parse Est Length for progress
    df["Est Length"] = pd.to_numeric(df["Est Length"], errors="coerce").fillna(8000)
    return df


def parse_stretch(stretch_str):
    """Parse 'start-end' or 'start - end' into (start, end) or (None, None)."""
    if pd.isna(stretch_str) or not str(stretch_str).strip():
        return None, None
    s = str(stretch_str).strip()
    # Match numbers separated by hyphen (e.g. 500-1000, 620-1000)
    m = re.match(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def merge_intervals(intervals):
    """Merge overlapping or adjacent intervals. intervals = list of (start, end)."""
    if not intervals:
        return []
    sorted_intervals = sorted([(s, e) for s, e in intervals if s is not None and e is not None])
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        if start <= merged[-1][1]:  # overlap or adjacent
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def compute_merged_and_analysis(df, est_length=8000):
    """Parse chainages, merge per Item, then compute overlaps and gaps."""
    # Parse Stretch
    parsed = df.apply(lambda r: parse_stretch(r["Stretch"]), axis=1)
    df = df.copy()
    df["chain_start"] = [p[0] for p in parsed]
    df["chain_end"] = [p[1] for p in parsed]

    merged_by_item = {}
    overlaps_list = []
    gaps_list = []

    for item in df["Item"].dropna().unique():
        sub = df[(df["Item"] == item) & df["chain_start"].notna()].copy()
        if sub.empty:
            continue

        intervals = list(zip(sub["chain_start"], sub["chain_end"]))
        merged = merge_intervals(intervals)
        merged_by_item[item] = merged

        # --- Overlap detection (before merge): pairs of segments that overlap
        rows = sub.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a_s, a_e = rows[i]["chain_start"], rows[i]["chain_end"]
                b_s, b_e = rows[j]["chain_start"], rows[j]["chain_end"]
                if a_s is None or a_e is None or b_s is None or b_e is None:
                    continue
                # Overlap if not (a_e <= b_s or b_e <= a_s)
                if not (a_e <= b_s or b_e <= a_s):
                    overlap_start = max(a_s, b_s)
                    overlap_end = min(a_e, b_e)
                    overlaps_list.append({
                        "Item": item,
                        "Stretch A": rows[i]["Stretch"],
                        "Bill A": rows[i]["Bill No"],
                        "Stretch B": rows[j]["Stretch"],
                        "Bill B": rows[j]["Bill No"],
                        "Overlap Start": overlap_start,
                        "Overlap End": overlap_end,
                        "Overlap Length (m)": round(overlap_end - overlap_start, 2),
                    })

        # --- Gap detection (after merge): gaps between 0 and first, between segments, and to est_length
        if not merged:
            gaps_list.append({
                "Item": item,
                "Gap Start": 0,
                "Gap End": est_length,
                "Gap Length (m)": est_length,
                "Remark": "No coverage",
            })
            continue
        # Gap from 0 to first segment
        if merged[0][0] > 0:
            gaps_list.append({
                "Item": item,
                "Gap Start": 0,
                "Gap End": merged[0][0],
                "Gap Length (m)": round(merged[0][0], 2),
                "Remark": "Start to first segment",
            })
        for k in range(len(merged) - 1):
            gap_start = merged[k][1]
            gap_end = merged[k + 1][0]
            if gap_end > gap_start:
                gaps_list.append({
                    "Item": item,
                    "Gap Start": gap_start,
                    "Gap End": gap_end,
                    "Gap Length (m)": round(gap_end - gap_start, 2),
                    "Remark": "Between segments",
                })
        if merged[-1][1] < est_length:
            gaps_list.append({
                "Item": item,
                "Gap Start": merged[-1][1],
                "Gap End": est_length,
                "Gap Length (m)": round(est_length - merged[-1][1], 2),
                "Remark": "Last segment to end",
            })

    df_with_chain = df
    merged_df = pd.DataFrame([
        {"Item": item, "Merged Start": s, "Merged End": e, "Merged Length (m)": round(e - s, 2)}
        for item, ranges in merged_by_item.items()
        for s, e in ranges
    ])
    overlaps_df = pd.DataFrame(overlaps_list)
    gaps_df = pd.DataFrame(gaps_list)
    return df_with_chain, merged_df, overlaps_df, gaps_df, merged_by_item


def gap_length_by_chunk(gaps_df, est_length=8000, chunk_size=1000):
    """For each Item and each chunk (0-1000, 1000-2000, ...), compute total gap length (m) in that chunk."""
    if gaps_df.empty:
        return pd.DataFrame(columns=["Item", "Chunk", "Chunk Start", "Gap Length (m)"])
    rows = []
    chunks = [(start, min(start + chunk_size, est_length)) for start in range(0, est_length, chunk_size)]
    for item in gaps_df["Item"].unique():
        item_gaps = gaps_df[gaps_df["Item"] == item]
        for (c_start, c_end) in chunks:
            length = 0
            for _, row in item_gaps.iterrows():
                gs, ge = row["Gap Start"], row["Gap End"]
                o_start = max(gs, c_start)
                o_end = min(ge, c_end)
                if o_end > o_start:
                    length += o_end - o_start
            rows.append({
                "Item": item,
                "Chunk": f"{c_start}-{c_end}",
                "Chunk Start": c_start,
                "Gap Length (m)": round(length, 2),
            })
    return pd.DataFrame(rows)


# Layer hierarchy: bottom (index 0) to top (index -1). Top layer chainages must be within bottom.
LAYER_ORDER = ["Embankment", "Subgrade", "GSB V", "GSB III", "WMM", "DBM", "BC"]


def segment_contained_in_intervals(seg_start, seg_end, intervals):
    """Check if segment [seg_start, seg_end] is fully contained in union of intervals."""
    if not intervals:
        return False
    remaining = [(seg_start, seg_end)]
    for (a, b) in sorted(intervals):
        new_remaining = []
        for (s, e) in remaining:
            if e <= a or s >= b:
                new_remaining.append((s, e))
            else:
                if s < a:
                    new_remaining.append((s, min(e, a)))
                if e > b:
                    new_remaining.append((max(s, b), e))
        remaining = [(s, e) for s, e in new_remaining if e > s]
        if not remaining:
            return True
    return False


def check_layer_consistency(merged_by_item, layer_order=None):
    """
    Top layer chainages must be within bottom layer. For each adjacent pair (lower, upper)
    in hierarchy, check every segment of upper is contained in lower's merged intervals.
    Returns list of dicts: { Upper Layer, Lower Layer, Ch. Start, Ch. End, Remark }.
    """
    layer_order = layer_order or LAYER_ORDER
    violations = []
    layers_present = [L for L in layer_order if L in merged_by_item and merged_by_item[L]]
    for i in range(len(layers_present) - 1):
        lower_layer = layers_present[i]
        upper_layer = layers_present[i + 1]
        lower_intervals = merged_by_item[lower_layer]
        upper_intervals = merged_by_item[upper_layer]
        for (u_start, u_end) in upper_intervals:
            if not segment_contained_in_intervals(u_start, u_end, lower_intervals):
                violations.append({
                    "Upper Layer": upper_layer,
                    "Lower Layer": lower_layer,
                    "Ch. Start": u_start,
                    "Ch. End": u_end,
                    "Length (m)": round(u_end - u_start, 2),
                    "Remark": f"{upper_layer} chainage {u_start}-{u_end} not fully within {lower_layer}",
                })
    return violations


def chunk_coverage(merged_by_item, est_length=8000, chunk_size=1000):
    """For each layer and each chunk (0-1000, 1000-2000, ...), compute covered length (m)."""
    chunks = []
    for start in range(0, est_length, chunk_size):
        end = min(start + chunk_size, est_length)
        chunks.append((start, end))
    rows = []
    for layer in LAYER_ORDER:
        if layer not in merged_by_item:
            continue
        for (c_start, c_end) in chunks:
            length = 0
            for (s, e) in merged_by_item[layer]:
                overlap_start = max(s, c_start)
                overlap_end = min(e, c_end)
                if overlap_end > overlap_start:
                    length += overlap_end - overlap_start
            rows.append({
                "Chunk": f"{c_start}-{c_end}",
                "Chunk Start": c_start,
                "Layer": layer,
                "Length (m)": round(length, 2),
            })
    return pd.DataFrame(rows)


df = load_data()
est_length = int(df["Est Length"].iloc[0]) if len(df) else 8000
df_with_chain, merged_df, overlaps_df, gaps_df, merged_by_item = compute_merged_and_analysis(df, est_length)
consistency_violations = check_layer_consistency(merged_by_item)
chunk_df = chunk_coverage(merged_by_item, est_length, chunk_size=1000)
gap_chunk_df = gap_length_by_chunk(gaps_df, est_length, chunk_size=1000)

# Sidebar filters
st.sidebar.header("Filters")
all_items = ["All"] + sorted(df["Item"].dropna().unique().tolist())
selected_item = st.sidebar.selectbox("Item", all_items, index=0)

bills = [b for b in df["Bill No"].dropna().unique().tolist() if str(b) != "0" and str(b) != "nan"]
# Sort by number in "Bill No. XX"
def bill_sort_key(x):
    s = str(x)
    if "No." in s:
        try:
            return int(s.split("No.")[-1].strip())
        except ValueError:
            return 0
    return 0
bill_options = ["All"] + sorted(bills, key=bill_sort_key)
selected_bill = st.sidebar.selectbox("Bill No", bill_options, index=0)

# Apply filters
filtered = df.copy()
if selected_item != "All":
    filtered = filtered[filtered["Item"] == selected_item]
if selected_bill != "All":
    filtered = filtered[filtered["Bill No"] == selected_bill]

# Header
st.title("📊 VNBR-Road Layers")
st.markdown("Overview of Embankment, Subgrade, and GSB V progress by bill and stretch.")
st.divider()

# Tiles: sum of length for each item (hierarchical order: Embankment → … → BC)
est_length = 8000
by_item_filtered = filtered.groupby("Item", as_index=False)["Length"].sum()
item_sums = by_item_filtered.set_index("Item")["Length"].to_dict()
items_for_tiles = [L for L in LAYER_ORDER if L in item_sums]
if not items_for_tiles:
    items_for_tiles = list(by_item_filtered["Item"].unique()) or ["—"]
n_tiles = len(items_for_tiles)
cols = st.columns(n_tiles)
for i, item in enumerate(items_for_tiles):
    with cols[i]:
        total = item_sums.get(item, 0)
        st.metric(item, f"{total:,.0f} m")

st.divider()

# Charts row
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Length by Item")
    item_totals = filtered.groupby("Item", as_index=False)["Length"].sum()
    # Order by hierarchy: Embankment, Subgrade, GSB V, GSB III, WMM, DBM, BC
    order_idx = {L: i for i, L in enumerate(LAYER_ORDER)}
    item_totals["_order"] = item_totals["Item"].map(lambda x: order_idx.get(x, 99))
    item_totals = item_totals.sort_values("_order").drop(columns=["_order"])
    fig_item = px.bar(
        item_totals,
        x="Item",
        y="Length",
        color="Length",
        color_continuous_scale="teal",
        text_auto=",.0f",
    )
    fig_item.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20),
        xaxis_title="",
        yaxis_title="Length (m)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(categoryorder="array", categoryarray=[i for i in LAYER_ORDER if i in item_totals["Item"].values]),
    )
    st.plotly_chart(fig_item, use_container_width=True)

with chart_col2:
    st.subheader("Length by Bill No")
    bill_df = filtered[filtered["Bill No"].astype(str).str.contains(r"\d", na=False)]
    bill_item_totals = bill_df.groupby(["Bill No", "Item"], as_index=False)["Length"].sum()
    if not bill_item_totals.empty:
        fig_bill = px.bar(
            bill_item_totals,
            x="Bill No",
            y="Length",
            color="Item",
            barmode="group",
            text_auto=",.0f",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_bill.update_layout(
            margin=dict(t=20, b=20),
            xaxis_title="",
            yaxis_title="Length (m)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bill, use_container_width=True)
    else:
        st.info("No bill data for current filters.")

# Progress by item (full dataset): length in meters
st.subheader("Overall Progress by Layer")
est_length = 8000
progress_data = []
for item in df["Item"].unique():
    if pd.isna(item):
        continue
    total = df[df["Item"] == item]["Length"].sum()
    pct = (total / est_length * 100) if est_length else 0
    progress_data.append({"Item": item, "Length": total, "Progress %": min(100, pct)})
progress_df = pd.DataFrame(progress_data)
if not progress_df.empty:
    progress_df["_order"] = progress_df["Item"].map(lambda x: LAYER_ORDER.index(x) if x in LAYER_ORDER else 99)
    progress_df = progress_df.sort_values("_order").drop(columns=["_order"])
    progress_df["Label"] = progress_df.apply(
        lambda r: f"{r['Length']:,.0f} m ({r['Progress %']:.1f}%)", axis=1
    )
if not progress_df.empty:
    fig_progress = px.bar(
        progress_df,
        x="Item",
        y="Length",
        color="Length",
        color_continuous_scale="viridis",
        text="Label",
    )
    fig_progress.add_hline(y=est_length, line_dash="dash", line_color="gray")
    fig_progress.update_traces(textposition="outside")
    fig_progress.update_layout(
        yaxis_title="Length (m)",
        showlegend=False,
        margin=dict(t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(categoryorder="array", categoryarray=progress_df["Item"].tolist()),
    )
    st.plotly_chart(fig_progress, use_container_width=True)

# --- Overlap and Gap Analysis (after merging chainages) ---
st.divider()
st.subheader("Overlap & Gap Analysis (after merging chainages)")
st.caption("Chainages from the Stretch column are parsed, merged per Item, then overlaps (duplicate coverage) and gaps (missing coverage) are reported.")

# Per-item overlap and gap length summary
overlap_by_item = overlaps_df.groupby("Item", as_index=False)["Overlap Length (m)"].sum() if not overlaps_df.empty else pd.DataFrame(columns=["Item", "Overlap Length (m)"])
gap_by_item = gaps_df.groupby("Item", as_index=False)["Gap Length (m)"].sum() if not gaps_df.empty else pd.DataFrame(columns=["Item", "Gap Length (m)"])
all_items_set = set(overlap_by_item["Item"].tolist()) | set(gap_by_item["Item"].tolist())
if not merged_df.empty:
    all_items_set |= set(merged_df["Item"].tolist())
items_with_data = sorted(
    all_items_set,
    key=lambda x: (LAYER_ORDER.index(x) if x in LAYER_ORDER else 99),
)
summary_rows = []
for item in items_with_data:
    ov = overlap_by_item[overlap_by_item["Item"] == item]["Overlap Length (m)"].sum()
    gp = gap_by_item[gap_by_item["Item"] == item]["Gap Length (m)"].sum()
    summary_rows.append({"Item": item, "Overlap Length (m)": round(ov, 2), "Gap Length (m)": round(gp, 2)})
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    st.markdown("**Overlap and gap length by item**")
    st.dataframe(summary_df, use_container_width=True, height=min(200, 50 * len(summary_df)), hide_index=True)

# Gap length per km (0-1000, 1000-2000, ...) for each item
if not gap_chunk_df.empty:
    st.markdown("**Gap length per km (by item)** — total gap length in each 1000 m chainage chunk.")
    # Pivot: Item x Chunk -> Gap Length (m)
    pivot_gap = gap_chunk_df.pivot(index="Item", columns="Chunk", values="Gap Length (m)").fillna(0)
    chunk_cols = [c for c in pivot_gap.columns if re.match(r"^\d+-\d+$", str(c))]
    chunk_cols_sorted = sorted(chunk_cols, key=lambda x: int(x.split("-")[0]))
    pivot_gap = pivot_gap.reindex(columns=chunk_cols_sorted)
    pivot_gap["Total (m)"] = pivot_gap.sum(axis=1)
    row_order = [i for i in LAYER_ORDER if i in pivot_gap.index] + [i for i in pivot_gap.index if i not in LAYER_ORDER]
    pivot_gap = pivot_gap.reindex(row_order).reset_index()
    # Ensure Item is leftmost: columns = [Item, chunk columns..., Total (m)]
    col_order = ["Item"] + chunk_cols_sorted + ["Total (m)"]
    pivot_gap = pivot_gap[col_order]
    numeric_cols = chunk_cols_sorted + ["Total (m)"]
    st.dataframe(pivot_gap.style.format("{:.2f}", subset=numeric_cols), use_container_width=True, height=min(350, 50 * len(pivot_gap) + 30), hide_index=True)
    # Bar chart: gap length by chunk for each item (grouped)
    fig_gap_chunk = px.bar(
        gap_chunk_df,
        x="Chunk",
        y="Gap Length (m)",
        color="Item",
        barmode="group",
        title="Gap length per 1000 m chunk (by item)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_gap_chunk.update_layout(
        xaxis_title="Chainage chunk (m)",
        yaxis_title="Gap length (m)",
        height=320,
        margin=dict(t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"categoryorder": "array", "categoryarray": chunk_cols_sorted if chunk_cols_sorted else gap_chunk_df["Chunk"].unique().tolist()},
    )
    st.plotly_chart(fig_gap_chunk, use_container_width=True)
st.markdown("")  # spacing before tabs

tab_merged, tab_overlaps, tab_gaps = st.tabs(["Merged chainages", "Overlaps", "Gaps"])

with tab_merged:
    st.markdown("**Merged chainage ranges** — overlapping or adjacent segments combined per Item.")
    if not merged_df.empty:
        st.dataframe(merged_df, use_container_width=True, height=300)
        st.metric("Total merged segments", len(merged_df))
    else:
        st.info("No merged segments.")

with tab_overlaps:
    st.markdown("**Overlaps** — segments (same Item) whose chainage ranges overlap (double-counted length).")
    if not overlaps_df.empty:
        st.dataframe(overlaps_df, use_container_width=True, height=300)
        total_overlap = overlaps_df["Overlap Length (m)"].sum()
        st.metric("Total overlap length (m)", f"{total_overlap:,.2f}")
    else:
        st.success("No overlaps detected.")

with tab_gaps:
    st.markdown("**Gaps** — missing chainage ranges per Item (between 0 and project end).")
    if not gaps_df.empty:
        st.dataframe(gaps_df, use_container_width=True, height=300)
    else:
        st.success("No gaps detected.")

# --- Layer Consistency (top layer chainages within bottom layer) ---
st.divider()
st.subheader("Layer Consistency Check")
st.caption("Hierarchy (bottom → top): Embankment → Subgrade → GSB V → GSB III → WMM → DBM → BC. Top layer chainages must lie within the layer below.")

# Include all layer chainages used in the check (merged ranges per layer)
all_layer_chainages = []
for layer in LAYER_ORDER:
    if layer not in merged_by_item or not merged_by_item[layer]:
        continue
    for (s, e) in merged_by_item[layer]:
        all_layer_chainages.append({
            "Layer": layer,
            "Ch. Start": s,
            "Ch. End": e,
            "Length (m)": round(e - s, 2),
        })
all_layer_chainages_df = pd.DataFrame(all_layer_chainages)
if not all_layer_chainages_df.empty:
    with st.expander("All layer chainages (merged ranges used in consistency check)", expanded=False):
        st.dataframe(all_layer_chainages_df, use_container_width=True, height=300)

consistency_df = pd.DataFrame(consistency_violations)
if consistency_df.empty:
    st.success("No layer consistency violations: all top-layer chainages are within the layer below.")
else:
    st.warning(f"Found {len(consistency_df)} violation(s): top layer has chainage not covered by the layer below.")
    st.dataframe(consistency_df, use_container_width=True, height=250)

# --- Layer coverage chart: one Gantt per 1000 m chunk ---
st.divider()
st.subheader("Layer Coverage by Chainage (per 1000 m)")
st.caption("Each chart shows one 1000 m chainage chunk; all layers with coverage in that chunk are shown.")

layers_with_data = [L for L in LAYER_ORDER if L in merged_by_item and merged_by_item[L]]
chunk_size = 1000
chunks = [(start, min(start + chunk_size, est_length)) for start in range(0, est_length, chunk_size)]

def segments_in_chunk(segments, c_start, c_end):
    """Return list of (start, end) for parts of segments that fall inside [c_start, c_end]."""
    out = []
    for (s, e) in segments:
        o_start = max(s, c_start)
        o_end = min(e, c_end)
        if o_end > o_start:
            out.append((o_start, o_end))
    return out

if layers_with_data and chunks:
    n_chunks = len(chunks)
    fig = make_subplots(
        rows=n_chunks,
        cols=1,
        subplot_titles=[f"{c_start}–{c_end} m" for c_start, c_end in chunks],
        shared_xaxes=False,
        vertical_spacing=0.06,
        row_heights=[1] * n_chunks,
    )
    colors = px.colors.qualitative.Set2
    for row, (c_start, c_end) in enumerate(chunks, start=1):
        for idx, layer in enumerate(layers_with_data):
            segments = segments_in_chunk(merged_by_item[layer], c_start, c_end)
            if not segments:
                continue
            x_lengths = [e - s for s, e in segments]
            bases = [s for s, e in segments]
            fig.add_trace(
                go.Bar(
                    x=x_lengths,
                    y=[layer] * len(segments),
                    base=bases,
                    orientation="h",
                    name=layer,
                    legendgroup=layer,
                    showlegend=(row == 1),
                    marker_color=colors[idx % len(colors)],
                ),
                row=row,
                col=1,
            )
    for row, (c_start, c_end) in enumerate(chunks, start=1):
        fig.update_xaxes(range=[c_start, c_end], row=row, col=1, title_text="Chainage (m)" if row == n_chunks else "")
    fig.update_layout(
        height=200 * n_chunks,
        margin=dict(t=60, b=40),
        barmode="overlay",
        bargap=0.3,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    for row in range(1, n_chunks + 1):
        fig.update_yaxes(categoryorder="array", categoryarray=layers_with_data, row=row, col=1)
    st.plotly_chart(fig, use_container_width=True)

# Chunk bars: X = chunks (0-1000, 1000-2000, ...), grouped bars by layer
if not chunk_df.empty:
    fig_chunk = px.bar(
        chunk_df,
        x="Chunk",
        y="Length (m)",
        color="Layer",
        barmode="group",
        title="Coverage length per 1000 m chunk (all layers)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_chunk.update_layout(
        xaxis_title="Chainage chunk (m)",
        yaxis_title="Length (m)",
        height=320,
        margin=dict(t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_chunk, use_container_width=True)

st.divider()
st.subheader("Data Table")
st.dataframe(
    filtered.style.format({"Length": "{:,}"}, subset=["Length"]),
    use_container_width=True,
    height=400,
)
