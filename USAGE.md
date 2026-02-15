# How to Use VNBR-Road Layers

This guide walks you through each part of the app and how to use it.

---

## 1. Start the App

1. Open a terminal in the project folder (where `app.py` and `3 layers.csv` are).
2. Run:
   ```bash
   streamlit run app.py
   ```
3. Your browser will open to the dashboard (e.g. http://localhost:8501).

---

## 2. Sidebar Filters

On the **left sidebar** you can filter the data used in most of the dashboard.

| Filter | What it does |
|--------|----------------|
| **Item** | Restrict to one layer: *All*, *Embankment*, *Subgrade*, or *GSB V*. Affects the KPI row, “Length by Item”, “Length by Bill No”, “Overall Progress”, and the **Data Table** at the bottom. |
| **Bill No** | Restrict to one bill: *All* or a specific bill (e.g. *Bill No. 07*). Same sections as above. |

- **All** = no filter for that field.
- Overlap/gap analysis, layer consistency, and layer coverage charts use the **full dataset** (all items, all bills) so chainage and hierarchy checks are consistent.

---

## 3. Top of Page: KPIs and Charts

### KPI row (4 metrics)

- **Total Length (filtered)** — Sum of *Length* for the current Item/Bill filter.
- **Progress vs Estimate** — Percentage of total length vs *Est Length* (e.g. 8000 m).
- **Records** — Number of rows after filtering.
- **Unique Bills** — Number of distinct Bill No values after filtering.

### Length by Item

- Bar chart: **Item** on X, **Length (m)** on Y.
- Shows total length per layer for the **filtered** data.

### Length by Bill No

- Bar chart: **Bill No** on X, **Length (m)** on Y.
- Only rows with a valid Bill No are included.

### Overall Progress by Layer

- Bar chart: **Item** on X, **Progress %** on Y (vs project length).
- 100% reference line shown.
- Uses **full** data (not sidebar filters).

---

## 4. Overlap & Gap Analysis

This section uses **merged chainages** (overlapping/adjacent stretches combined per Item).

### Tabs

- **Merged chainages**  
  Table of merged ranges per Item: *Item*, *Merged Start*, *Merged End*, *Merged Length (m)*.  
  Use this to see the “clean” chainage coverage per layer.

- **Overlaps**  
  Pairs of **original** segments (same Item) whose chainage ranges overlap.  
  Columns: Item, Stretch A/B, Bill A/B, Overlap Start/End, Overlap Length (m).  
  A total overlap length is shown. Use this to find double-counted or duplicate coverage.

- **Gaps**  
  Ranges where a layer has **no** coverage (from 0 to project end).  
  Columns: Item, Gap Start/End, Gap Length (m), Remark.  
  Use this to find missing chainage per layer.

---

## 5. Layer Consistency Check

- **Rule:** In the hierarchy (bottom → top)  
  **Embankment → Subgrade → GSB V → GSB III → WMM → DBM → BC**,  
  every upper layer’s chainage must lie **inside** the layer below.

- **All layer chainages**  
  Expand the section **“All layer chainages (merged ranges used in consistency check)”** to see the merged ranges per layer that are used for the check.

- **Result**
  - **Green message** — No violations; all top-layer chainages are within the layer below.
  - **Warning + table** — Violations listed with Upper Layer, Lower Layer, Ch. Start, Ch. End, Length (m), and Remark. Use this to fix chainage or data so top layers stay within bottom layers.

---

## 6. Layer Coverage by Chainage (per 1000 m)

### First chart block: one chart per 1000 m chunk

- **One subplot per chunk:** 0–1000 m, 1000–2000 m, 2000–3000 m, … up to project end.
- **In each subplot:**
  - **X-axis:** Chainage (m) in that chunk only.
  - **Y-axis:** Layers (bottom layer at **bottom**: Embankment, then Subgrade, then GSB V).
  - **Bars:** Coverage in that chunk only; each bar is a segment of a layer in that 1000 m range.

Use this to see **which layers** have coverage in **which 1000 m** band.

### Second chart: Coverage length per 1000 m chunk

- **X-axis:** Chunk label (0-1000, 1000-2000, …).
- **Y-axis:** Length (m).
- **Grouped bars by Layer** — One bar per layer per chunk; height = coverage length in that chunk.

Use this to compare **how much** of each layer is done in each 1000 m chunk.

---

## 7. Data Table

- **Bottom of the page.**  
- Shows the **filtered** rows (by Item and Bill No) with all columns.  
- **Length** is formatted with thousands separators.

Use this to inspect or export the subset of data you are analyzing.

---

## Quick Reference

| Goal | Where to look |
|-----|----------------|
| Total length / progress for a bill or item | KPIs + “Length by Item” / “Length by Bill No” |
| Clean chainage ranges per layer | Overlap & Gap → **Merged chainages** |
| Duplicate coverage (overlaps) | Overlap & Gap → **Overlaps** |
| Missing chainage (gaps) | Overlap & Gap → **Gaps** |
| Ranges used in hierarchy check | Layer Consistency → expand **All layer chainages** |
| Top layer outside bottom layer | Layer Consistency → violations table |
| Coverage per 1000 m per layer | **Layer Coverage** (Gantt per chunk + chunk bars) |

For setup and data format, see [README.md](README.md).
