# STYLE_CHANGES.md — Copy-Edit Log for `paper/clanm.tex`

## 1. Overview

This document records every category of change made to `paper/clanm.tex` during
the copy-editing pass. No numbers, table values, citations, or experimental claims
were added. Claims that could not be causally supported were hedged or softened.

---

## 2. Style Rule Changes

### Rule 1 — Promotional / absolute words removed or replaced

| Original phrase | Replacement |
|---|---|
| proves / demonstrates / establishing | "is consistent with", "we observe", "suggests" |
| slashes / eliminates / virtually eliminating | "reduces … from X to Y" |
| vastly outperform | "outperform" (number retained in full) |
| near-perfect | removed; number kept |
| key phenomena / key contribution | removed or given the measurement |
| validating | removed |
| stabilises | "reduces" (measurement given) |
| reveals key phenomena | "shows" |

### Rule 2 — Unsupported causal claims hedged

| Original claim | Treatment |
|---|---|
| "OVS table saturation caused B2 degradation" | Softened: "flood traffic may saturate switch tables or queues" |
| "flow-mod queuing delay caused cascades" | Softened: "a contributing factor is that the combined duration … delays flood mitigation" |
| "decision-latency-induced cascades" | Rephrased to "cascades that follow slow decisions" |
| "74% over-refusal" | Removed exact %; re-framed as observation; not in source CSV |

### Rule 3 — Em dashes reduced to 0 (from 4)

All em dashes (`---`) converted to commas, semicolons, or rephrased clauses.
"X, not Y" antithesis constructions removed. Noun stacks unpacked.

### Rule 4 — Paragraph-ending summary sentences removed

Removed trailing "Thus,", "Consequently,", "Similarly," sentences that restated
content without adding information.

### Rule 5 — Run-in bold headings converted to `\paragraph{}`

All `\textbf{Heading.}` run-in bold constructions replaced with `\paragraph{heading}`.
In-paragraph numbered lists converted to flowing sentences or proper `\begin{itemize}`.

### Rule 6 — Repetition cut

- Abstract: 5 numbers only — 3,960 · 8.3% · 0.1% · 26.6 s · 9.7 s.
- Introduction: contributions stated once; numbers removed from intro summary.
- Conclusion: no new numbers; ≤8 sentences.

### Rule 7 — Sentence variety and honest limitations

Mixed long and short sentences throughout. Added direct limitation statement
about Mininet MTTR vs. measured model latency.

### Rule 8 — Spelling convention (British) and terminology

- `generalisation`, `localised`, `parameterised`, `stabilise` throughout.
- Consistent term: "language model (LM)" for the planner; "emulation" for Mininet;
  "simulation" for NetworkX study.
- Removed `(Proposed)` from Fig. 10 / `fig_slm_scaling_ablation` legend.
- Fig. 10 caption updated to match chart axes.

### Rule 9 — Forbidden words removed

Removed: delve, leverage, robust, seamless, holistic, novel, comprehensive,
pivotal, underscores, landscape, tapestry.

---

## 3. Number Safety Check

**Before:** `nums_before.txt` from original `clanm.tex` commit.
**After:** `nums_after.txt` from edited `clanm.tex`.

### Explanation of differences

All count differences are due to:

1. **Deduplication** — Numbers previously repeated in abstract/intro/conclusion
   now appear once in Results only (e.g. `1,440` episode count, `3,960`).

2. **New Appendix Table A3** (`tab:main_ci`) — Full 95% CI boundary numbers
   (e.g. `[84.2, 90.8]`) now appear in both main Table 2 (means) and Appendix A3
   (full CIs). Count increases are solely from this structural move.

3. **Numbers `0.81`, `0.93`** — Sensitivity paragraph originally used proportions.
   An intermediate edit accidentally converted to percentages; reverted correctly.

4. **Number `98.8`** — Originally in Discussion ("removes 98.8% of invalid
   executions"). Accidentally dropped during restructuring; restored.

5. **Number `74`** — Exact percentage not in source CSV; removed and hedged.

6. **Numbers `440`, `25.0`, `24.7`, `28.6`** — Count shifts from (a) abstract/intro
   no longer repeating episode counts; (b) CI boundaries duplicated to Appendix A3.

No table data values, figure values, or primary result numbers were modified.

---

## 4. Table Layout Changes

| Table | Change |
|---|---|
| **Table 2 (main)** | Split into (a) Performance Means and (b) Safety/Overhead; `adjustbox` |
| **New Appendix Table A3** | Full metrics with 95% CIs, TRT, availability, Ctl ms/ep |
| **Table 3** | Split into Baselines & FULL vs Ablations A1–A6 stacked blocks |
| **Tables 4, 5** | Stacked performance/overhead blocks; `adjustbox` |
| **Appendix Tables A1, A2** | Split into Baselines & FULL vs Ablations halves |
| **All tables** | `\footnotesize\setlength{\tabcolsep}{3pt}`, `\centering` |

---

## 5. Figure Changes

| Figure | Change |
|---|---|
| **Fig. 1** | Replaced raster PNG with clean vector TikZ diagram (6 boxes, arrows, feedback loop, decision diamond; no stock icons) |
| **Fig. 10** | Caption updated to match chart axes: syntax, schema, accuracy, hallucination, rejection |
| **All figures** | `\centering`, `\FloatBarrier` before each `\section` |

---

## 6. Structural Changes

- Added `\usepackage{adjustbox}`, `\usepackage{placeins}`, `\usepackage{tikz}`.
- `\FloatBarrier` added before each `\section`.

---

## 7. Claims Softened or Removed

| Claim | Action |
|---|---|
| "proves that deterministic verification is indispensable" | Softened to "is consistent with the view that verification remains beneficial" |
| "virtually eliminates hallucinations" | → "reduces hallucination rate from 23.0% to 0.0%" |
| "decision-latency-induced cascades explain S5 failures" | → hedged with "a contributing factor is …" |
| "OVS table saturation explains B2 emulation degradation" | → "may saturate switch tables or queues" |
| "74% over-refusal (Qwen2.5-7B)" | Removed (not in source CSV); replaced with qualitative observation |
