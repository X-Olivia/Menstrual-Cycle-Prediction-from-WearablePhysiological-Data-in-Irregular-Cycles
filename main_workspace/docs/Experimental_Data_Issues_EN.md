# Experimental Data: Potential Issues Checklist

After reading the interim report and reviewing cycle_anchor code and output data, potential issues are listed here with whether they must be fixed before continuing.  
Source: filled during复盘; some items are pre-filled from current code/data—please verify and edit as needed.

---

## Issue List

| # | Brief description | Source | Must fix before continuing? | Recommended action | Status |
|---|-------------------|--------|-----------------------------|--------------------|--------|
| 1 | **One of 109 cycles is non-contiguous**: 18_2024 Cycle 2 has a missing day 974; only 18_2024 Cycle 1 was removed, Cycle 2 remains in clean. Sequence models assuming “contiguous days per cycle” will be wrong. | Code 7.1 output + weekly report | Recommended | Either remove 18_2024 Cycle 2 from clean or handle non-contiguous cycles in the model (e.g. mask or flag). | Pending |
| 2 | **cycle_anchor_clean.csv lacks cycle_id / day_in_cycle**: Export only has id, study_interval, day_in_study, phase, lh, estrogen, ovulation_day_method1/2. Downstream sequence models need grouping by cycle. | Export logic + CSV columns | Recommended | Add cycle_id (and optionally day_in_cycle) at export in the notebook, or derive and merge in a separate script. | Pending |
| 3 | **No next_menses_start_day or days_until_next_menses**: Main task time-to-event needs “days until next menses”. Current clean table has no such column. | Data plan vs current CSV | Yes | Add “next menses start day” for each cycle or compute days_until_next_menses in cycle/daily table. | Pending |
| 4 | **Very sparse ovulation labels**: Of 109 cycles ~81 have surge, ~28 have no ovulation day label (all 0); only ~78/77 days = 1. Auxiliary supervision is very sparse. | Notebook output + report 3.3 | Can defer | Document; report “cycles/days with ovulation label” in evaluation; sensitivity analysis if needed. | Pending |
| 5 | **Surge threshold mismatch**: find_surge_segments comment says “>= 2.5”, code uses ratio >= 2.0; weekly report says 2.0. | Notebook comment vs code | Recommended | Unify to 2.0 or 2.5 and update comment/docs. | Pending |
| 6 | (To be filled during复盘) |  |  |  |  |
| 7 | (To be filled during复盘) |  |  |  |  |

---

## Priority Fixes (Before Continuing Experiments)

After复盘, list 1–3 “must fix before continuing” items, e.g.:

- Must do: #1 (non-contiguous cycle), #2 (cycle_id), #3 (days_until_next_menses)
- Optional: #5 (threshold/comment alignment)

---

## Update Log

| Date | Update |
|------|--------|
| (Fill复盘 date) | Initial fill: 1–5 pre-filled; add 6–7 and priority fixes after复盘 |
