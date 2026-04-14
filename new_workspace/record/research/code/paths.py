from __future__ import annotations

from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = CODE_DIR.parent
RECORD_DIR = RESEARCH_DIR.parent
NEW_WS = RECORD_DIR.parent
PIPELINE_DIR = RECORD_DIR / "experiment" / "multisignal_pipeline"

DOCS_DIR = RESEARCH_DIR / "docs"
METHODOLOGY_DIR = DOCS_DIR / "methodology"
REPORTS_DIR = DOCS_DIR / "reports"
RESULTS_DIR = RESEARCH_DIR / "results"
SUBGROUP_DIR = RESULTS_DIR / "subgroup_tables"
MANIFEST_DIR = RESULTS_DIR / "manifests"
BASELINE_DIR = RESULTS_DIR / "baseline_subgroup_analysis"
PERSONALIZATION_DIR = RESULTS_DIR / "personalization"

SUBGROUP_TABLE_CSV = SUBGROUP_DIR / "user_cycle_subgroups_v1.csv"
SUBGROUP_SUMMARY_CSV = SUBGROUP_DIR / "subgroup_summary_v1.csv"
SUBGROUP_SUMMARY_MD = SUBGROUP_DIR / "subgroup_summary_v1.md"
MANIFEST_JSON = MANIFEST_DIR / "subgroup_build_manifest_v1.json"
BASELINE_RESULTS_CSV = BASELINE_DIR / "baseline_subgroup_results_v1.csv"
BASELINE_RESULTS_MD = BASELINE_DIR / "baseline_subgroup_results_v1.md"
BASELINE_MANIFEST_JSON = MANIFEST_DIR / "baseline_subgroup_manifest_v1.json"
L1_PROFILE_CSV = PERSONALIZATION_DIR / "l1_zero_shot_profile_v1.csv"
L1_RESULTS_CSV = PERSONALIZATION_DIR / "l1_zero_shot_results_v1.csv"
L1_RESULTS_MD = PERSONALIZATION_DIR / "l1_zero_shot_results_v1.md"
L1_MANIFEST_JSON = MANIFEST_DIR / "l1_zero_shot_manifest_v1.json"
L2_PROFILE_CSV = PERSONALIZATION_DIR / "l2_one_shot_profile_v1.csv"
L2_RESULTS_CSV = PERSONALIZATION_DIR / "l2_one_shot_results_v1.csv"
L2_RESULTS_MD = PERSONALIZATION_DIR / "l2_one_shot_results_v1.md"
L2_MANIFEST_JSON = MANIFEST_DIR / "l2_one_shot_manifest_v1.json"
L3_PROFILE_CSV = PERSONALIZATION_DIR / "l3_few_shot_profile_v1.csv"
L3_RESULTS_CSV = PERSONALIZATION_DIR / "l3_few_shot_results_v1.csv"
L3_RESULTS_MD = PERSONALIZATION_DIR / "l3_few_shot_results_v1.md"
L3_MANIFEST_JSON = MANIFEST_DIR / "l3_few_shot_manifest_v1.json"

L1_AUDIT_JSON = PERSONALIZATION_DIR / "l1_active_effect_audit_v1.json"
L1_AUDIT_CYCLE_CSV = PERSONALIZATION_DIR / "l1_active_effect_cycles_v1.csv"
L2_AUDIT_JSON = PERSONALIZATION_DIR / "l2_active_effect_audit_v1.json"
L2_AUDIT_CYCLE_CSV = PERSONALIZATION_DIR / "l2_active_effect_cycles_v1.csv"
L3_AUDIT_JSON = PERSONALIZATION_DIR / "l3_active_effect_audit_v1.json"
L3_AUDIT_CYCLE_CSV = PERSONALIZATION_DIR / "l3_active_effect_cycles_v1.csv"
L3_AUDIT_ADJUST_DAY_CSV = PERSONALIZATION_DIR / "l3_active_effect_adjust_days_v1.csv"

MATRIX_PROFILE_CSV = PERSONALIZATION_DIR / "personalization_matrix_histphys_refine_profile_v1.csv"
MATRIX_RESULTS_CSV = PERSONALIZATION_DIR / "personalization_matrix_results_v1.csv"
MATRIX_RESULTS_MD = PERSONALIZATION_DIR / "personalization_matrix_results_v1.md"
MATRIX_MANIFEST_JSON = MANIFEST_DIR / "personalization_matrix_manifest_v1.json"
MATRIX_AUDIT_JSON = PERSONALIZATION_DIR / "personalization_matrix_detect_audit_v1.json"


def ensure_research_dirs() -> None:
    for path in (
        DOCS_DIR,
        METHODOLOGY_DIR,
        REPORTS_DIR,
        RESULTS_DIR,
        SUBGROUP_DIR,
        MANIFEST_DIR,
        BASELINE_DIR,
        PERSONALIZATION_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
