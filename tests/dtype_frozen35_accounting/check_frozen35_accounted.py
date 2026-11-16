"""Bind the additive observer report to the unchanged original 35-case report."""
from pathlib import Path
import argparse, hashlib, json
PROFILES={
 "830f0b4ef8d6ad8c6a105be56d383628b45c84e3e43905fee6812c7c52d35a51":
  ("dtype-native-dropout-523","6ffe509afce28ff15705b4c7b269eaef8f8f89f9d8b2486d4d9fa45b6be66ef3"),
}
CHECKS={"python_matches","platform_matches","version_backend_matches",
 "git_matches_oracle_build_pin","distribution_version_matches","direct_url_matches",
 "direct_url_archive_sha256_matches","pip_report_url_matches","pip_report_archive_sha256_matches",
 "pip_report_version_matches","pip_report_requested"}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--observer-report",type=Path,required=True)
 p.add_argument("--original-report",type=Path,required=True)
 a=p.parse_args()
 raw=a.original_report.read_bytes(); inner=json.loads(raw)
 wrapper=json.loads(a.observer_report.read_bytes())
 assert wrapper["schema"]=="heptatorch.frozen35-private-node-accounted-run.v1"
 assert wrapper["status"]=="PASS" and wrapper["original_entry_return_code"]==0 and wrapper["exception"] is None
 bound=wrapper["original_oracle_report"]
 assert bound["sha256"]==sha(raw) and bound["bytes"]==len(raw)
 assert Path(bound["path"]).resolve()==a.original_report.resolve()
 assert inner["status"]==bound["status"]=="PASS" and inner["failed_cases"]==bound["failed_cases"]==0
 assert inner["cases"]==inner["expected_cases"]==bound["cases"]==len(inner["rows"])==35
 assert inner["groups"]["unchanged_numeric_28"]["cases"]==inner["groups"]["unchanged_numeric_28"]["passed"]==28
 assert inner["groups"]["additional_object_lifetime_7"]["cases"]==inner["groups"]["additional_object_lifetime_7"]["passed"]==7
 assert all(r["status"]=="PASS" for r in inner["rows"])
 # Frozen workload expectations; the original EOF/Version gates add no frames or values.
 assert inner["frames"]==6344 and inner["compared_values"]==2590274
 assert inner["explicit_state_absolute_version_differences"]==283
 assert inner["direct_parameter_absolute_version_mismatches"]==0
 provenance=inner["oracle_provenance"]
 assert provenance["status"]=="PASS" and set(provenance["checks"])==CHECKS
 assert all(v is True for v in provenance["checks"].values())
 assert inner["native_unchanged_during_run"] is True
 assert inner["source_files_unchanged_during_run"] and all(v is True for v in inner["source_files_unchanged_during_run"].values())
 assert wrapper["source_files_unchanged_during_run"] and all(v is True for v in wrapper["source_files_unchanged_during_run"].values())
 assert inner["native_executable_sha256"]==wrapper["profile"]["native_sha256"]
 expected_name,expected_source=PROFILES[inner["native_executable_sha256"]]
 assert wrapper["profile"]["name"]==expected_name and wrapper["profile"]["source_sha256"]==expected_source
 assert wrapper["source_custody"]["native_image"]["sha256"]==inner["native_executable_sha256"]
 assert wrapper["source_custody"]["native_source"]["sha256"]==expected_source
 assert wrapper["source_custody"]["observer"]["sha256"]=="4cd9c3d1fafd1eee04ac1b2b8bb720a099d53dc821b25b9cf4380e6c67b36a71"
 assert wrapper["source_custody"]["launcher"]["sha256"]=="1dca385159d332ed75adb57c1173b6a0b010597adabf5fe38001d97d1ea769d2"
 assert wrapper["source_custody"]["legacy_owner_rules"]["sha256"]=="281f1ea437046a9cd884de2c8da34f38649a3b6482ade23123ac4aa992faed64"
 assert wrapper["source_custody"]["structural_owner640"]["sha256"]=="f503df2d1dbce8242591e77211b96a7787201e991db8186c9f1cca6576244409"
 assert wrapper["source_custody"]["original_structural_owner640"]["sha256"]=="6ea9e8b33164089034a2a71a03f4850e1ec18ed1a09155d7841f700ef311c833"
 assert wrapper["source_custody"]["storage_tombstone_correction"]["sha256"]=="59ece06f7fef4736dd835f78bd6159c055a30f4d91f5ebd579bf109aae3c3577"
 assert wrapper["source_custody"]["accounting_profile"]["sha256"]=="83f673c3f1ff4fe3fa7a2f35749326ff8eeeaa60df570da080bc82d8712e683b"
 assert wrapper["profile"]["descriptor_bytes"]==640
 assert wrapper["profile"]["source_constructor_costs"] is True
 inverse=wrapper["profile"]["inverse_proof"]
 assert inverse["legacy_source_sha256"]==wrapper["source_custody"]["legacy_owner_rules"]["sha256"]
 assert inverse["structural_source_sha256"]==wrapper["source_custody"]["structural_owner640"]["sha256"]
 assert inverse["full_640_snapshot_no_projection"] is True
 assert set(inverse["functions"])=={"sha","canonical","need","word","identity","semantic_snapshot","verify_snapshot","verify_transition","Journal","install"}
 assert all(x["inverse_entire_function_AST_equals_original"] is True for x in inverse["functions"].values())
 assert sum(x["width_predicates_changed"] for x in inverse["functions"].values())==1
 assert inverse["functions"]["verify_snapshot"]["width_predicates_changed"]==1
 assert all(x["original_ast_sha256"]==x["executed_ast_sha256"] for name,x in inverse["functions"].items() if name not in ("verify_snapshot","verify_transition"))
 assert inner["frames"]==bound["frames"]==sum(r["frames"] for r in inner["rows"])
 assert inner["compared_values"]==bound["compared_values"]==sum(r["compared_values"] for r in inner["rows"])
 assert inner["direct_parameter_absolute_version_mismatches"]==sum(len(r.get("parameter_version_mismatches",[])) for r in inner["rows"])
 assert inner["explicit_state_absolute_version_differences"]==sum(len(r.get("explicit_state_version_differences",[])) for r in inner["rows"])
 assert [(r["case"],r["rtol"],r["atol"]) for r in inner["rows"]]==[(r["case"],r["rtol"],r["atol"]) for r in inner["case_plan"]]
 assert inner["source_custody"]["v3_entry"]["sha256"]=="149b41c455444edeb8d25d0ad98045e29683e780f37f7e210e7985242d359597"
 assert inner["source_custody"]["v3_bridge"]["sha256"]=="c2888687f8876ad3505f0e13c9aecddc45695eff22bfdd9fa16d760f9aa3dac5"
 assert inner["source_custody"]["oracle_lock"]["sha256"]=="ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944"
 overlay=wrapper["observer_overlay"]
 assert overlay["runtime_replacements"]==["bridge.install"]
 assert overlay["original_bridge_close_assertion_replaced"]=="stats[11] == self.next_id-1"
 assert overlay["inverse_runtime_restoration_checked_after_execution"] is True
 assert overlay["no_native_memory_writes"] is True
 assert wrapper["numeric_oracle_rows_reclassified_by_wrapper"] is False
 assert wrapper["all_original_EOF_and_post_close_Version_gates_executed"] is True
 assert len(wrapper["observer_cases"])==35
 for row,case in zip(inner["rows"],wrapper["observer_cases"]):
  observation=row["private_node_observer"]
  assert case==dict(case=row["case"],**observation)
  assert observation["status"]=="PASS"
  assert observation["eof_exit_assertions_executed"] is True
  assert observation["post_close_parameter_version_assertions_executed"] is True
 assert inner["full_rewrite_complete"] is wrapper["full_rewrite_complete"] is False
 assert inner["pytorch_api_denominator"]==wrapper["pytorch_api_denominator"]==2590
 print(json.dumps({"status":"PASS_EXACT_ORIGINAL_AND_ADDITIVE_REPORT_BINDING","cases":35,
  "oracle_provenance_checks":11,"original_report_sha256":sha(raw),
  "observer_report_sha256":sha(a.observer_report.read_bytes()),
  "preserved_explicit_state_version_differences":inner["explicit_state_absolute_version_differences"],
  "full_rewrite_complete":False}))
if __name__=="__main__":main()
