"""Bind the additive observer report to the unchanged original 35-case report."""
from pathlib import Path
import argparse, hashlib, json
PROFILES={
 "be18ab877f38a4027a527783db80a1418a137c7d6da91e5bc39e2af8c044cb6b":
  ("gradient-objects-355","1b18da2cf9791e1b20982f9706434c1359ea194dd983dc6c66eb005f152e77b5"),
 "6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe":
  ("gradient-adam-distributions-428","1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae"),
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
 assert wrapper["source_custody"]["observer"]["sha256"]=="e900466bff13bcaa5cd89a814dc411b4fc33320d84855ca6bd085179cd599c6a"
 assert wrapper["source_custody"]["launcher"]["sha256"]=="65c0d68725f0585b79f88ceb5fa106526a44805eca0b58535f32b5b29916cd12"
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
