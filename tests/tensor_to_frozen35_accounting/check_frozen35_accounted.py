"""Bind the additive observer report to the unchanged original 35-case report."""
from pathlib import Path
import argparse, hashlib, json
PROFILES={
 "53fb43b869858f020c0b936a65f00f3b14b97d9e37038a74c357e1da471fe083":
  ("tensor-to-copy-614-fixed35-F32-Saved","4f8f73f463f681b5eb29423d71a35a0b42d807fd92bf976bbf76c1970fbcee33"),
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
 assert wrapper["source_custody"]["observer"]["sha256"]=="a496aae3ca0b9f02b7f0e8c26c7177c3ce5b5a8c4198354c22059434b1491914"
 assert wrapper["source_custody"]["launcher"]["sha256"]=="45b635448e7c4d2db48cbc3a9b31e722eaf3ec93be7e252ef0c51c740d04d949"
 assert wrapper["source_custody"]["legacy_owner_rules"]["sha256"]=="281f1ea437046a9cd884de2c8da34f38649a3b6482ade23123ac4aa992faed64"
 assert wrapper["source_custody"]["structural_owner672"]["sha256"]=="e58adb23bb65cdc81ec36e3e2fcbd178c0d5198f3b10c5d274afa94da7761a6f"
 assert wrapper["source_custody"]["original_structural_owner672"]["sha256"]=="6d230e2df25e04e32f76fb10fd0936ec9d220b60acda37c40b52a45633372312"
 assert wrapper["source_custody"]["storage_tombstone_correction"]["sha256"]=="59ece06f7fef4736dd835f78bd6159c055a30f4d91f5ebd579bf109aae3c3577"
 assert wrapper["source_custody"]["accounting_profile"]["sha256"]=="1e5e8db18a2e9a7bc4b98ac1063a427eb0e60613781962371dacafc76414529f"
 assert wrapper["profile"]["descriptor_bytes"]==672
 assert wrapper["profile"]["source_constructor_costs"] is True
 inverse=wrapper["profile"]["inverse_proof"]
 assert inverse["legacy_source_sha256"]==wrapper["source_custody"]["legacy_owner_rules"]["sha256"]
 assert inverse["structural_source_sha256"]==wrapper["source_custody"]["structural_owner672"]["sha256"]
 assert inverse["full_672_snapshot_no_projection"] is True
 assert set(inverse["functions"])=={"sha","canonical","need","word","identity","semantic_snapshot","verify_snapshot","verify_transition","Journal","install"}
 assert all(x["inverse_entire_function_AST_equals_original"] is True for x in inverse["functions"].values())
 assert sum(x["width_predicates_changed"] for x in inverse["functions"].values())==1
 assert inverse["functions"]["verify_snapshot"]["width_predicates_changed"]==1
 assert [e["label"] for e in inverse["functions"]["verify_snapshot"]["explicit_role_edits"]]==["width","actual-original-T-pair-counter","dead-node-no-Saved-root","rowless-no-owned-T-root","reference-sum-from-pairs","exact600-from-pairs","reachable-original-T-roots"]
 assert [e["label"] for e in inverse["functions"]["verify_transition"]["explicit_role_edits"]]==["reset-original-T-root-closure"]
 assert all(not e["explicit_role_edits"] for n,e in inverse["functions"].items() if n not in ("verify_snapshot","verify_transition"))
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
