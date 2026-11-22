"""Reviewed constructor source transition for the original35 replay on owner800.

Never select a transition from observed digests alone. The owner must first
pass full same-manifest cold-build admission. Historical body pins remain the
authority for every unchanged constructor and for the entire inverse edit.
"""
from binding import require, sha

OWNER800 = (
    "aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c",
    "083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5",
)
HELPERS = {
    "linear_bias_output_version": "3bed2096862db6a53b7614a9feaa88d2a81ae8396ae2c01d4a2fe6651bfe46eb",
    "normalization_output_layout": "4405e11c2a1d41fe705a391f680c0b5f8406c005a280787ce512a0916fc5df1d",
}
EDITS = {
    "linear": (
        (b"  branch v145 b16(v134,v137,v144) b18(v134,v144)\nblock b16(v146:mutbuf,v149:u64,v154:u64)\n",
         b"  branch v145 b16(v134,v135,v137,v144) b18(v134,v144)\nblock b16(v146:mutbuf,v147:u64,v149:u64,v154:u64)\n"),
        (b"  jump b18(v146,v165)\n",
         b'  linear_version_result:u64 = call(v146,v165,v147,v149) {"function":"linear_bias_output_version"}\n'
         b"  jump b18(v146,linear_version_result)\n"),
    ),
    "layernorm": (
        (b"  ret(v313)\n",
         b'  v314:u64 = const() {"value":0}\n'
         b'  v315:u64 = call(v298,v313,v314) {"function":"normalization_output_layout"}\n'
         b"  ret(v315)\n"),
    ),
    "batchnorm_eval": (
        (b"  ret(v459)\n",
         b'  v460:u64 = const() {"value":1}\n'
         b'  v461:u64 = call(v443,v459,v460) {"function":"normalization_output_layout"}\n'
         b"  ret(v461)\n"),
    ),
}


def checked_transition(build, bodies, published_pins, constructor_names):
    """Return observed pins plus an explicit inverse receipt, or fail closed."""
    pins, changed = {}, {}
    reviewed_image = (build["source_sha256"], build["native_sha256"]) == OWNER800
    for name in constructor_names:
        expected = published_pins[name]
        message = "constructor or layout contract changed: " + name
        require(name in bodies, message)
        body = bodies[name]
        actual = sha(body)
        pins[name] = actual
        if actual == expected:
            continue
        require(reviewed_image and name in EDITS, message)
        inverse = body
        for before, after in reversed(EDITS[name]):
            require(inverse.count(after) == 1, message)
            inverse = inverse.replace(after, before, 1)
        require(sha(inverse) == expected, message)
        changed[name] = dict(observed_sha256=actual, inverse_sha256=expected,
                             exact_whole_body_inverse=True)
    if not reviewed_image:
        return pins, None
    require(set(changed) == set(EDITS), "all three reviewed owner800 changes are required")
    for name, expected in HELPERS.items():
        require(name in bodies and sha(bodies[name]) == expected, "changed constructor helper: " + name)
    return pins, dict(constructor_reverse_edits=changed, observed_helper_pins=HELPERS.copy(),
                      previously_qualified_runtime_result_transferred=False)


def observe_frozen_domain(rows, journal_root, inputs):
    """Bind all actual journals; do not reuse constructor counts for a broader workload."""
    import json
    from pathlib import Path

    expected = {15: 48, 26: 0, 28: 0}
    counts = dict.fromkeys(expected, 0)
    journals = set()
    total_frames = 0
    for row in rows:
        observation = row["private_node_observer"]
        directory = Path(observation["snapshot_directory"])
        require(directory.parent == journal_root and not directory.is_symlink()
                and directory not in journals, "exact distinct current35 owner journals")
        events = json.loads(inputs.bind(directory / "events.json"))
        require(isinstance(events, list) and len(events) == observation["frames_checked"],
                "all journal records must match the observed owner frames")
        for event in events:
            require(event.get("error") is None, "no failed owner observation")
            for command in event["commands"]:
                if command[0] in counts:
                    counts[command[0]] += 1
        journals.add(directory)
        total_frames += len(events)
    require(len(journals) == 35 and total_frames == 6344 and counts == expected,
            "owner800 reviewed original35 command scope changed")
    return dict(journals=len(journals), frames=total_frames,
                linear_commands=counts[15], layernorm_commands=counts[26], batchnorm_eval_commands=counts[28],
                normalization_allocation_qualified=False)
