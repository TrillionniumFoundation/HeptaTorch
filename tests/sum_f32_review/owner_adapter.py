"""Additional exact width-four and temporary-cast owner rules for Sum F32."""
from native_support import adapted_parent, require


def make_observer(repository, inputs, source_sha, image_sha):
    owner, owner_proof = adapted_parent(repository, "owner_integral.py", inputs, owner=True)
    observer, observer_proof = adapted_parent(repository, "native_observer_integral.py", inputs)
    original_metadata = owner._command_metadata
    original_frame = owner.verify_frame_boundary

    def live_rows(snapshot, kind):
        return {owner.token(kind, slot, row): row for slot, row in
                enumerate(snapshot["tables"][str(kind)], 1) if row[1]}

    def bool_metadata(before, after, command, record, body):
        # This is the complete accepted Bool700 predicate, preserved here for
        # its independent one-byte result contract on the new image.
        w, need = owner.w, owner.need
        need(record["opcode"] == 115 and record["payload"] == b"", "bool115-record")
        source = owner.public_node(before, command[1])
        result = owner.public_node(after, record["value"])
        sn, rn = before["nodes"][str(source)], after["nodes"][str(result)]
        need(w(sn, 576) in (1, 2) and w(rn, 576) == 1, "bool115-explicit-supported-dtype")
        need(source != result and w(sn, 320) != w(rn, 320), "bool115-fresh-Tensor")
        need(w(sn, 32) == 0 and all(w(rn, field) == 0 for field in
             (32, 40, 48, 56, 72, 80, 88, 176, 184, 280, 288, 296,
              312, 360, 384, 392, 400, 408, 584, 592, 600, 608, 616, 624,
              640, 648, 656, 664)), "bool115-no-grad-Saved-view-or-input-edge")
        need(after["nodes"][str(source)] == sn, "bool115-source-all84words-unchanged")
        need(w(rn, 216) != w(sn, 216) and w(rn, 224) != w(sn, 224), "bool115-fresh-Storage-Version")
        storage = owner._live_row(after, 2, w(rn, 216), "bool115-real-Storage")
        need(storage[3] == w(rn, 8) and w(rn, 232) == 0, "bool115-one-byte-requested-Storage")
        owner._live_row(after, 3, w(rn, 224), "bool115-real-Version")
        return dict(operation=115, result=result, source_dtype=w(sn, 576), result_dtype=1,
                    element_size=1, storage_requested_bytes=storage[3], requires_grad=False,
                    autograd_operation=0, fresh_Tensor_Storage_Version=True,
                    output_version_is_observed_not_assumed=True)

    def float32_metadata(before, after, command, record, body):
        w, need = owner.w, owner.need
        need(record["opcode"] == 115 and record["payload"] == b"" and command[5] == 0,
             "integral-F32-115-record")
        source = owner.public_node(before, command[1])
        result = owner.public_node(after, record["value"])
        sn, rn = before["nodes"][str(source)], after["nodes"][str(result)]
        need(w(sn, 576) in (1, 2) and w(rn, 576) == 0, "integral-F32-supported-dtypes")
        need(source != result and w(sn, 320) != w(rn, 320), "integral-F32-fresh-Tensor")
        need(w(sn, 32) == 0 and all(w(rn, field) == 0 for field in
             (32, 40, 48, 56, 72, 80, 88, 176, 184, 280, 288, 296,
              312, 360, 384, 392, 400, 408, 584, 592, 600, 608, 616, 624,
              640, 648, 656, 664)), "integral-F32-no-grad-Saved-view-or-input-edge")
        need(w(rn, 216) != w(sn, 216) and w(rn, 224) != w(sn, 224),
             "integral-F32-fresh-Storage-Version")
        storage = owner._live_row(after, 2, w(rn, 216), "integral-F32-real-Storage")
        owner._live_row(after, 3, w(rn, 224), "integral-F32-real-Version")
        need(storage[3] == 4 * w(rn, 8) and w(rn, 232) == 0,
             "integral-F32-exact-four-byte-Storage")
        need(before["header"]["next"] + 2 == after["header"]["next"]
             and before["header"]["incarnation"] + 2 == after["header"]["incarnation"],
             "integral-F32-one-private-cast-one-result-descriptor")
        temporary = before["header"]["next"]
        need(result == temporary + 1 and str(temporary) in after["nodes"],
             "integral-F32-exact-cast-then-result-order")
        dead = after["nodes"][str(temporary)]
        need(all(w(dead, field) == 0 for field in (32, 40, 48, 56, 72, 80, 88, 200, 208,
             240, 280, 288, 296, 304, 312, 360, 384, 392, 400, 408, 584, 592, 600,
             608, 616, 624, 640, 648, 656, 664)), "integral-F32-private-cast-fully-dead")
        need(before["pages"] == after["pages"], "integral-F32-test-descriptor-page-preexists")
        for name, old in before["nodes"].items():
            need(after["nodes"].get(name) == old, "integral-F32-all-existing-descriptor-words-unchanged", node=name)
        expected_new = {1: {w(rn, 192)}, 2: {w(rn, 216)}, 3: {w(rn, 224)}}
        changes = {}
        for kind in range(1, 9):
            old, new = live_rows(before, kind), live_rows(after, kind)
            need(set(new) - set(old) == expected_new.get(kind, set()),
                 "integral-F32-only-result-live-records", kind=kind)
            need(not (set(old) - set(new)), "integral-F32-no-existing-record-released", kind=kind)
            for handle in old:
                need(new[handle] == old[handle], "integral-F32-existing-record-unchanged", kind=kind, handle=handle)
            changes[str(kind)] = sorted(set(new) - set(old))
        need(after["edges"][str(result)] == [] and str(temporary) not in after["edges"],
             "integral-F32-no-result-or-cast-edges")
        return dict(operation=115, result=result, source_dtype=w(sn, 576), result_dtype=0,
                    element_size=4, storage_requested_bytes=storage[3], requires_grad=False,
                    autograd_operation=0, fresh_Tensor_Storage_Version=True,
                    private_cast_node=temporary, private_cast_reclaimed=True,
                    all_existing_descriptor_words_unchanged=True, live_record_additions=changes,
                    output_version_is_observed_not_assumed=True)

    def metadata(before, after, command, record, body):
        if command[0] == 115:
            if command[5] == 1:
                return bool_metadata(before, after, command, record, body)
            if command[5] == 0:
                source = owner.public_node(before, command[1])
                if owner.w(before["nodes"][str(source)], 576) in (1, 2):
                    return float32_metadata(before, after, command, record, body)
        return original_metadata(before, after, command, record, body)

    def frame(before, after, commands, status, records, body, **kwargs):
        result = original_frame(before, after, commands, status, records, body, **kwargs)
        checked = not any(c[0] in (60, 66) for c in commands)
        if checked:
            owner.need(before["generator_states"] == after["generator_states"],
                       "integral-F32-suite-complete5056-RNG-unchanged")
        return {**result, "complete_RNG_checked": checked}

    owner._command_metadata = metadata
    owner.verify_frame_boundary = frame
    observer.configure(owner, source_sha, image_sha, 712)
    require(owner.SOURCE == source_sha and owner.IMAGE == image_sha, "new-integral-F32-profile-bound")
    return observer, owner, dict(owner=owner_proof, observer=observer_proof,
                                old_integral_width8_contract_preserved=True,
                                bool_width1_contract_preserved=True,
                                new_float32_width4_contract="owner_adapter.float32_metadata",
                                full_transition_recomputed=False)
