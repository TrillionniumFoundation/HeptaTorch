"""New, explicit Bool owner rule; frozen I64 rules remain unchanged."""
from common import adapted_parent, require


def make_observer(repository, inputs, source_sha, image_sha):
    owner, owner_proof = adapted_parent(repository, "owner_integral.py", inputs, owner=True)
    observer, observer_proof = adapted_parent(repository, "native_observer_integral.py", inputs)
    original_metadata = owner._command_metadata
    original_frame = owner.verify_frame_boundary

    def bool_metadata(before, after, command, record, body):
        if command[0] != 115 or command[5] != 1:
            return original_metadata(before, after, command, record, body)
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
        # Canonical output bytes are checked against the actual Storage and
        # typed readback by the native runner. Version is compared to Torch.
        return dict(operation=115, result=result, source_dtype=w(sn, 576), result_dtype=1,
                    element_size=1, storage_requested_bytes=storage[3], requires_grad=False,
                    autograd_operation=0, fresh_Tensor_Storage_Version=True,
                    output_version_is_observed_not_assumed=True)

    def frame(before, after, commands, status, records, body, **kwargs):
        result = original_frame(before, after, commands, status, records, body, **kwargs)
        # This suite has no random-producing command. Its seed/create/drop
        # operations are the only changes to the generator table itself.
        if not any(c[0] in (60, 66) for c in commands):
            owner.need(before["generator_states"] == after["generator_states"],
                       "bool-suite-complete5056-RNG-unchanged")
        return {**result, "complete_RNG_checked": not any(c[0] in (60, 66) for c in commands)}

    owner._command_metadata = bool_metadata
    owner.verify_frame_boundary = frame
    observer.configure(owner, source_sha, image_sha, 700)
    require(owner.SOURCE == source_sha and owner.IMAGE == image_sha, "new-bool-profile-bound")
    return observer, owner, dict(owner=owner_proof, observer=observer_proof,
                                old_integral_width8_contract_preserved=True,
                                new_bool_width1_contract="owner_adapter.bool_metadata")
