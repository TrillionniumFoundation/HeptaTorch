"""Inject only a recorded transport into the immutable native observer.

The original Native.__init__ exchange wrapper, Memory, SumCase, parent Case,
and all owner predicates are imported byte-for-byte from the frozen source03
package. capture_owner supplies its actual archived bytes and finish admits the
original transport EOF. Neither method obtains a new process observation.
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
from admission_common import SHA, need, read, same, write
from recorded_wire import RecordedSession
from control_owners import owners_for_record as CompleteOwners

class RecordedObserver:
    def __init__(self, original, owner_module, protocol_path, runtime_path, binary_path):
        self.original, self.owner_module = original, owner_module
        self.protocol_path = Path(protocol_path).resolve()
        self.runtime_path, self.binary_path = Path(runtime_path).resolve(), Path(binary_path).resolve()
        self.wire = original.wire(self.protocol_path)
        self.active = None
        context = self

        def recorded_session(executable, *, directory):
            need(context.active is not None, "one-explicit-recorded-case-context")
            active = context.active
            need("session" not in active, "one-recorded-session-per-case")
            need(Path(executable).resolve() == context.binary_path
                 and Path(directory) == active["output"]/"raw",
                 "original-observer-requested-exact-bound-image-and-case-output")
            Path(directory).mkdir(parents=True, exist_ok=False)
            session = RecordedSession(active["original"]/"raw", context.wire, active["report"]["frames"])
            session.executable = context.binary_path
            owners = CompleteOwners(active["original"], active["report"], session, owner_module)
            session.owner_source = owners
            active.update(session=session, owners=owners)
            return session

        def recorded_wire(path):
            need(Path(path).resolve() == context.protocol_path,
                 "unchanged-explicit-original-protocol-path")
            return context.wire

        # Only the constructor target changes. Its original packet/parse/meta
        # functions and every 8Q generated command remain unchanged.
        self.wire.Session = recorded_session
        original.wire = recorded_wire

        class RecordedNative(original.Native):
            def __init__(self, binary, source, protocol, output):
                need(context.active is not None and Path(output) == context.active["output"]
                     and Path(binary).resolve() == context.binary_path
                     and Path(source).resolve() == context.runtime_path
                     and Path(protocol).resolve() == context.protocol_path,
                     "all-original-Native-construction-arguments-bound")
                super().__init__(binary, source, protocol, output)
                context.active["native"] = self

            def capture_owner(self, label):
                owners = self.s.owner_source
                snapshot, reference, raw_bytes = owners.capture(label, self.s.frames)
                digest = reference["sha256"]
                if digest not in self.owner_blobs:
                    self.owner_blobs[digest] = copy.deepcopy(owners.blobs[digest])
                return snapshot, reference, raw_bytes

            def finish(self):
                if getattr(self, "_finished", False):
                    return
                self._finished = True
                # This exit code and the two EOF facts are the observed native
                # values from the sealed original transport, never a new child.
                self.s.finish()
                owners = self.s.owner_source
                same(self.boundaries, owners.journal,
                     "every-original-full-boundary-result-recomputed")
                same(self.owner_blobs, owners.blobs, "all-original-complete-owner-blobs-consumed")
                need(self.frame_journal is not None and not self.frame_journal.closed,
                     "fresh-pure-owner-journal-present")
                self.frame_journal.close()
                generated = (self.observation_output/"owner-frames.jsonl").read_bytes()
                original_raw = read(owners.directory/"owner-frames.jsonl")
                need(generated == original_raw, "complete-original-owner-journal-byte-identical")
                need(SHA(read(self.s.executable)) == self.s.binary_sha256,
                     "bound-ELF-remained-data-only-and-unchanged")
                write(self.observation_output/"raw"/"transport.json", self.s.transport)
                context.active["owner_admission"] = self.s.owner_finish_summary
                context.active["finished"] = True

        values = dict(vars(original))
        values["Native"] = RecordedNative
        self.facade = SimpleNamespace(**values)

    def begin(self, original_directory, original_report, output):
        need(self.active is None or self.active.get("finished") is True,
             "previous-recorded-case-complete-before-next")
        self.active = {"original": Path(original_directory), "report": original_report,
                       "output": Path(output), "finished": False}
        return self.facade

    def completed(self):
        active = self.active
        need(active is not None and active.get("finished") is True,
             "recorded-case-transport-finished")
        return {"owner_admission": active["owner_admission"],
                "generated_boundary_count": len(active["native"].boundaries),
                "original_transport_exit": active["session"].exit_code,
                "native_executed_here": False, "new_memory_observation": False}
