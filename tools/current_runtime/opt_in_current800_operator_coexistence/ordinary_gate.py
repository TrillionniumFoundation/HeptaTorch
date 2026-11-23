"""Offline verifier for the sealed #71 ordinary conformance capture."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import tempfile

FILES = {
    "fixtures_v3.json": "08ab226b27f729c544adb42365ed4629414561575f40ac6cba0bbc4a4397cc4b",
    "receipt.json": "cddbbd855bd10a26d681a1b956625d6ca3fb0e01d3d8765a702f1149f8603cc3",
    "pinned-manifest.json": "5e21a228eec25347b7c737ab9862897153a69c3bf0b471c2d8a71a89f2a6745b",
    "readback_v3.py": "1b686f69ace8f711796ef235286567f6e667699f676020f824cedb386c608673",
    "ordinary-readback.json": "0da1726261af635bee98310783ae3fe0568640a4f242deb0200c581521ee9c8a",
}
ARCHIVE_SHA = "01d79608824cc6ae5962ee3cfffb8739d1b30428b84cc4b62f4e31697b32707a"
ARCHIVE_BYTES = 1654860
IMAGES = {
    "combined": "5c8dfb2d329e211e7871438d64754b0f8912cc687c1d9ae3fc432f20e95d165d",
    "where": "2ee8c061f44e0496321141992881ca59557a22ccb7782e5b7cf1a1a313fce8f6",
    "add": "31a267d82b274ae81a956a03f8da1c9eb5a28bbcc903f78e56744333d1a88d4d",
}
LANES = {
    "combined": (50, 171, 1340, 1362, 768, 27),
    "where": (14, 14, 224, 173, 42, 0),
    "add": (50, 157, 1172, 1189, 628, 27),
}
FIELDS = ("cases", "ordinary_outputs", "frames", "words",
          "untouched_live_snapshots", "rank2_different_raw_order")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_capped(path, maximum):
    with path.open("rb") as handle:
        data = handle.read(maximum + 1)
    require(len(data) <= maximum, f"{path.name}: exceeds read bound")
    return data


def load_json(raw):
    def unique(pairs):
        obj = {}
        for key, val in pairs:
            require(key not in obj, f"duplicate JSON key: {key}")
            obj[key] = val
        return obj

    def constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=constant)


def verify_summary(data):
    p = load_json(data)
    require(isinstance(p, dict) and p.get("status") == "PASS", "readback did not pass")
    require(p.get("scope") == "reviewed ordinary v3 numerical fixture only", "unexpected scope")
    lanes = p.get("metrics")
    require(isinstance(lanes, dict) and lanes.keys() == LANES.keys(), "wrong lane set")
    for lane, expected in LANES.items():
        row = lanes[lane]
        require(isinstance(row, dict) and row.get("image") == IMAGES[lane], f"{lane}: wrong image")
        require(all(type(row.get(name)) is int and row[name] == count
                    for name, count in zip(FIELDS, expected, strict=True)), f"{lane}: wrong counters")
    return p


def materialize_ordinary(archive, root):
    lane_names = "|".join(LANES)
    data_leaf = re.compile(
        rf"ordinary/(?:{lane_names})/(?:report\.json|requests\.frames\.bin|responses\.frames\.bin|owners/[0-9a-f]{{64}}\.zlib)\Z"
    )
    seen = set()
    total = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as entries:
        for member in entries:
            if not data_leaf.fullmatch(member.name):
                continue
            require(member.isfile() and member.name not in seen, "invalid ordinary archive member")
            require(0 < member.size <= 4 * 1024 * 1024, "invalid ordinary member size")
            require(len(seen) < 1024, "too many ordinary archive members")
            total += member.size
            require(total <= 16 * 1024 * 1024, "ordinary archive content exceeds bound")
            seen.add(member.name)
            with entries.extractfile(member) as handle:
                raw = handle.read(member.size + 1)
            require(len(raw) == member.size, "ordinary archive member size mismatch")
            dest = root / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as handle:
                handle.write(raw)
    required = {f"ordinary/{lane}/{leaf}" for lane in LANES
                for leaf in ("report.json", "requests.frames.bin", "responses.frames.bin")}
    require(required <= seen, "ordinary archive is incomplete")


def gate(archive_path, bundle):
    require(__debug__, "run without Python optimization; readback uses assertions")
    sealed = {}
    for name, checksum in FILES.items():
        blob = read_capped(bundle / name, 1024 * 1024)
        require(digest(blob) == checksum, f"{name}: wrong sealed bytes")
        sealed[name] = blob
    fixture = load_json(sealed["fixtures_v3.json"])
    require(fixture["schema"] == "heptatorch.current800.coexistence.numerical.v1", "wrong fixture schema")
    require(len(fixture["cases"]) == 50 and len(fixture["new_where_cases"]) == 14
            and sum(len(c["expect"]) for c in fixture["cases"]) == 171, "wrong fixture counts")
    locked = load_json(sealed["receipt.json"])
    require(locked["status"] == "PASS" and locked["mismatches"] == []
            and locked["fixture_sha256"] == FILES["fixtures_v3.json"]
            and locked["output_checks"] == 342, "wrong locked receipt")
    manifest = load_json(sealed["pinned-manifest.json"])
    require(manifest["expected_native_sha256"] == IMAGES["combined"], "wrong manifest image")
    verify_summary(sealed["ordinary-readback.json"])
    archive = read_capped(archive_path, ARCHIVE_BYTES)
    require(len(archive) == ARCHIVE_BYTES and digest(archive) == ARCHIVE_SHA,
            "archive is not the sealed #71 ordinary capture")
    with tempfile.TemporaryDirectory(prefix="heptatorch-ordinary-") as directory:
        root = Path(directory)
        materialize_ordinary(archive, root)
        for name in ("fixtures_v3.json", "receipt.json"):
            (root / name).write_bytes(sealed[name])
        args = sys.argv
        capture = io.StringIO()
        try:
            sys.argv = ["readback_v3.py", str(root / "ordinary"),
                        str(root / "fixtures_v3.json"), str(root / "receipt.json")]
            with contextlib.redirect_stdout(capture):
                exec(compile(sealed["readback_v3.py"], "readback_v3.py", "exec"),
                     {"__name__": "__main__"})
        finally:
            sys.argv = args
    result = capture.getvalue().encode("utf-8")
    require(result == sealed["ordinary-readback.json"], "sealed ordinary readback differs")
    verify_summary(result)
    return {"status": "PASS_OFFLINE_ORDINARY_REPLAY_ONLY", "archive_sha256": ARCHIVE_SHA,
            "readback_sha256": digest(result), "fixture_sha256": FILES["fixtures_v3.json"],
            "receipt_sha256": FILES["receipt.json"], "combined_outputs": 171,
            "isolated_outputs": 171, "new_native_execution": False, "new_torch_execution": False,
            "default_qualified": False, "hosted_qualified": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, required=True, help="explicit local sealed #71 raw archive")
    args = ap.parse_args()
    try:
        print(json.dumps(gate(args.archive, Path(__file__).resolve().parent), sort_keys=True))
    except (ValueError, OSError, tarfile.TarError, KeyError, TypeError, AssertionError) as error:
        print(f"ordinary evidence not verified: {error or type(error).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
