"""Snapshot a previously installed abi3 build without changing its source environment."""
import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dist = importlib.metadata.distribution("vllm")
    if dist.version != "0.19.1":
        raise RuntimeError(f"Unexpected installed vllm {dist.version}")
    root = Path(dist.locate_file("")).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    wheel = out / "vllm-0.19.1-cp38-abi3-linux_x86_64.whl"
    record_path = "vllm-0.19.1.dist-info/RECORD"
    rows, sources = [], []
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_STORED) as archive:
        for relative in sorted(dist.files or [], key=str):
            name = str(relative)
            path = Path(dist.locate_file(relative)).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                continue
            if name == record_path or name.endswith(".pyc") or "__pycache__" in name:
                continue
            if name.endswith("/direct_url.json") or name.endswith("/INSTALLER"):
                continue
            body = path.read_bytes()
            digest = hashlib.sha256(body).digest()
            encoded = base64.urlsafe_b64encode(digest).decode().rstrip("=")
            archive.writestr(name, body)
            rows.append([name, "sha256=" + encoded, len(body)])
            sources.append({"file": name, "bytes": len(body), "sha256": digest.hex()})
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerows(rows + [[record_path, "", ""]])
        archive.writestr(record_path, stream.getvalue())
    h = hashlib.sha256()
    with wheel.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    provenance = {"source": str(root), "version": dist.version,
                  "wheel": str(wheel), "wheel_sha256": h.hexdigest(),
                  "source_files": sources, "method": "installed-abi3-build-snapshot",
                  "requires_runtime_compatibility_validation": True}
    (out / "vllm_artifact.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in provenance.items() if k != "source_files"}))


if __name__ == "__main__":
    main()
