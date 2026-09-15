#!/usr/bin/env python3
"""Verify immutable B12X wheel release assets before publication or promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_BUNDLE_ASSETS = {
    "SHA256SUMS",
    "install.sh",
    "manifest.json",
    "requirements-github.txt",
    "runtime.lock",
}


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one release asset."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Reject a release whose declared invariant is not satisfied."""
    if not condition:
        raise ValueError(message)


def read_checksum_entries(path: Path) -> dict[str, str]:
    """Parse the GNU sha256sum file while rejecting duplicate entries."""
    entries: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.removeprefix("*")
        require(name not in entries, f"duplicate checksum entry: {name}")
        entries[name] = digest
    return entries


def main() -> None:
    """Validate all immutable beta or byte-identical promotion assets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--beta-tag", required=True)
    parser.add_argument("--promotion", action="store_true")
    parser.add_argument(
        "--reference-directory",
        type=Path,
        help="Independent flat asset directory from a fresh build or source beta",
    )
    args = parser.parse_args()

    require(args.directory.is_dir(), f"asset directory is missing: {args.directory}")
    manifest_path = args.directory / "manifest.json"
    require(manifest_path.is_file(), "manifest.json is missing")
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest.get("schema") == "local-inference-b12x-wheel-release/v1",
        "unexpected manifest schema",
    )
    require(
        manifest.get("source", {}).get("commit") == args.source_commit,
        "manifest source commit does not match",
    )
    require(
        manifest.get("release_tag") == args.beta_tag,
        "manifest release tag does not match",
    )
    packages = manifest.get("packages")
    require(
        isinstance(packages, list) and len(packages) == 1,
        "the B12X release must contain exactly one package",
    )
    require(packages[0].get("name") == "b12x", "the release package must be named b12x")

    expected_assets = set(REQUIRED_BUNDLE_ASSETS)
    expected_assets.update(package["file"] for package in packages)
    archive = f"b12x-cu134-{args.source_commit}.tar.zst"
    expected_assets.update({archive, f"{archive}.sha256"})
    if args.promotion:
        expected_assets.add("stable-promotion.json")
    actual_assets = {path.name for path in args.directory.iterdir() if path.is_file()}
    require(
        actual_assets == expected_assets,
        f"release asset set differs: expected {sorted(expected_assets)}, "
        f"found {sorted(actual_assets)}",
    )

    checksum_path = args.directory / "SHA256SUMS"
    checksum_entries = read_checksum_entries(checksum_path)
    expected_checksum_entries = {
        "manifest.json",
        "requirements-github.txt",
        "runtime.lock",
        "install.sh",
        *(f"wheels/{package['file']}" for package in packages),
    }
    require(
        set(checksum_entries) == expected_checksum_entries,
        "SHA256SUMS does not describe the complete bundle",
    )
    for name, digest in checksum_entries.items():
        asset_name = Path(name).name
        require(
            sha256(args.directory / asset_name) == digest,
            f"SHA256SUMS mismatch for {asset_name}",
        )

    for package in packages:
        wheel = args.directory / package["file"]
        require(wheel.is_file(), f"wheel is missing: {package['file']}")
        require(
            sha256(wheel) == package["sha256"],
            f"wheel digest mismatch: {package['file']}",
        )

    archive_checksum = (args.directory / f"{archive}.sha256").read_text().split()
    require(
        len(archive_checksum) == 2 and archive_checksum[1] == archive,
        "archive checksum does not identify the release archive",
    )
    require(
        sha256(args.directory / archive) == archive_checksum[0],
        "release archive digest mismatch",
    )

    if args.reference_directory is not None:
        require(args.reference_directory.is_dir(), "reference directory is missing")
        reference_assets = {
            path.name for path in args.reference_directory.iterdir() if path.is_file()
        }
        beta_assets = expected_assets - {"stable-promotion.json"}
        require(reference_assets == beta_assets, "reference asset set differs")
        for name in sorted(beta_assets):
            require(
                sha256(args.directory / name)
                == sha256(args.reference_directory / name),
                f"independent reference mismatch: {name}",
            )

    if args.promotion:
        promotion = json.loads((args.directory / "stable-promotion.json").read_text())
        require(
            promotion.get("schema") == "local-inference-b12x-promotion/v1",
            "unexpected promotion schema",
        )
        require(
            promotion.get("status") == "byte-identical-promotion",
            "promotion status must describe byte identity",
        )
        require(
            promotion.get("source_release") == args.beta_tag,
            "promotion source release does not match",
        )
        require(
            promotion.get("source_commit") == args.source_commit,
            "promotion source commit does not match",
        )
        require(
            promotion.get("source_manifest_sha256") == sha256(manifest_path),
            "promotion manifest digest does not match",
        )
    print("B12X release assets: PASS")


if __name__ == "__main__":
    main()
