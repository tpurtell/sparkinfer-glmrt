"""Tests for the immutable B12X release-asset verifier."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SOURCE_COMMIT = "a" * 40
BETA_TAG = f"b12x-cu134-beta-{SOURCE_COMMIT}"
WHEEL_NAME = "b12x-1.3.0-py3-none-any.whl"


def digest(data: bytes) -> str:
    """Return the SHA-256 digest for generated fixture bytes."""
    return hashlib.sha256(data).hexdigest()


class VerifyReleaseAssetsTest(unittest.TestCase):
    """Exercise complete beta and promotion asset contracts."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.reference_temporary_directory = tempfile.TemporaryDirectory()
        self.reference_directory = Path(self.reference_temporary_directory.name)
        self.script = Path(__file__).with_name("verify_release_assets.py")
        self._write_assets(promotion=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()
        self.reference_temporary_directory.cleanup()

    def _write_assets(self, *, promotion: bool) -> None:
        wheel = b"source-locked B12X wheel"
        files = {
            WHEEL_NAME: wheel,
            "install.sh": b"#!/bin/sh\n",
            "requirements-github.txt": b"b12x @ https://example.invalid/wheel\n",
            "runtime.lock": b"cxx11-abi=1\n",
        }
        for name, data in files.items():
            (self.directory / name).write_bytes(data)
        manifest = {
            "schema": "local-inference-b12x-wheel-release/v1",
            "source": {"commit": SOURCE_COMMIT},
            "release_tag": BETA_TAG,
            "packages": [
                {
                    "name": "b12x",
                    "file": WHEEL_NAME,
                    "sha256": digest(wheel),
                }
            ],
        }
        manifest_path = self.directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        checksummed = {
            "manifest.json": manifest_path.read_bytes(),
            "requirements-github.txt": files["requirements-github.txt"],
            "runtime.lock": files["runtime.lock"],
            "install.sh": files["install.sh"],
            f"wheels/{WHEEL_NAME}": wheel,
        }
        (self.directory / "SHA256SUMS").write_text(
            "".join(f"{digest(data)}  {name}\n" for name, data in checksummed.items())
        )
        archive_name = f"b12x-cu134-{SOURCE_COMMIT}.tar.zst"
        archive = b"deterministic release archive"
        (self.directory / archive_name).write_bytes(archive)
        (self.directory / f"{archive_name}.sha256").write_text(
            f"{digest(archive)}  {archive_name}\n"
        )
        if promotion:
            promotion_marker = {
                "schema": "local-inference-b12x-promotion/v1",
                "status": "byte-identical-promotion",
                "source_release": BETA_TAG,
                "source_commit": SOURCE_COMMIT,
                "source_manifest_sha256": digest(manifest_path.read_bytes()),
            }
            (self.directory / "stable-promotion.json").write_text(
                json.dumps(promotion_marker)
            )

    def _verify(
        self, *, promotion: bool = False, reference: bool = False
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(self.script),
            "--directory",
            str(self.directory),
            "--source-commit",
            SOURCE_COMMIT,
            "--beta-tag",
            BETA_TAG,
        ]
        if promotion:
            command.append("--promotion")
        if reference:
            command.extend(["--reference-directory", str(self.reference_directory)])
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def test_complete_beta_release_passes(self) -> None:
        """A complete beta release satisfies every declared digest."""
        result = self._verify()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unexpected_asset_is_rejected(self) -> None:
        """An undeclared file cannot be included in a release."""
        (self.directory / "unexpected.bin").write_bytes(b"data")
        result = self._verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("release asset set differs", result.stderr)

    def test_modified_wheel_is_rejected(self) -> None:
        """Wheel bytes must match the manifest and checksum inventory."""
        (self.directory / WHEEL_NAME).write_bytes(b"modified")
        result = self._verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA256SUMS mismatch", result.stderr)

    def test_complete_promotion_passes(self) -> None:
        """A promotion marker binds stable assets to one beta manifest."""
        self._write_assets(promotion=True)
        result = self._verify(promotion=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reference_beta_bytes_pass(self) -> None:
        """Matching reference assets satisfy byte-identity verification."""
        shutil.copytree(self.directory, self.reference_directory, dirs_exist_ok=True)
        result = self._verify(reference=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_stable_matches_beta_without_promotion_marker(self) -> None:
        """A stable marker is additional metadata, not a change to beta bytes."""
        shutil.copytree(self.directory, self.reference_directory, dirs_exist_ok=True)
        self._write_assets(promotion=True)
        result = self._verify(promotion=True, reference=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_coordinated_asset_and_metadata_replacement_is_rejected(self) -> None:
        """Self-consistent replacement bytes must not override trusted bytes."""
        shutil.copytree(self.directory, self.reference_directory, dirs_exist_ok=True)
        replacement = b"substituted wheel"
        (self.directory / WHEEL_NAME).write_bytes(replacement)
        manifest_path = self.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["packages"][0]["sha256"] = digest(replacement)
        manifest_path.write_text(json.dumps(manifest))
        checksums = self.directory / "SHA256SUMS"
        entries = [line.split(maxsplit=1)[1] for line in checksums.read_text().splitlines()]
        checksums.write_text(
            "".join(
                f"{digest((self.directory / Path(name).name).read_bytes())}  {name}\n"
                for name in entries
            )
        )
        self.assertEqual(self._verify().returncode, 0)
        result = self._verify(reference=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("independent reference mismatch", result.stderr)

    def test_incomplete_reference_is_rejected(self) -> None:
        """A partial independent inventory cannot establish byte identity."""
        result = self._verify(reference=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reference asset set differs", result.stderr)


if __name__ == "__main__":
    unittest.main()
