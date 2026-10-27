"""Prevent training leakage when auditing models from different dataset versions."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from src.monitoring.audit import verify_audit_membership


class AuditMembershipTests(unittest.TestCase):
    def test_registered_selection_allows_heldout_rejects_training_and_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = root / 'selection.json'
            selection.write_text(json.dumps({'train': ['trained'], 'validation': ['heldout']}))
            hashes = {'inputs/selection.json': hashlib.sha256(selection.read_bytes()).hexdigest()}
            bundle = root / 'bundle-sha256.json'
            bundle.write_text(json.dumps(hashes))
            manifest = {'run_id': 'test', 'bundle_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
            client = Mock()
            client.download_artifacts.side_effect = lambda run, name, destination: str(bundle if name == 'bundle-sha256.json' else selection)
            verify_audit_membership(client, manifest, ['heldout', 'unseen'])
            with self.assertRaisesRegex(ValueError, 'overlap'):
                verify_audit_membership(client, manifest, ['trained'])
            selection.write_text(json.dumps({'train': [], 'validation': ['heldout']}))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify_audit_membership(client, manifest, ['trained'])
            manifest['bundle_sha256'] = 'tampered'
            with self.assertRaisesRegex(ValueError, 'bundle changed'):
                verify_audit_membership(client, manifest, ['heldout'])
