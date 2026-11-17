"""CPU checks for fail-closed adapter publication and recovery guards."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import common
class RecoveryTests(unittest.TestCase):
    def test_correct_hash(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"adapter";f.write_bytes(b"valid")
            self.assertEqual(common.verified_adapter(f,common.sha(f),timeout=0),common.sha(f))
    def test_transient_read(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"adapter";f.write_bytes(b"valid")
            with patch.object(common,"sha",side_effect=["stale","expected"]),patch.object(common.time,"sleep"):
                self.assertEqual(common.verified_adapter(f,"expected"),"expected")
    def test_wrong_hash_not_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"adapter";f.write_bytes(b"invalid")
            with self.assertRaisesRegex(RuntimeError,"expected=correct"):
                common.verified_adapter(f,"correct",timeout=0)
if __name__=="__main__":unittest.main()
