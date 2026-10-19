import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import stage_training_payload as staging  # noqa: E402


class SonameParsingTests(unittest.TestCase):
    def test_parses_common_readelf_wording(self):
        text = " 0x000000000000000e (SONAME) Library soname: [libc.so.6]\n"
        self.assertEqual(staging.parse_shared_object_name(text), "libc.so.6")

    def test_parses_alternate_readelf_wording(self):
        text = " 0x000000000000000e (SONAME) Shared library: [libdemo.so.1]\n"
        self.assertEqual(staging.parse_shared_object_name(text), "libdemo.so.1")

    def test_missing_soname_returns_none(self):
        self.assertIsNone(staging.parse_shared_object_name("Dynamic section is empty\n"))


if __name__ == "__main__":
    unittest.main()
