"""
Unit and regression test for the cmiles package.
"""

# Import package, test suite, and other packages as needed
import cmiles
import pytest
import sys

def test_cmiles_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "cmiles" in sys.modules
