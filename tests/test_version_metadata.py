import importlib.metadata

import da


def test_da_version_matches_installed_metadata():
    assert da.__version__ == importlib.metadata.version("da_py")


def test_da_version_is_next_minor_release():
    assert da.__version__ == "0.7.0"
