import importlib

import pytest


@pytest.mark.parametrize("mod", ["config", "io", "normalize", "split", "evaluate",
                                 "blocking", "features", "train", "decide", "run", "package"])
def test_module_imports(mod):
    importlib.import_module(f"src.{mod}")
