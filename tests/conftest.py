import pytest


@pytest.fixture(autouse=True)
def _no_skeleton_file(monkeypatch):
    """Tests never read data/human_skeletons.json by accident; the ones that want skeletons pass Deps(skeletons=[...])."""
    monkeypatch.setenv("HARNESS_SKELETONS", "off")
    monkeypatch.setenv("HARNESS_SHAPE", "off")        # fixtures are one-sentence paragraphs; the shape repairs are tested directly
    monkeypatch.setenv("HARNESS_DETECTOR", "off")     # the shipped Claude-style detector would add spans to fixtures built to be flat; its tests pass a model
    monkeypatch.setenv("HARNESS_LAMP", "off")        # the LAMP tests switch it on; other fixtures are flat by construction


@pytest.fixture(autouse=True)
def _no_burst_check(monkeypatch):
    """Most fixtures repeat one sentence, which is flat by construction; tests of the burstiness check put the real one back."""
    from harness import graph
    monkeypatch.setattr(graph, "burst_flags", lambda text: [])
