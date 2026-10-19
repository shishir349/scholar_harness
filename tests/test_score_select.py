from harness.corpus.score import select


def rows(n, field="f", excluded=()):
    return [{"openalex_id": f"W{i}", "field_slug": field, "score": 100 - i, "exclude": i in excluded} for i in range(n)]


def test_excluded_rows_are_backfilled_from_spares_and_never_selected():
    out = select(rows(8, excluded={0}))
    chosen = [r for r in out if not r["exclude"]]
    assert len(chosen) == 6 and "W0" not in {r["openalex_id"] for r in chosen}
    assert sum(r["holdout"] for r in chosen) == 1


def test_exclude_flag_survives_in_the_output_so_the_next_reselect_keeps_it():
    out = select(rows(8, excluded={0}))
    kept = [r for r in out if r["openalex_id"] == "W0"]
    assert len(kept) == 1 and kept[0]["exclude"] is True and kept[0]["holdout"] is False


def test_small_field_gets_no_holdout_under_four():
    out = select(rows(3))
    assert not any(r["holdout"] for r in out)
