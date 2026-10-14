import numpy as np
import pandas as pd
import pytest

from alpha.splits import purged_before, walk_forward

CAL = pd.bdate_range("2012-01-02", "2026-09-30")
H, EMB = 5, 5


def _label_end(d):
    return CAL[CAL.get_loc(d) + 1 + H]


@pytest.fixture(scope="module")
def dev():
    return walk_forward(CAL, range(2015, 2023), "2013-01-01", H, EMB, last_date="2022-12-31")


def test_no_training_label_reaches_into_the_embargo(dev):
    for s in dev:
        j0 = CAL.get_loc(s.test[0])
        assert CAL.get_loc(_label_end(s.train[-1])) + EMB < j0
        # the purge is tight: the next calendar day would violate it
        nxt = CAL[CAL.get_loc(s.train[-1]) + 1]
        assert CAL.get_loc(_label_end(nxt)) + EMB >= j0
        assert j0 - CAL.get_loc(s.train[-1]) == 1 + H + EMB + 1      # 11 trading days between them


def test_inner_validation_is_purged_and_inside_training(dev):
    for s in dev:
        assert s.val.isin(s.train).all() and s.inner_train.isin(s.train).all()
        j0 = CAL.get_loc(s.val[0])
        assert CAL.get_loc(_label_end(s.inner_train[-1])) + EMB < j0
        assert 240 <= len(s.val) <= 262
        assert s.inner_train.max() < s.val.min() and s.train.max() < s.test.min()


def test_test_blocks_are_years_disjoint_and_expanding(dev):
    tests = [s.test for s in dev]
    assert all(t.year.unique().tolist() == [int(s.name)] for s, t in zip(dev, tests))
    assert len(pd.Index(np.concatenate(tests)).unique()) == sum(map(len, tests))
    assert all(s.train[0] == pd.Timestamp("2013-01-01") for s in dev)
    assert all(len(a.train) < len(b.train) for a, b in zip(dev, dev[1:]))
    # every development date from 2015 on is tested exactly once, except the last 1 + h, whose labels
    # would need 2023 (hold-out) prices
    dev_days = CAL[(CAL >= "2015-01-01") & (CAL <= "2022-12-31")]
    assert pd.Index(np.concatenate(tests)).equals(dev_days[: -(1 + H)])
    assert _label_end(tests[-1][-1]) <= pd.Timestamp("2022-12-31")


def test_holdout_test_dates_have_complete_labels():
    ho = walk_forward(CAL, [2023, 2024, 2025, 2026], "2013-01-01", H, EMB)
    last = ho[-1].test[-1]
    assert CAL.get_loc(last) + 1 + H == len(CAL) - 1
    assert all(s.train.max() < pd.Timestamp(f"{s.name}-01-01") for s in ho)


def test_purge_scales_with_horizon():
    d = CAL[(CAL >= "2018-01-01") & (CAL < "2019-01-01")]
    first = pd.Timestamp("2019-01-02")
    for h in (1, 5, 21):
        kept = purged_before(CAL, d, first, h, h)
        assert CAL.get_loc(first) - CAL.get_loc(kept[-1]) == 1 + 2 * h + 1
