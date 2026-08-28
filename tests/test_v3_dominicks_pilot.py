from __future__ import annotations

import numpy as np

from scripts.run_v3_dominicks_pilot import (
    ItemModel,
    SeriesRow,
    _bounded_shock_vectors,
    _observed_price_rows,
    _repeated_training_menu,
)


def _model(key: tuple[int, int], rows: dict[int, SeriesRow]) -> ItemModel:
    return ItemModel(
        key=key,
        beta=np.array([np.log(11.0), 0.0, 0.0, 0.0, 0.0]),
        cost=1.0,
        reference=2.0,
        menu=np.array([2.0]),
        demand=np.array([10.0]),
        rows=rows,
        train_rows=1,
        tune_rows=1,
        log_price_coefficient=-1.0,
        baseline_week=2,
        menu_training_regular_occurrences=np.array([2]),
    )


def test_zero_price_record_is_not_carried_forward() -> None:
    rows = [
        SeriesRow(week=1, demand=4.0, price=2.0, sale=False, profit_pct=10.0),
        SeriesRow(week=2, demand=0.0, price=0.0, sale=False, profit_pct=None),
        SeriesRow(week=3, demand=5.0, price=2.1, sale=False, profit_pct=10.0),
    ]

    observed = _observed_price_rows(rows)

    assert set(observed) == {1, 3}
    assert 2 not in observed


def test_portfolio_residual_requires_positive_price_for_every_roster_item() -> None:
    common_row = SeriesRow(
        week=3, demand=10.0, price=2.0, sale=False, profit_pct=10.0
    )
    first = _model((1, 1), {3: common_row, 4: common_row})
    second = _model((1, 2), {3: common_row})

    weeks, shocks = _bounded_shock_vectors([first, second], 2, 4)

    assert weeks == [3]
    assert shocks.shape == (1, 2)


def test_menu_excludes_training_price_singletons() -> None:
    menu, occurrences = _repeated_training_menu(
        [1.99, 1.99, 2.09, 2.09, 2.19, 2.29], maximum=8
    )

    assert menu.tolist() == [1.99, 2.09]
    assert occurrences.tolist() == [2, 2]
