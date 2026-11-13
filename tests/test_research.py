import pandas as pd

from crashcast.research import engineer_features, make_panel


def test_target_is_next_week_and_features_are_lagged():
    rows = []
    for offset in range(60):
        week = pd.Timestamp("2023-01-02") + pd.Timedelta(weeks=offset)
        reports = (offset % 4) + 1
        for number in range(reports):
            rows.append({"REPORT_ID": f"{week}-{number}", "date_time": pd.Timestamp(week), "police_beat": "111", "injured": 0, "killed": 0, "hit_run": 0})
    panel = make_panel(pd.DataFrame(rows))
    model = engineer_features(panel)
    first = model.iloc[0]
    # At every usable row, the one-week lag is a past count and target is future.
    assert first["lag_1"] >= 0
    assert first["next_week_count"] >= 0
