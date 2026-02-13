"""Tests for scripts.prizepicks.collect_prizepicks — empty data handling."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts.prizepicks.collect_prizepicks import main


class TestEmptyDataExitBehavior:
    def test_empty_data_exits_zero(self):
        empty_payload = {"data": []}
        with (
            patch("sys.argv", ["collect_prizepicks"]),
            patch(
                "scripts.prizepicks.collect_prizepicks.fetch_projections",
                return_value=empty_payload,
            ),
            patch("scripts.prizepicks.collect_prizepicks.set_log_path"),
            patch("scripts.prizepicks.collect_prizepicks.log_validation"),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0

    def test_malformed_data_exits_one(self):
        bad_payload = "not a dict"
        with (
            patch("sys.argv", ["collect_prizepicks"]),
            patch(
                "scripts.prizepicks.collect_prizepicks.fetch_projections",
                return_value=bad_payload,
            ),
            patch("scripts.prizepicks.collect_prizepicks.set_log_path"),
            patch("scripts.prizepicks.collect_prizepicks.log_validation"),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1
