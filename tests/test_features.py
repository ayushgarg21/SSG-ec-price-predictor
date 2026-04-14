"""Tests for feature engineering functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    extract_floor_mid,
    extract_lease_commence_year,
    parse_contract_date,
    compute_years_from_launch,
)


class TestExtractFloorMid:
    def test_standard_range(self) -> None:
        assert extract_floor_mid("06-10") == 8

    def test_single_digit(self) -> None:
        assert extract_floor_mid("01-05") == 3

    def test_high_floor(self) -> None:
        assert extract_floor_mid("31-35") == 33

    def test_none(self) -> None:
        assert extract_floor_mid(None) is None

    def test_empty_string(self) -> None:
        assert extract_floor_mid("") is None


class TestExtractLeaseCommenceYear:
    def test_standard(self) -> None:
        assert extract_lease_commence_year("99 yrs lease commencing from 2014") == 2014

    def test_none(self) -> None:
        assert extract_lease_commence_year(None) is None

    def test_freehold(self) -> None:
        assert extract_lease_commence_year("Freehold") is None


class TestParseContractDate:
    def test_standard(self) -> None:
        assert parse_contract_date("0325") == (3, 2025)

    def test_old_date(self) -> None:
        assert parse_contract_date("1299") == (12, 1999)

    def test_none(self) -> None:
        assert parse_contract_date(None) == (None, None)

    def test_invalid(self) -> None:
        assert parse_contract_date("abc") == (None, None)


class TestComputeYearsFromLaunch:
    def test_standard(self) -> None:
        assert compute_years_from_launch(2014, 2019) == 5

    def test_none_inputs(self) -> None:
        assert compute_years_from_launch(None, 2019) is None
