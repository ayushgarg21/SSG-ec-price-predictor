"""Tests for data ingestion."""

import pytest
from src.ingestion.ingest import flatten_transactions


class TestFlattenTransactions:
    def test_basic_flatten(self):
        raw = [
            {
                "project": "Piermont Grand",
                "street": "SUMANG WALK",
                "x": "38123.45",
                "y": "40456.78",
                "marketSegment": "OCR",
                "transaction": [
                    {
                        "area": "93",
                        "floorRange": "06-10",
                        "noOfUnits": "1",
                        "contractDate": "0324",
                        "typeOfSale": "New Sale",
                        "price": "1050000",
                        "propertyType": "Executive Condominium",
                        "district": "19",
                        "typeOfArea": "Strata",
                        "tenure": "99 yrs lease commencing from 2019",
                        "nettPrice": "",
                    },
                    {
                        "area": "110",
                        "floorRange": "11-15",
                        "noOfUnits": "1",
                        "contractDate": "0324",
                        "typeOfSale": "New Sale",
                        "price": "1250000",
                        "propertyType": "Executive Condominium",
                        "district": "19",
                        "typeOfArea": "Strata",
                        "tenure": "99 yrs lease commencing from 2019",
                        "nettPrice": "",
                    },
                ],
            }
        ]
        rows = flatten_transactions(raw)
        assert len(rows) == 2
        assert rows[0]["project"] == "Piermont Grand"
        assert rows[0]["price"] == 1050000.0
        assert rows[1]["area"] == 110.0

    def test_empty_input(self):
        assert flatten_transactions([]) == []

    def test_project_with_no_transactions(self):
        raw = [{"project": "Empty EC", "transaction": []}]
        assert flatten_transactions(raw) == []
