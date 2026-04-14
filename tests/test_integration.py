"""Integration tests using testcontainers for a real Postgres instance."""

from __future__ import annotations

import pytest

try:
    from testcontainers.postgres import PostgresContainer
    HAS_TESTCONTAINERS = True
except ImportError:
    HAS_TESTCONTAINERS = False

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pg_container():
    if not HAS_TESTCONTAINERS:
        pytest.skip("testcontainers not installed — run `pip install -e '.[dev]'`")

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


@pytest.fixture(scope="module")
def db_engine(pg_container):
    engine = create_engine(pg_container.get_connection_url())
    # Apply schema
    init_sql_path = "scripts/init_db.sql"
    with open(init_sql_path) as f:
        sql = f.read()
    with engine.begin() as conn:
        for statement in sql.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
    return engine


@pytest.fixture
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


class TestSchemaCreation:
    def test_tables_exist(self, db_engine) -> None:
        with db_engine.connect() as conn:
            result = conn.execute(text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            ))
            tables = {row[0] for row in result}
        assert "ura_transactions" in tables
        assert "ec_features" in tables
        assert "model_registry" in tables
        assert "prediction_logs" in tables


class TestDataIngestion:
    def test_insert_and_query_transaction(self, db_session) -> None:
        db_session.execute(text("""
            INSERT INTO ura_transactions
                (project, street, market_segment, area, floor_range, contract_date,
                 type_of_sale, price, property_type, district, tenure)
            VALUES
                ('Test EC', 'Test St', 'OCR', 95, '06-10', '0324',
                 'Resale', 850000, 'Executive Condominium', '19',
                 '99 yrs lease commencing from 2018')
        """))
        db_session.commit()

        result = db_session.execute(
            text("SELECT COUNT(*) FROM ura_transactions WHERE project = 'Test EC'")
        )
        assert result.scalar() == 1

    def test_prediction_logging(self, db_session) -> None:
        db_session.execute(text("""
            INSERT INTO prediction_logs (model_version, input_features, predicted_price, latency_ms)
            VALUES ('v1.0', '{"district": 19}', 850000, 12.5)
        """))
        db_session.commit()

        result = db_session.execute(
            text("SELECT predicted_price FROM prediction_logs WHERE model_version = 'v1.0'")
        )
        assert float(result.scalar()) == 850000.0


class TestFeatureEngineering:
    """Test the full flow: insert raw data, query it, build features."""

    def test_ec_filter_via_sql(self, db_session) -> None:
        # Insert mixed property types
        for ptype, price in [("Executive Condominium", 900000), ("Condominium", 1500000)]:
            db_session.execute(text("""
                INSERT INTO ura_transactions
                    (project, market_segment, area, floor_range, contract_date,
                     type_of_sale, price, property_type, district, tenure)
                VALUES
                    (:proj, 'OCR', 95, '06-10', '0324', 'Resale', :price, :ptype, '19',
                     '99 yrs lease commencing from 2018')
            """), {"proj": f"Test {ptype}", "price": price, "ptype": ptype})
        db_session.commit()

        result = db_session.execute(text(
            "SELECT COUNT(*) FROM ura_transactions WHERE property_type = 'Executive Condominium'"
        ))
        ec_count = result.scalar()
        assert ec_count >= 1
