from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    return load_dataset(file_name="")


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
