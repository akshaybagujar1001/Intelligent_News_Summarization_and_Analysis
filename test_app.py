# -*- coding: utf-8 -*-

# Install--> pip install pytest-asyncio pytest-trio pytest-twisted pytest-tornasync anyio twisted
# run--> pytest test_app.py

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from fastapi_app import app, fetch_news_api, process_article_content

# Initialize TestClient for FastAPI
client = TestClient(app)


@pytest.mark.asyncio
async def test_get_news():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/news/", params={"query": "AI"})
    assert response.status_code == 200
    assert "articles" in response.json()
    assert isinstance(response.json()["articles"], list)


@pytest.mark.asyncio
async def test_get_trending():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/trending/")
    assert response.status_code == 200
    assert "trending_topics" in response.json()
    assert isinstance(response.json()["trending_topics"], list)


# Define tests for core functions in the original code
@pytest.mark.asyncio
async def test_fetch_news_api():
    news_items = await fetch_news_api("AI")
    assert isinstance(news_items, list)
    assert len(news_items) > 0
    for item in news_items:
        assert "title" in item
        assert "date" in item
        assert "content" in item
        assert "source" in item


@pytest.mark.asyncio
async def test_process_article_content():
    raw_content = "<html><body>Sample Content &copy; 2024</body></html>"
    processed_content = await process_article_content(raw_content)
    assert processed_content == "Sample Content © 2024"
