# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from pydantic import BaseModel
from typing import List
import asyncio
import re
import logging
import redis.asyncio as aioredis  # Async Redis client for better performance

# Initialize FastAPI application
app = FastAPI()

# Set up logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS middleware setup (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For security, restrict this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get Redis client asynchronously
async def get_redis_client():
    return aioredis.Redis(host="localhost", port=6379, db=0)

# Initialize cache with Redis on startup
@app.on_event("startup")
async def startup():
    try:
        redis = await get_redis_client()
        FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
        logger.info("Successfully connected to Redis and initialized cache.")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")

# Define a NewsItem model
class NewsItem(BaseModel):
    title: str
    date: str
    content: str
    source: str

# Mock external functions to replace with actual implementations
async def fetch_news_api(query: str) -> List[NewsItem]:
    # Placeholder for async news fetching function
    await asyncio.sleep(1)  # Simulate delay
    return [NewsItem(title="Sample Title", date="2024-11-04", content="Sample Content", source="Sample Source")]

async def process_article_content(content: str) -> str:
    # Example async function to process article content
    content = re.sub(r"<.*?>", "", content)  # Remove HTML tags
    return content

# Define routes with caching
@app.get("/news/")
@cache(expire=60)
async def get_news(query: str = Query(..., min_length=1, description="Search term for news articles")):
    logger.info(f"Fetching news for query: {query}")
    try:
        news = await fetch_news_api(query)
        processed_news = [
            {
                "title": item.title,
                "date": item.date,
                "content": await process_article_content(item.content),
                "source": item.source
            } for item in news
        ]
        return JSONResponse(content={"articles": processed_news})
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

@app.get("/trending/")
@cache(expire=60)
async def get_trending():
    logger.info("Fetching trending topics")
    try:
        await asyncio.sleep(1)  # Simulate processing delay
        return JSONResponse(content={"trending_topics": ["AI", "Climate", "Economy"]})
    except Exception as e:
        logger.error(f"Error fetching trending topics: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching trending topics.")

# Custom exception handler for global error handling
@app.exception_handler(Exception)
async def custom_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An error occurred while processing your request."},
    )

# Run the app with: uvicorn fastapi_app:app --reload
