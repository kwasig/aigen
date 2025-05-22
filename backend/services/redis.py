import redis.asyncio as redis
import os
from dotenv import load_dotenv
import asyncio
from utils.logger import logger
from typing import List, Any

# Redis client and connection pool
client = None
pool = None
_initialized = False
_init_lock = asyncio.Lock()

# For connection caching to prevent repeated connections
_client_cache = {}

# Constants
REDIS_KEY_TTL = 3600 * 24  # 24 hour TTL as safety mechanism


def initialize():
    """Initialize Redis connection using environment variables."""
    global client, pool

    # Load environment variables if not already loaded
    load_dotenv()

    # Get Redis configuration
    redis_host = os.getenv('REDIS_HOST', 'redis')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    redis_password = os.getenv('REDIS_PASSWORD', '')
    # Convert string 'True'/'False' to boolean
    redis_ssl_str = os.getenv('REDIS_SSL', 'False')
    redis_ssl = redis_ssl_str.lower() == 'true'

    logger.info(f"Initializing Redis connection pool to {redis_host}:{redis_port}")

    # Create connection parameters dictionary, conditionally adding SSL
    connection_params = {
        "host": redis_host,
        "port": redis_port,
        "decode_responses": True,
        "max_connections": 20,  # Allow more concurrent connections
        "health_check_interval": 15
    }
    
    # Only add password if it's not empty
    if redis_password:
        connection_params["password"] = redis_password
        
    # Only add SSL if true and the library supports it
    if redis_ssl:
        try:
            # Test if SSL is supported by creating a temporary connection
            test_conn = redis.Redis(host=redis_host, port=redis_port, ssl=True)
            del test_conn
            # If we get here, SSL is supported
            connection_params["ssl"] = True
            logger.info("Redis SSL connection enabled")
        except (TypeError, ValueError) as e:
            logger.warning(f"Redis SSL parameter not supported by this library version: {e}")
            # Continue without SSL

    # Create Redis connection pool
    try:
        pool = redis.ConnectionPool(**connection_params)
        
        # Create Redis client with the connection pool
        client = redis.Redis(
            connection_pool=pool,
            socket_timeout=3.0,
            socket_connect_timeout=2.0,
            retry_on_timeout=True
        )
        logger.info("Redis pool and client created successfully")
    except Exception as e:
        logger.error(f"Error creating Redis pool: {e}")
        # Fallback to simplest possible connection
        try:
            client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_timeout=3.0,
                socket_connect_timeout=2.0
            )
            logger.info("Created fallback Redis client with basic parameters")
        except Exception as e2:
            logger.error(f"Even fallback Redis connection failed: {e2}")
            client = None

    return client


async def initialize_async():
    """Initialize Redis connection asynchronously."""
    global client, pool, _initialized

    # Early return if already initialized to prevent locking
    if _initialized and client:
        return client

    async with _init_lock:
        if not _initialized:
            logger.info("Initializing Redis connection")
            initialize()

            if client:
                try:
                    # Test connection with a short timeout
                    await asyncio.wait_for(client.ping(), timeout=1.0)
                    logger.info("Successfully connected to Redis")
                    _initialized = True
                except asyncio.TimeoutError:
                    logger.error("Redis connection timed out")
                    client = None
                    pool = None
                    # Don't raise, let the app work without Redis
                except Exception as e:
                    logger.error(f"Failed to connect to Redis: {e}")
                    client = None
                    pool = None
                    # Don't raise, let the app work without Redis
            else:
                logger.warning("Redis client not initialized, will operate without Redis")
                _initialized = True  # Mark as initialized to avoid retry loops

    return client


async def close():
    """Close Redis connection pool."""
    global client, pool, _initialized, _client_cache
    
    # Close any cached pubsub connections
    for key, cached_client in _client_cache.items():
        try:
            if hasattr(cached_client, 'close') and callable(cached_client.close):
                await cached_client.close()
            elif hasattr(cached_client, 'aclose') and callable(cached_client.aclose):
                await cached_client.aclose()
        except Exception as e:
            logger.warning(f"Error closing cached Redis client {key}: {e}")
    
    # Clear the cache
    _client_cache.clear()
    
    if pool:
        logger.info("Closing Redis connection pool")
        try:
            await pool.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting Redis pool: {e}")
        pool = None
        
    if client:
        try:
            await client.aclose()
        except Exception as e:
            logger.warning(f"Error closing Redis client: {e}")
        client = None
        
    _initialized = False
    logger.info("Redis connection resources released")


async def get_client():
    """Get the Redis client, initializing if necessary."""
    global client, _initialized
    
    # Cache key for the main client
    cache_key = "main_client"
    
    # Check if we have a cached client first
    if cache_key in _client_cache:
        return _client_cache[cache_key]
        
    if client is None or not _initialized:
        await initialize_async()
    
    # Cache the client for future use
    if client:
        _client_cache[cache_key] = client
        
    return client


# Helper function to safely execute Redis commands with fallbacks
async def _safe_redis_operation(operation_name, operation_func, *args, default_value=None, **kwargs):
    """Execute Redis operation safely with timeout and error handling."""
    if client is None:
        logger.debug(f"Redis client not available, skipping {operation_name}")
        return default_value
    
    try:
        return await asyncio.wait_for(operation_func(*args, **kwargs), timeout=2.0)
    except asyncio.TimeoutError:
        logger.warning(f"Redis {operation_name} operation timed out")
        return default_value
    except Exception as e:
        logger.warning(f"Redis {operation_name} operation failed: {e}")
        return default_value


# Optimized Redis operations with timeout protection
async def set(key: str, value: str, ex: int = None):
    """Set a Redis key with timeout protection."""
    redis_client = await get_client()
    if not redis_client:
        return False
        
    try:
        return await asyncio.wait_for(redis_client.set(key, value, ex=ex), timeout=2.0)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"Redis SET operation failed for key {key}: {e}")
        return False


async def get(key: str, default: str = None):
    """Get a Redis key with timeout protection."""
    redis_client = await get_client()
    if not redis_client:
        return default
        
    try:
        result = await asyncio.wait_for(redis_client.get(key), timeout=2.0)
        return result if result is not None else default
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"Redis GET operation failed for key {key}: {e}")
        return default


async def delete(key: str):
    """Delete a Redis key with timeout protection."""
    redis_client = await get_client()
    if not redis_client:
        return 0
        
    try:
        return await asyncio.wait_for(redis_client.delete(key), timeout=2.0)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"Redis DELETE operation failed for key {key}: {e}")
        return 0


async def publish(channel: str, message: str):
    """Publish a message to a Redis channel with timeout protection."""
    redis_client = await get_client()
    if not redis_client:
        return 0
        
    try:
        return await asyncio.wait_for(redis_client.publish(channel, message), timeout=2.0)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"Redis PUBLISH operation failed for channel {channel}: {e}")
        return 0


async def create_pubsub():
    """Create a Redis pubsub object without caching to avoid shared connection conflicts."""
    # Always create a fresh pubsub connection to avoid concurrent read issues
    redis_client = await get_client()
    if not redis_client:
        # Return a dummy pubsub object that does nothing
        from unittest.mock import AsyncMock
        dummy = AsyncMock()
        logger.warning("Creating dummy pubsub object since Redis is not available")
        return dummy
    
    # Create new pubsub connection
    pubsub = redis_client.pubsub()
    return pubsub


# List operations with timeout protection
async def rpush(key: str, *values: Any):
    """Append one or more values to a list with timeout protection."""
    redis_client = await get_client()
    if not redis_client:
        return 0
        
    try:
        return await asyncio.wait_for(redis_client.rpush(key, *values), timeout=2.0)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"Redis RPUSH operation failed for key {key}: {e}")
        return 0


async def lrange(key: str, start: int, end: int) -> List[str]:
    """Get a range of elements from a list with timeout protection."""
    redis_client = await get_client()
    if not redis_client:
        return []
        
    try:
        return await asyncio.wait_for(redis_client.lrange(key, start, end), timeout=2.0)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"Redis LRANGE operation failed for key {key}: {e}")
        return []


async def llen(key: str) -> int:
    """Get the length of a list with timeout protection."""
    redis_client = await get_client()
    if not redis_client:
        return 0
        
    try:
        return await asyncio.wait_for(redis_client.llen(key), timeout=2.0)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"Redis LLEN operation failed for key {key}: {e}")
        return 0


# Key management with timeout protection
async def expire(key: str, time: int):
    """Set a key's time to live in seconds with timeout protection."""
    redis_client = await get_client()
    if not redis_client:
        return False
        
    try:
        return await asyncio.wait_for(redis_client.expire(key, time), timeout=2.0)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"Redis EXPIRE operation failed for key {key}: {e}")
        return False


async def keys(pattern: str) -> List[str]:
    """Get keys matching a pattern with timeout protection."""
    redis_client = await get_client()
    if not redis_client:
        return []
        
    try:
        return await asyncio.wait_for(redis_client.keys(pattern), timeout=2.0)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"Redis KEYS operation failed for pattern {pattern}: {e}")
        return []