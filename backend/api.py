from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from contextlib import asynccontextmanager
from agentpress.thread_manager import ThreadManager
from services.supabase import DBConnection
from datetime import datetime, timezone
from dotenv import load_dotenv
from utils.config import config, EnvMode
import asyncio
from utils.logger import logger
import uuid
import time
from collections import OrderedDict
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram
from utils.metrics import REQUEST_COUNT, REQUEST_LATENCY, registry as METRICS_REGISTRY

# Import the agent API module
from agent import api as agent_api
from sandbox import api as sandbox_api
from services import billing as billing_api
from services import redis

# Load environment variables early (these will be available through config)
load_dotenv()

# Create DB and thread manager instance early
db = DBConnection()
thread_manager = ThreadManager()
instance_id = "single"

# Pre-initialize certain connections during module loading to reduce startup time
def preload_connections():
    # Asynchronous preloading will happen in lifespan function
    # This just creates the connection objects but doesn't actually connect yet
    logger.info("Pre-initializing connection objects to reduce startup time")
    
# Call preload immediately
preload_connections()

# Prometheus metrics for observability
# Use a dedicated registry to avoid duplicate metric registration when the app
# is loaded multiple times (e.g. in multiprocessing scenarios).
METRICS_REGISTRY = CollectorRegistry()
REQUEST_COUNT = Counter(
    "api_request_total",
    "Total API requests",
    ["method", "endpoint"],
    registry=METRICS_REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["method", "endpoint"],
    registry=METRICS_REGISTRY,
)

# Rate limiter state
ip_tracker = OrderedDict()
MAX_CONCURRENT_IPS = 25

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global thread_manager
    startup_time = time.time()
    logger.info(f"Starting up FastAPI application with instance ID: {instance_id} in {config.ENV_MODE.value} mode")
    
    try:
        # Initialize connections in parallel to reduce startup time
        init_tasks = [
            asyncio.create_task(db.initialize()),
            asyncio.create_task(redis.initialize_async()),
        ]
        
        # Wait for all initialization tasks to complete with timeout
        try:
            done, pending = await asyncio.wait(
                init_tasks, 
                timeout=5.0,  # 5 second timeout for initialization
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                
            # Check for exceptions in completed tasks - but don't abort startup
            for task in done:
                if task.exception():
                    logger.warning(f"Initialization task raised exception: {task.exception()}")
        except asyncio.TimeoutError:
            logger.warning("Initialization tasks timed out - proceeding anyway")
            for task in init_tasks:
                if not task.done():
                    task.cancel()
                    
        # Pre-warm Redis connection pool by making a simple ping request
        try:
            await asyncio.wait_for(redis.get_client(), timeout=1.0)
            await asyncio.wait_for(redis.set("api_startup", str(datetime.now(timezone.utc)), ex=300), timeout=1.0)
            logger.info("Redis connection pool pre-warmed")
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Failed to pre-warm Redis pool: {e}")
            
        # Initialize API resources
        thread_manager = ThreadManager()
        
        # Initialize agent API
        agent_api.initialize(thread_manager, db, instance_id)
        logger.info(f"Initialized agent API with instance ID: {instance_id}")
        
        # Initialize sandbox API
        sandbox_api.initialize(db)
        logger.info(f"Initialized sandbox API with database connection")
        
        startup_duration = time.time() - startup_time
        logger.info(f"All essential services initialized in {startup_duration:.2f}s")
        
        yield
    
    except Exception as e:
        logger.error(f"Error during API initialization: {e}", exc_info=True)
        yield
    finally:
        # Shutdown
        logger.info("Shutting down FastAPI application")
        
        # Stop any running agents
        try:
            await agent_api.cleanup()
        except Exception as e:
            logger.error(f"Error during agent API cleanup: {e}")
            
        # Close database connection
        try:
            await db.close()
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
        
        # Close Redis connection
        try:
            await redis.close()
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
            
        logger.info("Shutdown complete")

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Middleware to log requests and measure execution time."""
    # Add a unique ID to each request for tracking
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    
    # Extract client info
    client_host = request.client.host if request.client else "unknown"
    
    # Get path and query string
    path = request.url.path
    query = str(request.url.query)
    query_display = f" | Query: {query}" if query else ""
    
    # Start timer and log request start
    start_time = time.time()
    logger.info(f"Request started: {request.method} {path} from {client_host}{query_display}")
    
    # Pre-initialize connections for faster response
    try:
        # Perform these operations in parallel to reduce latency
        # Initialize DB if needed
        asyncio.create_task(db.initialize())
            
        # Pre-initialize Redis for better caching performance
        asyncio.create_task(redis.get_client())
    except Exception as e:
        logger.warning(f"Failed to pre-initialize connections: {str(e)}")

    # Get response with error handling
    try:
        # Increment the request count metric
        REQUEST_COUNT.labels(
            method=request.method, 
            endpoint=path
        ).inc()
        
        # Execute the request
        response = await call_next(request)
        
        # Record response time
        process_time = time.time() - start_time
        
        # Record latency in Prometheus
        REQUEST_LATENCY.labels(
            method=request.method, 
            endpoint=path
        ).observe(process_time)
        
        # Record response status and time
        status_code = response.status_code
        
        # Log request completion with detailed info for non-200 responses
        if status_code >= 400:
            logger.warning(f"Request error: {request.method} {path} | Status: {status_code} | Time: {process_time:.2f}s")
        else:
            logger.debug(f"Request completed: {request.method} {path} | Status: {status_code} | Time: {process_time:.2f}s")
            
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Unhandled exception: {request.method} {path} | Error: {str(e)} | Time: {process_time:.2f}s", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"}
        )

# Define allowed origins based on environment
allowed_origins = ["https://www.suna.so", "https://suna.so", "https://staging.suna.so", "http://localhost:3000"]

# Add staging-specific origins
if config.ENV_MODE == EnvMode.STAGING:
    allowed_origins.append("http://localhost:3000")
    
# Add local-specific origins
if config.ENV_MODE == EnvMode.LOCAL:
    allowed_origins.append("http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Include the agent router with a prefix
app.include_router(agent_api.router, prefix="/api")

# Include the sandbox router with a prefix
app.include_router(sandbox_api.router, prefix="/api")

# Include the billing router with a prefix
app.include_router(billing_api.router, prefix="/api")

@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify API is working."""
    logger.info("Health check endpoint called")
    return {
        "status": "ok", 
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "instance_id": instance_id
    }

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    data = generate_latest(METRICS_REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    
    workers = 2
    
    logger.info(f"Starting server on 0.0.0.0:8000 with {workers} workers")
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000,
        workers=workers,
        # reload=True
    )