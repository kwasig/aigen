import asyncio
import json
import traceback
from datetime import datetime, timezone
from typing import Optional
from services import redis
from agent.run import run_agent
from utils.logger import logger
import dramatiq
import uuid
from agentpress.thread_manager import ThreadManager
from services.supabase import DBConnection
from services import redis
from dramatiq.brokers.rabbitmq import RabbitmqBroker
import os
import time
from utils.profiling import profile

# Configure RabbitMQ with improved settings
rabbitmq_host = os.getenv('RABBITMQ_HOST', 'rabbitmq')
rabbitmq_port = int(os.getenv('RABBITMQ_PORT', 5672))

# Initialize RabbitMQ connection with better defaults and retry logic
MAX_RETRIES = 3
retry_count = 0
rabbitmq_broker = None

while retry_count < MAX_RETRIES and rabbitmq_broker is None:
    try:
        rabbitmq_broker = RabbitmqBroker(
            host=rabbitmq_host,
            port=rabbitmq_port,
            middleware=[
                dramatiq.middleware.AsyncIO(),
                dramatiq.middleware.Retries(max_retries=3),
                dramatiq.middleware.TimeLimit(time_limit=3600000),  # 1 hour
            ],
            # Add performance optimizations
            confirm_delivery=False,  # Disable confirmations for better throughput
            heartbeat=60,   # Longer heartbeat interval
            connection_attempts=3,   # Connection retry attempts
        )
        logger.info(f"Successfully connected to RabbitMQ at {rabbitmq_host}:{rabbitmq_port}")
    except Exception as e:
        retry_count += 1
        if retry_count >= MAX_RETRIES:
            logger.error(f"Failed to connect to RabbitMQ after {MAX_RETRIES} attempts: {e}")
            # Create a stub broker - we'll handle messages differently without RabbitMQ
            from dramatiq.brokers.stub import StubBroker
            rabbitmq_broker = StubBroker()
            logger.warning("Using StubBroker as fallback - background tasks will be simulated")
        else:
            logger.warning(f"RabbitMQ connection attempt {retry_count} failed: {e}, retrying...")
            time.sleep(1)  # Wait a bit before retrying

# Set as the global broker
dramatiq.set_broker(rabbitmq_broker)

# Define an actor for running the agent in the background
@dramatiq.actor
@profile
async def run_agent_background(
    agent_run_id: str,
    thread_id: str,
    instance_id: str = None,
    project_id: str = None,
    model_name: str = None,
    enable_thinking: bool = True,
    reasoning_effort: str = 'medium',
    stream: bool = True,
    enable_context_manager: bool = False
):
    """Run an agent in the background with the specified configuration."""
    # Log start with simple parameters
    logger.info(f"Starting background agent run {agent_run_id} for thread {thread_id} (Instance: {instance_id})")
    
    start_time = time.time()
    thread_manager = None
    db = None
    
    try:
        # Initialize resources in parallel to reduce latency
        # 1. Initialize Redis
        redis_task = asyncio.create_task(redis.initialize_async())
        
        # 2. Initialize DB connection
        db = DBConnection()
        db_task = asyncio.create_task(db.initialize())
        
        # Wait for both tasks to complete with timeout
        try:
            done, pending = await asyncio.wait(
                [redis_task, db_task], 
                timeout=2.0,  # Reduced timeout for faster startup
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any tasks that didn't complete in time
            for task in pending:
                task.cancel()
                
        except Exception as e:
            logger.warning(f"Error during parallel initialization: {e}")
            # Continue anyway, we'll check individual results
            
        # Get database client (will fall back to initialization if needed)
        client = await db.client
        
        # Register this run in Redis with TTL
        try:
            # Set the active run key with instance ID
            instance_key = f"active_run:{instance_id}:{agent_run_id}"
            await redis.set(instance_key, "running", ex=redis.REDIS_KEY_TTL)
            logger.debug(f"Registered agent run in Redis: {instance_key}")
        except Exception as e:
            logger.warning(f"Failed to register run in Redis: {e}")
        
        # Update status as we start
        update_success = await update_agent_run_status(
            client, agent_run_id, "running", 
            responses=[{"type": "status", "status": "running", "message": "Starting agent..."}]
        )
        
        if not update_success:
            logger.warning(f"Could not update status for agent run {agent_run_id}")
        
        # Create ThreadManager
        thread_manager = ThreadManager()
        
        # Create Redis keys
        response_list_key = f"agent_run:{agent_run_id}:responses"
        response_channel = f"agent_run:{agent_run_id}:new_response"
        
        # Run the agent and stream responses to Redis
        agent_responses = []
        async for response in run_agent(
            thread_id=thread_id,
            project_id=project_id,
            thread_manager=thread_manager,
            model_name=model_name,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            stream=stream,
            enable_context_manager=enable_context_manager
        ):
            agent_responses.append(response)
            # Push to Redis for the SSE stream
            try:
                await redis.rpush(response_list_key, json.dumps(response))
                await redis.publish(response_channel, "new")
            except Exception as e:
                logger.warning(f"Failed to push response to Redis for {agent_run_id}: {e}")
            
        # Determine success based on final response
        agent_success = True
        for response in reversed(agent_responses):
            if response.get('type') == 'status' and response.get('status') in ['error', 'failed']:
                agent_success = False
                break
        
        # Determine final status based on agent success
        final_status = "completed" if agent_success else "failed"
        
        # Update the agent run status in the database
        update_success = await update_agent_run_status(
            client, agent_run_id, final_status, responses=agent_responses
        )
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Agent run {agent_run_id} completed in {duration:.2f}s with status: {final_status}")
        
        # Initiate cleanup of Redis resources
        await _cleanup_redis_response_list(agent_run_id, delay=3600)  # Keep responses in Redis for 1 hour
        
        return agent_success
        
    except Exception as e:
        logger.error(f"Error in agent background run {agent_run_id}: {str(e)}\n{traceback.format_exc()}")
        
        try:
            # Mark the run as failed in the database
            if db:
                client = await db.client
                await update_agent_run_status(
                    client, agent_run_id, "failed", 
                    error=f"Error: {str(e)}",
                    responses=[{"type": "status", "status": "error", "message": f"Error: {str(e)}"}]
                )
        except Exception as db_err:
            logger.error(f"Failed to update error status in database: {str(db_err)}")
            
        # Try to clean up Redis anyway
        try:
            await _cleanup_redis_response_list(agent_run_id)
        except Exception:
            pass
            
        # Re-raise to let Dramatiq handle the failure
        raise
    finally:
        # Clean up any Redis instance key (not the response list yet)
        try:
            if instance_id:
                instance_key = f"active_run:{instance_id}:{agent_run_id}"
                await redis.delete(instance_key)
                logger.debug(f"Cleaned up Redis key: {instance_key}")
        except Exception as redis_err:
            logger.warning(f"Failed to clean up Redis key: {str(redis_err)}")


async def _cleanup_redis_response_list(agent_run_id: str, delay: int = 0):
    """Clean up Redis response list after a delay (useful to keep responses for SSE)."""
    try:
        if delay > 0:
            # Set a TTL on the response list rather than deleting immediately
            response_list_key = f"agent_run:{agent_run_id}:responses"
            await redis.expire(response_list_key, delay)
            logger.debug(f"Set TTL of {delay}s on Redis key: {response_list_key}")
        else:
            # Delete immediately
            response_list_key = f"agent_run:{agent_run_id}:responses"
            await redis.delete(response_list_key)
            logger.debug(f"Deleted Redis key: {response_list_key}")
    except Exception as e:
        logger.warning(f"Failed to clean up Redis response list for {agent_run_id}: {str(e)}")


async def update_agent_run_status(
    client, agent_run_id: str, status: str, error: Optional[str] = None, responses=None
):
    """Update the status of an agent run in the database."""
    try:
        update_data = {
            "status": status
        }
        
        if status in ["completed", "failed", "stopped"]:
            update_data["completed_at"] = datetime.now(timezone.utc).isoformat()
            
        if error:
            update_data["error"] = error
            
        # Update responses if provided and status indicates completion
        if responses and status in ["completed", "failed", "stopped"]:
            # Limit size of responses to avoid database issues
            # Only store a subset of important responses
            filtered_responses = []
            for resp in responses:
                # Keep status messages, errors, and important tool outputs
                if resp.get('type') in ['status', 'error', 'final', 'thinking']:
                    filtered_responses.append(resp)
                elif resp.get('type') == 'tool' and resp.get('name') in ['execute_code', 'file_operation']:
                    filtered_responses.append(resp)
                    
            # Still too big? Keep just the most important ones
            if len(json.dumps(filtered_responses)) > 100000:  # ~100KB limit
                logger.warning(f"Responses for {agent_run_id} too large, truncating for DB storage")
                # Only keep status messages
                filtered_responses = [r for r in filtered_responses if r.get('type') in ['status', 'error', 'final']]
                
            # Add responses to update data if we have any left
            if filtered_responses:
                update_data["responses"] = json.dumps(filtered_responses)
                
        # Update the database
        update_result = await client.table('agent_runs').update(update_data).eq('id', agent_run_id).execute()
        
        # Check if update was successful
        if hasattr(update_result, 'data') and update_result.data:
            return True
        else:
            logger.warning(f"No rows updated when setting agent run {agent_run_id} status to {status}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating agent run status in database: {str(e)}")
        return False
