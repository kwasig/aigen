from fastapi import APIRouter, HTTPException, Depends, Request, Body, File, UploadFile, Form
from fastapi.responses import StreamingResponse
import asyncio
import json
import traceback
from datetime import datetime, timezone
import uuid
from typing import Optional, List, Dict, Any
import jwt
from pydantic import BaseModel
import tempfile
import os
from pathlib import Path
import time

from agentpress.thread_manager import ThreadManager
from services.supabase import DBConnection
from services import redis
from agent.run import run_agent
from utils.auth_utils import get_current_user_id_from_jwt, get_user_id_from_stream_auth, verify_thread_access
from utils.logger import logger
from services.billing import check_billing_status, can_use_model
from utils.config import config
from sandbox.sandbox import create_sandbox, get_or_start_sandbox
from services.llm import make_llm_api_call
from utils.profiling import profile
from run_agent_background import run_agent_background, _cleanup_redis_response_list, update_agent_run_status

# Initialize shared resources
router = APIRouter()
thread_manager = None
db = None
instance_id = None # Global instance ID for this backend instance

# TTL for Redis response lists (24 hours)
REDIS_RESPONSE_LIST_TTL = 3600 * 24

DEFAULT_MODEL_NAME_ALIASES = {
    # Short names to full names
    "sonnet-3.7": "anthropic/claude-3-7-sonnet-latest",
    # "gpt-4.1": "openai/gpt-4.1-2025-04-14",  # Commented out in constants.py
    "gpt-4o": "openai/gpt-4o",
    # "gpt-4-turbo": "openai/gpt-4-turbo",  # Commented out in constants.py
    # "gpt-4": "openai/gpt-4",  # Commented out in constants.py
    # "gemini-flash-2.5": "openrouter/google/gemini-2.5-flash-preview",  # Commented out in constants.py
    # "grok-3": "xai/grok-3-fast-latest",  # Commented out in constants.py
    "deepseek": "openrouter/deepseek/deepseek-chat",
    # "deepseek-r1": "openrouter/deepseek/deepseek-r1",
    # "grok-3-mini": "xai/grok-3-mini-fast-beta",  # Commented out in constants.py
    "qwen3": "openrouter/qwen/qwen3-235b-a22b",  # Commented out in constants.py



    # Also include full names as keys to ensure they map to themselves
    "anthropic/claude-3-7-sonnet-latest": "anthropic/claude-3-7-sonnet-latest",
    # "openai/gpt-4.1-2025-04-14": "openai/gpt-4.1-2025-04-14",  # Commented out in constants.py
    "openai/gpt-4o": "openai/gpt-4o",
    # "openai/gpt-4-turbo": "openai/gpt-4-turbo",  # Commented out in constants.py
    # "openai/gpt-4": "openai/gpt-4",  # Commented out in constants.py
    # "openrouter/google/gemini-2.5-flash-preview": "openrouter/google/gemini-2.5-flash-preview",  # Commented out in constants.py
    # "xai/grok-3-fast-latest": "xai/grok-3-fast-latest",  # Commented out in constants.py
    "deepseek/deepseek-chat": "openrouter/deepseek/deepseek-chat",
    # "deepseek/deepseek-r1": "openrouter/deepseek/deepseek-r1",

    "qwen/qwen3-235b-a22b": "openrouter/qwen/qwen3-235b-a22b",
    # "xai/grok-3-mini-fast-beta": "xai/grok-3-mini-fast-beta",  # Commented out in constants.py
}

# Mutable alias dictionary used by the API
MODEL_NAME_ALIASES = DEFAULT_MODEL_NAME_ALIASES.copy()

# Path to additional provider definitions
PROVIDERS_FILE = Path(__file__).with_name("providers.json")

def load_extra_providers() -> None:
    if PROVIDERS_FILE.exists():
        try:
            with open(PROVIDERS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    MODEL_NAME_ALIASES.update(data)
        except Exception as e:
            logger.error(f"Failed to load providers file: {e}")

def save_extra_providers(extra: Dict[str, str]) -> None:
    try:
        with open(PROVIDERS_FILE, "w") as f:
            json.dump(extra, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save providers file: {e}")

load_extra_providers()

# Utility to update the backend .env file with new API keys
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

def update_env_file(var: str, value: str) -> None:
    lines = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text().splitlines()
    updated = False
    for idx, line in enumerate(lines):
        if line.startswith(f"{var}="):
            lines[idx] = f"{var}={value}"
            updated = True
            break
    if not updated:
        lines.append(f"{var}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n")

class AgentStartRequest(BaseModel):
    model_name: Optional[str] = None  # Will be set from config.MODEL_TO_USE in the endpoint
    enable_thinking: Optional[bool] = True
    reasoning_effort: Optional[str] = 'medium'
    stream: Optional[bool] = True
    enable_context_manager: Optional[bool] = False

class InitiateAgentResponse(BaseModel):
    thread_id: str
    agent_run_id: Optional[str] = None


class AddProviderRequest(BaseModel):
    alias: str
    model_name: str
    env_var_name: str
    api_key: str

def initialize(
    _thread_manager: ThreadManager,
    _db: DBConnection,
    _instance_id: str = None
):
    """Initialize the agent API with resources from the main API."""
    global thread_manager, db, instance_id
    thread_manager = _thread_manager
    db = _db

    # Use provided instance_id or generate a new one
    if _instance_id:
        instance_id = _instance_id
    else:
        # Generate instance ID
        instance_id = str(uuid.uuid4())[:8]

    logger.info(f"Initialized agent API with instance ID: {instance_id}")

    # Note: Redis will be initialized in the lifespan function in api.py

async def cleanup():
    """Clean up resources and stop running agents on shutdown."""
    logger.info("Starting cleanup of agent API resources")

    # Use the instance_id to find and clean up this instance's keys
    try:
        if instance_id: # Ensure instance_id is set
            running_keys = await redis.keys(f"active_run:{instance_id}:*")
            logger.info(f"Found {len(running_keys)} running agent runs for instance {instance_id} to clean up")

            for key in running_keys:
                # Key format: active_run:{instance_id}:{agent_run_id}
                parts = key.split(":")
                if len(parts) == 3:
                    agent_run_id = parts[2]
                    await stop_agent_run(agent_run_id, error_message=f"Instance {instance_id} shutting down")
                else:
                    logger.warning(f"Unexpected key format found: {key}")
        else:
            logger.warning("Instance ID not set, cannot clean up instance-specific agent runs.")

    except Exception as e:
        logger.error(f"Failed to clean up running agent runs: {str(e)}")

    # Close Redis connection
    await redis.close()
    logger.info("Completed cleanup of agent API resources")

async def stop_agent_run(agent_run_id: str, error_message: Optional[str] = None):
    """Update database and publish stop signal to Redis."""
    logger.info(f"Stopping agent run: {agent_run_id}")
    client = await db.client
    final_status = "failed" if error_message else "stopped"

    # Attempt to fetch final responses from Redis
    response_list_key = f"agent_run:{agent_run_id}:responses"
    all_responses = []
    try:
        all_responses_json = await redis.lrange(response_list_key, 0, -1)
        all_responses = [json.loads(r) for r in all_responses_json]
        logger.info(f"Fetched {len(all_responses)} responses from Redis for DB update on stop/fail: {agent_run_id}")
    except Exception as e:
        logger.error(f"Failed to fetch responses from Redis for {agent_run_id} during stop/fail: {e}")
        # Try fetching from DB as a fallback? Or proceed without responses? Proceeding without for now.

    # Update the agent run status in the database
    update_success = await update_agent_run_status(
        client, agent_run_id, final_status, error=error_message, responses=all_responses
    )

    if not update_success:
        logger.error(f"Failed to update database status for stopped/failed run {agent_run_id}")

    # Send STOP signal to the global control channel
    global_control_channel = f"agent_run:{agent_run_id}:control"
    try:
        await redis.publish(global_control_channel, "STOP")
        logger.debug(f"Published STOP signal to global channel {global_control_channel}")
    except Exception as e:
        logger.error(f"Failed to publish STOP signal to global channel {global_control_channel}: {str(e)}")

    # Find all instances handling this agent run and send STOP to instance-specific channels
    try:
        instance_keys = await redis.keys(f"active_run:*:{agent_run_id}")
        logger.debug(f"Found {len(instance_keys)} active instance keys for agent run {agent_run_id}")

        for key in instance_keys:
            # Key format: active_run:{instance_id}:{agent_run_id}
            parts = key.split(":")
            if len(parts) == 3:
                instance_id_from_key = parts[1]
                instance_control_channel = f"agent_run:{agent_run_id}:control:{instance_id_from_key}"
                try:
                    await redis.publish(instance_control_channel, "STOP")
                    logger.debug(f"Published STOP signal to instance channel {instance_control_channel}")
                except Exception as e:
                    logger.warning(f"Failed to publish STOP signal to instance channel {instance_control_channel}: {str(e)}")
            else:
                 logger.warning(f"Unexpected key format found: {key}")

        # Clean up the response list immediately on stop/fail
        await _cleanup_redis_response_list(agent_run_id)

    except Exception as e:
        logger.error(f"Failed to find or signal active instances for {agent_run_id}: {str(e)}")

    logger.info(f"Successfully initiated stop process for agent run: {agent_run_id}")

# async def restore_running_agent_runs():
#     """Mark agent runs that were still 'running' in the database as failed and clean up Redis resources."""
#     logger.info("Restoring running agent runs after server restart")
#     client = await db.client
#     running_agent_runs = await client.table('agent_runs').select('id').eq("status", "running").execute()

#     for run in running_agent_runs.data:
#         agent_run_id = run['id']
#         logger.warning(f"Found running agent run {agent_run_id} from before server restart")

#         # Clean up Redis resources for this run
#         try:
#             # Clean up active run key
#             active_run_key = f"active_run:{instance_id}:{agent_run_id}"
#             await redis.delete(active_run_key)

#             # Clean up response list
#             response_list_key = f"agent_run:{agent_run_id}:responses"
#             await redis.delete(response_list_key)

#             # Clean up control channels
#             control_channel = f"agent_run:{agent_run_id}:control"
#             instance_control_channel = f"agent_run:{agent_run_id}:control:{instance_id}"
#             await redis.delete(control_channel)
#             await redis.delete(instance_control_channel)

#             logger.info(f"Cleaned up Redis resources for agent run {agent_run_id}")
#         except Exception as e:
#             logger.error(f"Error cleaning up Redis resources for agent run {agent_run_id}: {e}")

#         # Call stop_agent_run to handle status update and cleanup
#         await stop_agent_run(agent_run_id, error_message="Server restarted while agent was running")

async def check_for_active_project_agent_run(client, project_id: str):
    """
    Check if there is an active agent run for any thread in the given project.
    If found, returns the ID of the active run, otherwise returns None.
    """
    project_threads = await client.table('threads').select('thread_id').eq('project_id', project_id).execute()
    project_thread_ids = [t['thread_id'] for t in project_threads.data]

    if project_thread_ids:
        active_runs = await client.table('agent_runs').select('id').in_('thread_id', project_thread_ids).eq('status', 'running').execute()
        if active_runs.data and len(active_runs.data) > 0:
            return active_runs.data[0]['id']
    return None

async def get_agent_run_with_access_check(client, agent_run_id: str, user_id: str):
    """Get agent run data after verifying user access."""
    agent_run = await client.table('agent_runs').select('*').eq('id', agent_run_id).execute()
    if not agent_run.data:
        raise HTTPException(status_code=404, detail="Agent run not found")

    agent_run_data = agent_run.data[0]
    thread_id = agent_run_data['thread_id']
    await verify_thread_access(client, thread_id, user_id)
    return agent_run_data


@router.post("/models/providers")
async def add_llm_provider(request: AddProviderRequest, user_id: str = Depends(get_current_user_id_from_jwt)):
    """Add a new LLM provider and store its API key."""
    alias = request.alias.strip()
    model = request.model_name.strip()
    env_var = request.env_var_name.strip().upper()
    api_key = request.api_key.strip()

    if not alias or not model or not env_var or not api_key:
        raise HTTPException(status_code=400, detail="Invalid provider details")

    MODEL_NAME_ALIASES[alias] = model
    extras = {k: v for k, v in MODEL_NAME_ALIASES.items() if k not in DEFAULT_MODEL_NAME_ALIASES}
    save_extra_providers(extras)

    try:
        update_env_file(env_var, api_key)
        os.environ[env_var] = api_key
    except Exception as e:
        logger.error(f"Failed to update env file: {e}")
        raise HTTPException(status_code=500, detail="Failed to update environment file")

    return {"status": "success", "alias": alias, "model": model}

from utils.profiling import profile

@router.post("/thread/{thread_id}/agent/start")
@profile
async def start_agent(
    thread_id: str,
    body: AgentStartRequest = Body(...),
    user_id: str = Depends(get_current_user_id_from_jwt)
):
    """Start an agent for a specific thread in the background."""
    start_time = time.time()
    global instance_id # Ensure instance_id is accessible
    if not instance_id:
        raise HTTPException(status_code=500, detail="Agent API not initialized with instance ID")

    # Use model from config if not specified in the request
    model_name = body.model_name
    logger.info(f"Original model_name from request: {model_name}")

    if model_name is None:
        model_name = config.MODEL_TO_USE
        logger.info(f"Using model from config: {model_name}")

    # Log the model name after alias resolution
    resolved_model = MODEL_NAME_ALIASES.get(model_name, model_name)
    logger.info(f"Resolved model name: {resolved_model}")

    # Update model_name to use the resolved version
    model_name = resolved_model

    logger.info(f"Starting new agent for thread: {thread_id} with config: model={model_name}, thinking={body.enable_thinking}, effort={body.reasoning_effort}, stream={body.stream}, context_manager={body.enable_context_manager} (Instance: {instance_id})")
    client = await db.client

    # Create tasks for parallel execution to reduce latency
    # 1. Access check
    access_check_task = asyncio.create_task(verify_thread_access(client, thread_id, user_id))
    # 2. Get thread data
    thread_data_task = asyncio.create_task(client.table('threads').select('project_id', 'account_id').eq('thread_id', thread_id).execute())
    
    # Wait for both tasks to complete
    try:
        done, pending = await asyncio.wait(
            [access_check_task, thread_data_task],
            timeout=5.0,  # Add timeout to prevent hanging
            return_when=asyncio.ALL_COMPLETED
        )
        # Cancel any pending tasks
        for task in pending:
            task.cancel()
            
        # Check for exceptions
        for task in done:
            if task.exception():
                raise task.exception()
    except Exception as e:
        logger.error(f"Error during parallel thread checks: {e}")
        raise HTTPException(status_code=500, detail=f"Error verifying thread access: {str(e)}")

    # Process thread data result
    thread_result = thread_data_task.result()
    if not thread_result.data:
        raise HTTPException(status_code=404, detail="Thread not found")
    thread_data = thread_result.data[0]
    project_id = thread_data.get('project_id')
    account_id = thread_data.get('account_id')

    # Run billing and model checks in parallel
    billing_check_task = asyncio.create_task(check_billing_status(client, account_id))
    model_check_task = asyncio.create_task(can_use_model(client, account_id, model_name))
    active_run_check_task = asyncio.create_task(check_for_active_project_agent_run(client, project_id))
    
    # Wait for all tasks to complete
    try:
        done, pending = await asyncio.wait(
            [billing_check_task, model_check_task, active_run_check_task],
            timeout=5.0,
            return_when=asyncio.ALL_COMPLETED
        )
        # Cancel any pending tasks
        for task in pending:
            task.cancel()
            
        # Check for exceptions
        for task in done:
            if task.exception():
                raise task.exception()
    except Exception as e:
        logger.error(f"Error during parallel checks: {e}")
        raise HTTPException(status_code=500, detail=f"Error during authorization checks: {str(e)}")

    # Process results
    can_use, model_message, allowed_models = model_check_task.result()
    if not can_use:
        raise HTTPException(status_code=403, detail={"message": model_message, "allowed_models": allowed_models})

    can_run, message, subscription = billing_check_task.result()
    if not can_run:
        raise HTTPException(status_code=402, detail={"message": message, "subscription": subscription})

    active_run_id = active_run_check_task.result()
    if active_run_id:
        logger.info(f"Stopping existing agent run {active_run_id} for project {project_id}")
        await stop_agent_run(active_run_id)

    # Fast path: Skip sandbox initialization and register agent run immediately
    # This will let the frontend see progress faster
    agent_run = await client.table('agent_runs').insert({
        "thread_id": thread_id, 
        "status": "initializing",  # Start with initializing to indicate progress
        "started_at": datetime.now(timezone.utc).isoformat()
    }).execute()
    agent_run_id = agent_run.data[0]['id']
    logger.info(f"Created new agent run: {agent_run_id}")

    # Register this run in Redis with TTL using instance ID
    instance_key = f"active_run:{instance_id}:{agent_run_id}"
    try:
        await redis.set(instance_key, "running", ex=redis.REDIS_KEY_TTL)
    except Exception as e:
        logger.warning(f"Failed to register agent run in Redis ({instance_key}): {str(e)}")

    # Start a background task to handle sandbox initialization
    # This allows us to return a response to the frontend immediately
    asyncio.create_task(
        initialize_sandbox_and_start_agent(
            agent_run_id=agent_run_id,
            thread_id=thread_id,
            instance_id=instance_id,
            project_id=project_id,
            model_name=model_name,
            enable_thinking=body.enable_thinking,
            reasoning_effort=body.reasoning_effort,
            stream=body.stream,
            enable_context_manager=body.enable_context_manager
        )
    )

    process_time = time.time() - start_time
    logger.info(f"start_agent completed in {process_time:.2f}s, returning response")
    
    return {"agent_run_id": agent_run_id, "status": "initializing"}


async def initialize_sandbox_and_start_agent(
    agent_run_id: str,
    thread_id: str,
    instance_id: str,
    project_id: str,
    model_name: str,
    enable_thinking: bool,
    reasoning_effort: str,
    stream: bool,
    enable_context_manager: bool
):
    """Background task to initialize sandbox and start agent."""
    client = await db.client
    
    try:
        # Get project data to find sandbox ID
        project_result = await client.table('projects').select('*').eq('project_id', project_id).execute()
        if not project_result.data:
            logger.error(f"Project not found: {project_id}")
            await update_agent_run_status(client, agent_run_id, "failed", error="Project not found")
            return
        
        project_data = project_result.data[0]
        sandbox_info = project_data.get('sandbox', {})
        if not sandbox_info.get('id'):
            logger.error(f"No sandbox found for project: {project_id}")
            await update_agent_run_status(client, agent_run_id, "failed", error="No sandbox found for this project")
            return
            
        sandbox_id = sandbox_info['id']
        
        # Update status to show progress
        await update_agent_run_status(
            client, 
            agent_run_id, 
            "preparing",  # Transitional status
            responses=[{"type": "status", "status": "preparing", "message": "Starting sandbox..."}]
        )
        
        # Start sandbox
        try:
            sandbox = await get_or_start_sandbox(sandbox_id)
            logger.info(f"Successfully started sandbox {sandbox_id} for project {project_id}")
        except Exception as e:
            logger.error(f"Failed to start sandbox for project {project_id}: {str(e)}")
            await update_agent_run_status(client, agent_run_id, "failed", error=f"Failed to initialize sandbox: {str(e)}")
            return

        # Update to running status
        await update_agent_run_status(
            client, 
            agent_run_id, 
            "running",
            responses=[{"type": "status", "status": "running", "message": "Agent is running..."}]
        )
        
        # Run the agent in the background
        run_agent_background.send(
            agent_run_id=agent_run_id, thread_id=thread_id, instance_id=instance_id,
            project_id=project_id,
            model_name=model_name,  # Already resolved above
            enable_thinking=enable_thinking, reasoning_effort=reasoning_effort,
            stream=stream, enable_context_manager=enable_context_manager
        )
        
    except Exception as e:
        logger.error(f"Error in sandbox initialization for agent run {agent_run_id}: {str(e)}")
        await update_agent_run_status(client, agent_run_id, "failed", error=f"Failed to start agent: {str(e)}")


@router.get("/agent-run/{agent_run_id}/stream")
async def stream_agent_run(
    agent_run_id: str,
    token: Optional[str] = None,
    request: Request = None
):
    """Stream the responses of an agent run using Redis Lists and Pub/Sub."""
    stream_start_time = time.time()
    logger.info(f"Starting stream for agent run: {agent_run_id}")
    
    # Get DB and Redis clients in parallel
    client_task = asyncio.create_task(db.client)
    redis_client_task = asyncio.create_task(redis.get_client())

    # Prepare Redis keys/channels in advance
    response_list_key = f"agent_run:{agent_run_id}:responses"
    response_channel = f"agent_run:{agent_run_id}:new_response"
    control_channel = f"agent_run:{agent_run_id}:control" # Global control channel
    
    # Start auth check in parallel too
    auth_task = asyncio.create_task(get_user_id_from_stream_auth(request, token))
    
    # Wait for all tasks with a timeout to prevent blocking
    try:
        done, pending = await asyncio.wait(
            [client_task, redis_client_task, auth_task],
            timeout=1.0,  # Short timeout for faster response
            return_when=asyncio.ALL_COMPLETED
        )
        
        # Process completed tasks
        client = None
        user_id = None
        
        for task in done:
            if task == client_task and not task.exception():
                client = task.result()
            elif task == auth_task and not task.exception():
                user_id = task.result()
        
        # Cancel any pending tasks
        for task in pending:
            task.cancel()
            
        # Handle missing results
        if client is None:
            client = await db.client  # Fallback if the parallel task failed
            
        if user_id is None:
            # Auth failed or timed out
            logger.warning(f"Auth timeout for agent run {agent_run_id}, treating as unauthenticated")
            raise HTTPException(status_code=401, detail="Authentication timed out")
            
    except Exception as e:
        logger.error(f"Error in parallel initialization: {e}")
        # Fallback to sequential initialization
        client = await db.client
        user_id = await get_user_id_from_stream_auth(request, token)
    
    # Check if the user has access to this agent run
    try:
        # Use a shorter timeout for access check
        agent_run_data = await asyncio.wait_for(
            get_agent_run_with_access_check(client, agent_run_id, user_id),
            timeout=1.0
        )
    except asyncio.TimeoutError:
        logger.warning(f"Access check timeout for agent run {agent_run_id}")
        raise HTTPException(status_code=408, detail="Request timeout during access check")
    except HTTPException as e:
        raise e  # Re-raise HTTP exceptions directly
    except Exception as e:
        logger.error(f"Error getting agent run: {e}")
        raise HTTPException(status_code=500, detail=f"Error accessing agent run: {str(e)}")

    # Define helper function to set up pubsub channels
    async def setup_pubsub_channels():
        psr = await redis.create_pubsub()
        await psr.subscribe(response_channel)
        
        psc = await redis.create_pubsub()
        await psc.subscribe(control_channel)
        
        return psr, psc

    # Log performance metric for initial request processing
    initial_process_time = time.time() - stream_start_time
    logger.info(f"Initial stream setup completed in {initial_process_time:.2f}s, starting stream generation")

    async def stream_generator():
        logger.debug(f"Streaming responses for {agent_run_id}")
        last_processed_index = -1
        pubsub_response = None
        pubsub_control = None
        listener_task = None
        terminate_stream = False
        initial_yield_complete = False
        generator_start_time = time.time()
        message_queue = asyncio.Queue()

        # Define the polling function first before using it
        async def poll_for_new_responses(interval):
            """Fallback polling mechanism when pubsub isn't available."""
            logger.info(f"Starting polling fallback for {agent_run_id} with interval {interval}s")
            last_check_time = time.time()
            # Initial poll immediately
            await message_queue.put({"type": "new_response"})
            
            while not terminate_stream:
                try:
                    # Check Redis directly - use a shorter interval for better responsiveness
                    await asyncio.sleep(interval * 0.5)  # Poll more frequently
                    await message_queue.put({"type": "new_response"})
                    
                    # Also check run status periodically (every 5 polling intervals)
                    current_time = time.time()
                    if current_time - last_check_time > interval * 3:  # Check status more frequently
                        last_check_time = current_time
                        status_result = await client.table('agent_runs').select('status').eq("id", agent_run_id).maybe_single().execute()
                        if status_result.data:
                            status = status_result.data.get('status')
                            if status in ['completed', 'failed', 'stopped']:
                                await message_queue.put({"type": "control", "data": status.upper()})
                                return  # Stop polling
                except Exception as e:
                    logger.error(f"Error in polling fallback: {e}")
                    await asyncio.sleep(interval * 2)  # Back off on error

        try:
            # 1. Immediately yield an initial status message to reduce perceived latency
            # This lets the client know we're starting to process
            initial_status = {"type": "status", "status": "connecting", "message": "Starting up agent connection..."}
            yield f"data: {json.dumps(initial_status)}\n\n"
            initial_yield_complete = True  # Mark that we've sent at least one message
            
            # Yield current agent status immediately based on database record
            agent_status = {"type": "status", "status": agent_run_data['status'], "message": f"Agent status: {agent_run_data['status']}"}
            yield f"data: {json.dumps(agent_status)}\n\n"
            
            # Start a background task for initial data fetching
            polling_task = asyncio.create_task(poll_for_new_responses(0.2))  # Start polling immediately with a short interval
            
            # 2. Fetch and yield initial responses from Redis list with a timeout
            try:
                initial_responses_json = await asyncio.wait_for(
                    redis.lrange(response_list_key, 0, -1), 
                    timeout=1.0  # Reduced timeout for better responsiveness
                )
                initial_responses = []
                if initial_responses_json:
                    initial_responses = [json.loads(r) for r in initial_responses_json]
                    logger.debug(f"Sending {len(initial_responses)} initial responses for {agent_run_id}")
                    for response in initial_responses:
                        yield f"data: {json.dumps(response)}\n\n"
                    last_processed_index = len(initial_responses) - 1
                elif agent_run_data['status'] not in ['running', 'initializing', 'preparing']:
                    # If no responses and not in a running state, yield the status
                    status_response = {
                        "type": "status", 
                        "status": agent_run_data['status'],
                        "message": f"Agent is in {agent_run_data['status']} state"
                    }
                    yield f"data: {json.dumps(status_response)}\n\n"
                    
                initial_yield_complete = True
                # Log how long it took to deliver first response
                first_response_time = time.time() - generator_start_time
                logger.info(f"First responses delivered in {first_response_time:.2f}s for {agent_run_id}")
                
            except asyncio.TimeoutError:
                logger.warning(f"Initial Redis fetch timed out for {agent_run_id}")
                initial_yield_complete = True  # Still mark as complete to continue
                # Yield a fallback message
                fallback_status = {"type": "status", "status": "connecting", "message": "Still connecting..."}
                yield f"data: {json.dumps(fallback_status)}\n\n"
            
            # 3. Check run status in parallel with Pub/Sub setup
            status_check_task = asyncio.create_task(
                client.table('agent_runs').select('status').eq("id", agent_run_id).maybe_single().execute()
            )

            # 4. Set up Pub/Sub listeners for new responses and control signals
            pubsub_setup_task = asyncio.create_task(setup_pubsub_channels())
            
            # Wait for both tasks with timeout
            try:
                done, pending = await asyncio.wait(
                    [status_check_task, pubsub_setup_task],
                    timeout=1.0,  # Reduced timeout for better responsiveness
                    return_when=asyncio.ALL_COMPLETED
                )
                # Cancel any pending tasks
                for task in pending:
                    task.cancel()
                
                # Process status check result
                if status_check_task in done and not status_check_task.exception():
                    run_status = status_check_task.result()
                    current_status = run_status.data.get('status') if run_status.data else None
                    
                    if current_status != 'running' and current_status not in ['initializing', 'preparing']:
                        logger.info(f"Agent run {agent_run_id} is not running (status: {current_status}). Ending stream.")
                        yield f"data: {json.dumps({'type': 'status', 'status': current_status})}\n\n"
                        return
                
                # Process pubsub setup result
                if pubsub_setup_task in done and not pubsub_setup_task.exception():
                    pubsub_response, pubsub_control = pubsub_setup_task.result()
                else:
                    logger.warning(f"Pubsub setup failed for {agent_run_id}")
                    # Continue anyway - we'll check again if pubsub is available
                
            except Exception as e:
                logger.warning(f"Error setting up status check or pubsub: {e}")
                # Continue anyway to show existing messages
            
            # If pubsub setup failed, try again with shorter timeout
            if pubsub_response is None or pubsub_control is None:
                try:
                    pubsub_response, pubsub_control = await asyncio.wait_for(
                        setup_pubsub_channels(),
                        timeout=1.0  # Faster timeout for better responsiveness
                    )
                except Exception as e:
                    logger.warning(f"Second pubsub setup attempt failed: {e}")
                    # Continue without pubsub - we'll use polling fallback
            
            # Queue to communicate between listeners and the main generator loop
            message_queue = asyncio.Queue()

            async def listen_messages():
                try:
                    if pubsub_response is None or pubsub_control is None:
                        logger.warning("Cannot start message listener as pubsub is not available")
                        return
                        
                    response_reader = pubsub_response.listen()
                    control_reader = pubsub_control.listen()
                    tasks = [
                        asyncio.create_task(response_reader.__anext__()), 
                        asyncio.create_task(control_reader.__anext__())
                    ]

                    while not terminate_stream:
                        done, pending = await asyncio.wait(
                            tasks, 
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=0.5  # Add timeout to prevent blocking indefinitely
                        )
                        
                        # Check if we timed out
                        if not done:
                            await asyncio.sleep(0.01)  # Small sleep to prevent tight loop
                            continue
                            
                        for task in done:
                            try:
                                message = task.result()
                                if message and isinstance(message, dict) and message.get("type") == "message":
                                    channel = message.get("channel")
                                    data = message.get("data")
                                    if isinstance(data, bytes): data = data.decode('utf-8')

                                    if channel == response_channel and data == "new":
                                        await message_queue.put({"type": "new_response"})
                                    elif channel == control_channel and data in ["STOP", "END_STREAM", "ERROR"]:
                                        logger.info(f"Received control signal '{data}' for {agent_run_id}")
                                        await message_queue.put({"type": "control", "data": data})
                                        return # Stop listening on control signal

                            except StopAsyncIteration:
                                logger.warning(f"Listener {task} stopped.")
                                return
                            except Exception as e:
                                logger.error(f"Error in listener for {agent_run_id}: {e}")
                                return
                            finally:
                                # Reschedule the completed listener task
                                if task in tasks:
                                    tasks.remove(task)
                                    if message and isinstance(message, dict):
                                        if message.get("channel") == response_channel:
                                            tasks.append(asyncio.create_task(response_reader.__anext__()))
                                        elif message.get("channel") == control_channel:
                                            tasks.append(asyncio.create_task(control_reader.__anext__()))

                    # Cancel pending listener tasks on exit
                    for p_task in pending: p_task.cancel()
                    for task in tasks: task.cancel()
                    
                except Exception as e:
                    logger.error(f"Fatal error in message listener: {e}")
                    await message_queue.put({"type": "error", "data": f"Listener failed: {str(e)}"})

            # Start message listener if pubsub is available
            if pubsub_response is not None and pubsub_control is not None:
                listener_task = asyncio.create_task(listen_messages())
            else:
                # Set up polling fallback for when pubsub isn't available
                polling_interval = 0.5  # seconds
                asyncio.create_task(poll_for_new_responses(polling_interval))
                
            # 4. Main loop to process messages from the queue with timeout protection
            while not terminate_stream:
                try:
                    # Use wait_for with timeout to prevent blocking forever
                    queue_item = await asyncio.wait_for(message_queue.get(), timeout=0.5)

                    if queue_item["type"] == "new_response":
                        # Fetch new responses from Redis list starting after the last processed index
                        new_start_index = last_processed_index + 1
                        try:
                            new_responses_json = await asyncio.wait_for(
                                redis.lrange(response_list_key, new_start_index, -1),
                                timeout=1.0
                            )

                            if new_responses_json:
                                new_responses = [json.loads(r) for r in new_responses_json]
                                num_new = len(new_responses)
                                for response in new_responses:
                                    yield f"data: {json.dumps(response)}\n\n"
                                    # Check if this response signals completion
                                    if response.get('type') == 'status' and response.get('status') in ['completed', 'failed', 'stopped']:
                                        logger.info(f"Detected run completion via status message in stream: {response.get('status')}")
                                        terminate_stream = True
                                        break # Stop processing further new responses
                                last_processed_index += num_new
                            if terminate_stream: break
                        except asyncio.TimeoutError:
                            logger.warning(f"Redis lrange timed out for {agent_run_id}")
                            # Continue processing - don't terminate

                    elif queue_item["type"] == "control":
                        control_signal = queue_item["data"]
                        terminate_stream = True # Stop the stream on any control signal
                        yield f"data: {json.dumps({'type': 'status', 'status': control_signal})}\n\n"
                        break

                    elif queue_item["type"] == "error":
                        logger.error(f"Listener error for {agent_run_id}: {queue_item['data']}")
                        terminate_stream = True
                        yield f"data: {json.dumps({'type': 'status', 'status': 'error'})}\n\n"
                        break

                except asyncio.TimeoutError:
                    # No new messages in queue, just continue the loop
                    await asyncio.sleep(0.01)  # Small sleep to prevent tight loop
                except asyncio.CancelledError:
                    logger.info(f"Stream generator main loop cancelled for {agent_run_id}")
                    terminate_stream = True
                    break
                except Exception as loop_err:
                    logger.error(f"Error in stream generator main loop for {agent_run_id}: {loop_err}", exc_info=True)
                    terminate_stream = True
                    yield f"data: {json.dumps({'type': 'status', 'status': 'error', 'message': f'Stream failed: {loop_err}'})}\n\n"
                    break

        except Exception as e:
            logger.error(f"Error setting up stream for agent run {agent_run_id}: {e}", exc_info=True)
            # Only yield error if initial yield didn't happen
            if not initial_yield_complete:
                 yield f"data: {json.dumps({'type': 'status', 'status': 'error', 'message': f'Failed to start stream: {e}'})}\n\n"
        finally:
            terminate_stream = True
            # Graceful shutdown order: unsubscribe → close → cancel
            try:
                if pubsub_response: 
                    await asyncio.wait_for(pubsub_response.unsubscribe(response_channel), timeout=1.0)
                if pubsub_control: 
                    await asyncio.wait_for(pubsub_control.unsubscribe(control_channel), timeout=1.0)
                if pubsub_response: 
                    await asyncio.wait_for(pubsub_response.close(), timeout=1.0)
                if pubsub_control: 
                    await asyncio.wait_for(pubsub_control.close(), timeout=1.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Error during pubsub cleanup: {e}")

            if listener_task:
                listener_task.cancel()
                try:
                    await asyncio.wait_for(listener_task, timeout=0.5)  # Short timeout for cancellation
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass
                    
            logger.debug(f"Streaming cleanup complete for agent run: {agent_run_id}")
            
            # Log total stream time
            total_stream_time = time.time() - generator_start_time
            logger.info(f"Total stream time for {agent_run_id}: {total_stream_time:.2f}s")

    # Set response headers to avoid buffering and optimize for streaming
    return StreamingResponse(
        stream_generator(), 
        media_type="text/event-stream", 
        headers={
            "Cache-Control": "no-cache, no-transform", 
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no", 
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            # Add additional headers to help with client rendering
            "X-Stream-ID": agent_run_id
        }
    )

async def generate_and_update_project_name(project_id: str, prompt: str):
    """Generates a project name using an LLM and updates the database."""
    logger.info(f"Starting background task to generate name for project: {project_id}")
    try:
        db_conn = DBConnection()
        client = await db_conn.client

        model_name = "openai/gpt-4o-mini"
        system_prompt = "You are a helpful assistant that generates extremely concise titles (2-4 words maximum) for chat threads based on the user's message. Respond with only the title, no other text or punctuation."
        user_message = f"Generate an extremely brief title (2-4 words only) for a chat thread that starts with this message: \"{prompt}\""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]

        logger.debug(f"Calling LLM ({model_name}) for project {project_id} naming.")
        response = await make_llm_api_call(messages=messages, model_name=model_name, max_tokens=20, temperature=0.7)

        generated_name = None
        if response and response.get('choices') and response['choices'][0].get('message'):
            raw_name = response['choices'][0]['message'].get('content', '').strip()
            cleaned_name = raw_name.strip('\'" \n\t')
            if cleaned_name:
                generated_name = cleaned_name
                logger.info(f"LLM generated name for project {project_id}: '{generated_name}'")
            else:
                logger.warning(f"LLM returned an empty name for project {project_id}.")
        else:
            logger.warning(f"Failed to get valid response from LLM for project {project_id} naming. Response: {response}")

        if generated_name:
            update_result = await client.table('projects').update({"name": generated_name}).eq("project_id", project_id).execute()
            if hasattr(update_result, 'data') and update_result.data:
                logger.info(f"Successfully updated project {project_id} name to '{generated_name}'")
            else:
                logger.error(f"Failed to update project {project_id} name in database. Update result: {update_result}")
        else:
            logger.warning(f"No generated name, skipping database update for project {project_id}.")

    except Exception as e:
        logger.error(f"Error in background naming task for project {project_id}: {str(e)}\n{traceback.format_exc()}")
    finally:
        # No need to disconnect DBConnection singleton instance here
        logger.info(f"Finished background naming task for project: {project_id}")

@router.post("/agent/initiate", response_model=InitiateAgentResponse)
@profile
async def initiate_agent_with_files(
    prompt: str = Form(...),
    model_name: Optional[str] = Form(None),  # Default to None to use config.MODEL_TO_USE
    enable_thinking: Optional[bool] = Form(True),
    reasoning_effort: Optional[str] = Form("medium"),
    stream: Optional[bool] = Form(True),
    enable_context_manager: Optional[bool] = Form(False),
    files: List[UploadFile] = File(default=[]),
    user_id: str = Depends(get_current_user_id_from_jwt)
):
    """Initiate a new agent session with optional file attachments."""
    global instance_id # Ensure instance_id is accessible
    if not instance_id:
        raise HTTPException(status_code=500, detail="Agent API not initialized with instance ID")

    # Use model from config if not specified in the request
    logger.info(f"Original model_name from request: {model_name}")

    if model_name is None:
        model_name = config.MODEL_TO_USE
        logger.info(f"Using model from config: {model_name}")

    # Log the model name after alias resolution
    resolved_model = MODEL_NAME_ALIASES.get(model_name, model_name)
    logger.info(f"Resolved model name: {resolved_model}")

    # Update model_name to use the resolved version
    model_name = resolved_model

    logger.info(f"[\033[91mDEBUG\033[0m] Initiating new agent with prompt and {len(files)} files (Instance: {instance_id}), model: {model_name}, enable_thinking: {enable_thinking}")
    client = await db.client
    account_id = user_id # In Basejump, personal account_id is the same as user_id
    
    can_use, model_message, allowed_models = await can_use_model(client, account_id, model_name)
    if not can_use:
        raise HTTPException(status_code=403, detail={"message": model_message, "allowed_models": allowed_models})

    can_run, message, subscription = await check_billing_status(client, account_id)
    if not can_run:
        raise HTTPException(status_code=402, detail={"message": message, "subscription": subscription})

    try:
        # 1. Create Project
        placeholder_name = f"{prompt[:30]}..." if len(prompt) > 30 else prompt
        project = await client.table('projects').insert({
            "project_id": str(uuid.uuid4()), "account_id": account_id, "name": placeholder_name,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        project_id = project.data[0]['project_id']
        logger.info(f"Created new project: {project_id}")

        # 2. Create Thread
        thread = await client.table('threads').insert({
            "thread_id": str(uuid.uuid4()), "project_id": project_id, "account_id": account_id,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        thread_id = thread.data[0]['thread_id']
        logger.info(f"Created new thread: {thread_id}")

        # Trigger Background Naming Task
        asyncio.create_task(generate_and_update_project_name(project_id=project_id, prompt=prompt))

        # 3. Create Sandbox
        sandbox_pass = str(uuid.uuid4())
        sandbox = create_sandbox(sandbox_pass, project_id)
        sandbox_id = sandbox.id
        logger.info(f"Created new sandbox {sandbox_id} for project {project_id}")

        # Get preview links
        vnc_link = sandbox.get_preview_link(6080)
        website_link = sandbox.get_preview_link(8080)
        vnc_url = vnc_link.url if hasattr(vnc_link, 'url') else str(vnc_link).split("url='")[1].split("'")[0]
        website_url = website_link.url if hasattr(website_link, 'url') else str(website_link).split("url='")[1].split("'")[0]
        token = None
        if hasattr(vnc_link, 'token'):
            token = vnc_link.token
        elif "token='" in str(vnc_link):
            token = str(vnc_link).split("token='")[1].split("'")[0]

        # Update project with sandbox info
        update_result = await client.table('projects').update({
            'sandbox': {
                'id': sandbox_id, 'pass': sandbox_pass, 'vnc_preview': vnc_url,
                'sandbox_url': website_url, 'token': token
            }
        }).eq('project_id', project_id).execute()

        if not update_result.data:
            logger.error(f"Failed to update project {project_id} with new sandbox {sandbox_id}")
            raise Exception("Database update failed")

        # 4. Upload Files to Sandbox (if any)
        message_content = prompt
        if files:
            successful_uploads = []
            failed_uploads = []
            for file in files:
                if file.filename:
                    try:
                        safe_filename = file.filename.replace('/', '_').replace('\\', '_')
                        target_path = f"/workspace/{safe_filename}"
                        logger.info(f"Attempting to upload {safe_filename} to {target_path} in sandbox {sandbox_id}")
                        content = await file.read()
                        upload_successful = False
                        try:
                            if hasattr(sandbox, 'fs') and hasattr(sandbox.fs, 'upload_file'):
                                import inspect
                                if inspect.iscoroutinefunction(sandbox.fs.upload_file):
                                    await sandbox.fs.upload_file(target_path, content)
                                else:
                                    sandbox.fs.upload_file(target_path, content)
                                logger.debug(f"Called sandbox.fs.upload_file for {target_path}")
                                upload_successful = True
                            else:
                                raise NotImplementedError("Suitable upload method not found on sandbox object.")
                        except Exception as upload_error:
                            logger.error(f"Error during sandbox upload call for {safe_filename}: {str(upload_error)}", exc_info=True)

                        if upload_successful:
                            try:
                                await asyncio.sleep(0.2)
                                parent_dir = os.path.dirname(target_path)
                                files_in_dir = sandbox.fs.list_files(parent_dir)
                                file_names_in_dir = [f.name for f in files_in_dir]
                                if safe_filename in file_names_in_dir:
                                    successful_uploads.append(target_path)
                                    logger.info(f"Successfully uploaded and verified file {safe_filename} to sandbox path {target_path}")
                                else:
                                    logger.error(f"Verification failed for {safe_filename}: File not found in {parent_dir} after upload attempt.")
                                    failed_uploads.append(safe_filename)
                            except Exception as verify_error:
                                logger.error(f"Error verifying file {safe_filename} after upload: {str(verify_error)}", exc_info=True)
                                failed_uploads.append(safe_filename)
                        else:
                            failed_uploads.append(safe_filename)
                    except Exception as file_error:
                        logger.error(f"Error processing file {file.filename}: {str(file_error)}", exc_info=True)
                        failed_uploads.append(file.filename)
                    finally:
                        await file.close()

            if successful_uploads:
                message_content += "\n\n" if message_content else ""
                for file_path in successful_uploads: message_content += f"[Uploaded File: {file_path}]\n"
            if failed_uploads:
                message_content += "\n\nThe following files failed to upload:\n"
                for failed_file in failed_uploads: message_content += f"- {failed_file}\n"

        # 5. Add initial user message to thread
        message_id = str(uuid.uuid4())
        message_payload = {"role": "user", "content": message_content}
        await client.table('messages').insert({
            "message_id": message_id, "thread_id": thread_id, "type": "user",
            "is_llm_message": True, "content": json.dumps(message_payload),
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()

        # 6. Start Agent Run
        agent_run = await client.table('agent_runs').insert({
            "thread_id": thread_id, "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        agent_run_id = agent_run.data[0]['id']
        logger.info(f"Created new agent run: {agent_run_id}")

        # Register run in Redis
        instance_key = f"active_run:{instance_id}:{agent_run_id}"
        try:
            await redis.set(instance_key, "running", ex=redis.REDIS_KEY_TTL)
        except Exception as e:
            logger.warning(f"Failed to register agent run in Redis ({instance_key}): {str(e)}")

        # Run agent in background
        run_agent_background.send(
            agent_run_id=agent_run_id, thread_id=thread_id, instance_id=instance_id,
            project_id=project_id,
            model_name=model_name,  # Already resolved above
            enable_thinking=enable_thinking, reasoning_effort=reasoning_effort,
            stream=stream, enable_context_manager=enable_context_manager
        )

        return {"thread_id": thread_id, "agent_run_id": agent_run_id}

    except Exception as e:
        logger.error(f"Error in agent initiation: {str(e)}\n{traceback.format_exc()}")
        # TODO: Clean up created project/thread if initiation fails mid-way
        raise HTTPException(status_code=500, detail=f"Failed to initiate agent session: {str(e)}")

@router.post("/agent-run/{agent_run_id}/stop")
async def stop_agent(agent_run_id: str, user_id: str = Depends(get_current_user_id_from_jwt)):
    """Stop a running agent."""
    logger.info(f"Received request to stop agent run: {agent_run_id}")
    client = await db.client
    await get_agent_run_with_access_check(client, agent_run_id, user_id)
    await stop_agent_run(agent_run_id)
    return {"status": "stopped"}


@router.get("/thread/{thread_id}/agent-runs")
async def get_agent_runs(thread_id: str, user_id: str = Depends(get_current_user_id_from_jwt)):
    """Get all agent runs for a thread."""
    logger.info(f"Fetching agent runs for thread: {thread_id}")
    client = await db.client
    await verify_thread_access(client, thread_id, user_id)
    agent_runs = await client.table('agent_runs').select('*').eq("thread_id", thread_id).order('created_at', desc=True).execute()
    logger.debug(f"Found {len(agent_runs.data)} agent runs for thread: {thread_id}")
    return {"agent_runs": agent_runs.data}


@router.get("/agent-run/{agent_run_id}")
async def get_agent_run(agent_run_id: str, user_id: str = Depends(get_current_user_id_from_jwt)):
    """Get agent run status and responses."""
    logger.info(f"Fetching agent run details: {agent_run_id}")
    client = await db.client
    agent_run_data = await get_agent_run_with_access_check(client, agent_run_id, user_id)
    # Note: Responses are not included here by default, they are in the stream or DB
    return {
        "id": agent_run_data['id'],
        "threadId": agent_run_data['thread_id'],
        "status": agent_run_data['status'],
        "startedAt": agent_run_data['started_at'],
        "completedAt": agent_run_data['completed_at'],
        "error": agent_run_data['error']
    }
