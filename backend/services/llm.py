"""
LLM API interface for making calls to various language models.

This module provides a unified interface for making API calls to different LLM providers
(OpenAI, Anthropic, Groq, etc.) using LiteLLM. It includes support for:
- Streaming responses
- Tool calls and function calling
- Retry logic with exponential backoff
- Model-specific configurations
- Comprehensive error handling and logging
"""

from typing import Union, Dict, Any, Optional, AsyncGenerator, List
import os
import json
import asyncio
import hashlib
from openai import OpenAIError
import litellm
from litellm.utils import ModelResponse  # For reconstructing the response object
from utils.logger import logger
from utils.config import config
from . import redis as redis_service # Import redis service from same directory

# litellm.set_verbose=True
litellm.modify_params=True

# Constants
MAX_RETRIES = 2
RATE_LIMIT_DELAY = 30
RETRY_DELAY = 0.1

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class LLMRetryError(LLMError):
    """Exception raised when retries are exhausted."""
    pass

def setup_api_keys() -> None:
    """Set up API keys from environment variables."""
    providers = ['OPENAI', 'ANTHROPIC', 'GROQ', 'OPENROUTER']
    for provider in providers:
        key = getattr(config, f'{provider}_API_KEY')
        if key:
            logger.debug(f"API key set for provider: {provider}")
        else:
            logger.warning(f"No API key found for provider: {provider}")

    # Set up OpenRouter API base if not already set
    if config.OPENROUTER_API_KEY and config.OPENROUTER_API_BASE:
        os.environ['OPENROUTER_API_BASE'] = config.OPENROUTER_API_BASE
        logger.debug(f"Set OPENROUTER_API_BASE to {config.OPENROUTER_API_BASE}")

    # Set up AWS Bedrock credentials
    aws_access_key = config.AWS_ACCESS_KEY_ID
    aws_secret_key = config.AWS_SECRET_ACCESS_KEY
    aws_region = config.AWS_REGION_NAME

    if aws_access_key and aws_secret_key and aws_region:
        logger.debug(f"AWS credentials set for Bedrock in region: {aws_region}")
        # Configure LiteLLM to use AWS credentials
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
        os.environ['AWS_REGION_NAME'] = aws_region
    else:
        logger.warning(f"Missing AWS credentials for Bedrock integration - access_key: {bool(aws_access_key)}, secret_key: {bool(aws_secret_key)}, region: {aws_region}")

async def handle_error(error: Exception, attempt: int, max_attempts: int) -> None:
    """Handle API errors with appropriate delays and logging."""
    delay = RATE_LIMIT_DELAY if isinstance(error, litellm.exceptions.RateLimitError) else RETRY_DELAY
    logger.warning(f"Error on attempt {attempt + 1}/{max_attempts}: {str(error)}")
    logger.debug(f"Waiting {delay} seconds before retry...")
    await asyncio.sleep(delay)

def prepare_params(
    messages: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    response_format: Optional[Any] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    stream: bool = False,
    top_p: Optional[float] = None,
    model_id: Optional[str] = None,
    enable_thinking: Optional[bool] = False,
    reasoning_effort: Optional[str] = 'low'
) -> Dict[str, Any]:
    """Prepare parameters for the API call."""
    params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "response_format": response_format,
        "top_p": top_p,
        "stream": stream,
    }

    if api_key:
        params["api_key"] = api_key
    if api_base:
        params["api_base"] = api_base
    if model_id:
        params["model_id"] = model_id

    # Handle token limits
    if max_tokens is not None:
        # For Claude 3.7 in Bedrock, do not set max_tokens or max_tokens_to_sample
        # as it causes errors with inference profiles
        if model_name.startswith("bedrock/") and "claude-3-7" in model_name:
            logger.debug(f"Skipping max_tokens for Claude 3.7 model: {model_name}")
            # Do not add any max_tokens parameter for Claude 3.7
        else:
            param_name = "max_completion_tokens" if 'o1' in model_name else "max_tokens"
            params[param_name] = max_tokens

    # Add tools if provided
    if tools:
        params.update({
            "tools": tools,
            "tool_choice": tool_choice
        })
        logger.debug(f"Added {len(tools)} tools to API parameters")

    # # Add Claude-specific headers
    if "claude" in model_name.lower() or "anthropic" in model_name.lower():
        params["extra_headers"] = {
            # "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"
            "anthropic-beta": "output-128k-2025-02-19"
        }
        logger.debug("Added Claude-specific headers")

    # Add OpenRouter-specific parameters
    if model_name.startswith("openrouter/"):
        logger.debug(f"Preparing OpenRouter parameters for model: {model_name}")

        # Add optional site URL and app name from config
        site_url = config.OR_SITE_URL
        app_name = config.OR_APP_NAME
        if site_url or app_name:
            extra_headers = params.get("extra_headers", {})
            if site_url:
                extra_headers["HTTP-Referer"] = site_url
            if app_name:
                extra_headers["X-Title"] = app_name
            params["extra_headers"] = extra_headers
            logger.debug(f"Added OpenRouter site URL and app name to headers")

    # Add Bedrock-specific parameters
    if model_name.startswith("bedrock/"):
        logger.debug(f"Preparing AWS Bedrock parameters for model: {model_name}")

        if not model_id and "anthropic.claude-3-7-sonnet" in model_name:
            params["model_id"] = "arn:aws:bedrock:us-west-2:935064898258:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            logger.debug(f"Auto-set model_id for Claude 3.7 Sonnet: {params['model_id']}")

    # Apply Anthropic prompt caching (minimal implementation)
    # Check model name *after* potential modifications (like adding bedrock/ prefix)
    effective_model_name = params.get("model", model_name) # Use model from params if set, else original
    if "claude" in effective_model_name.lower() or "anthropic" in effective_model_name.lower():
        messages = params["messages"] # Direct reference, modification affects params

        # Ensure messages is a list
        if not isinstance(messages, list):
            return params # Return early if messages format is unexpected

        # 1. Process the first message if it's a system prompt with string content
        if messages and messages[0].get("role") == "system":
            content = messages[0].get("content")
            if isinstance(content, str):
                # Wrap the string content in the required list structure
                messages[0]["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
            elif isinstance(content, list):
                 # If content is already a list, check if the first text block needs cache_control
                 for item in content:
                     if isinstance(item, dict) and item.get("type") == "text":
                         if "cache_control" not in item:
                             item["cache_control"] = {"type": "ephemeral"}
                             break # Apply to the first text block only for system prompt

        # 2. Find and process relevant user and assistant messages
        last_user_idx = -1
        second_last_user_idx = -1
        last_assistant_idx = -1

        for i in range(len(messages) - 1, -1, -1):
            role = messages[i].get("role")
            if role == "user":
                if last_user_idx == -1:
                    last_user_idx = i
                elif second_last_user_idx == -1:
                    second_last_user_idx = i
            elif role == "assistant":
                if last_assistant_idx == -1:
                    last_assistant_idx = i

            # Stop searching if we've found all needed messages
            if last_user_idx != -1 and second_last_user_idx != -1 and last_assistant_idx != -1:
                 break

        # Helper function to apply cache control
        def apply_cache_control(message_idx: int, message_role: str):
            if message_idx == -1:
                return

            message = messages[message_idx]
            content = message.get("content")

            if isinstance(content, str):
                message["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        if "cache_control" not in item:
                           item["cache_control"] = {"type": "ephemeral"}

        # Apply cache control to the identified messages
        apply_cache_control(last_user_idx, "last user")
        apply_cache_control(second_last_user_idx, "second last user")
        apply_cache_control(last_assistant_idx, "last assistant")

    # Add reasoning_effort for Anthropic models if enabled
    use_thinking = enable_thinking if enable_thinking is not None else False
    is_anthropic = "anthropic" in effective_model_name.lower() or "claude" in effective_model_name.lower()

    if is_anthropic and use_thinking:
        effort_level = reasoning_effort if reasoning_effort else 'low'
        params["reasoning_effort"] = effort_level
        params["temperature"] = 1.0 # Required by Anthropic when reasoning_effort is used
        logger.info(f"Anthropic thinking enabled with reasoning_effort='{effort_level}'")

    return params

async def make_llm_api_call(
    messages: List[Dict[str, Any]],
    model_name: str,
    response_format: Optional[Any] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    stream: bool = False,
    top_p: Optional[float] = None,
    model_id: Optional[str] = None,
    enable_thinking: Optional[bool] = False,
    reasoning_effort: Optional[str] = 'low',
    # enable_caching: bool = True # Future parameter to explicitly control caching per call
) -> Union[ModelResponse, AsyncGenerator]: # Return type updated to ModelResponse
    """
    Make an API call to a language model using LiteLLM.

    Args:
        messages: List of message dictionaries for the conversation
        model_name: Name of the model to use (e.g., "gpt-4", "claude-3", "openrouter/openai/gpt-4", "bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
        response_format: Desired format for the response
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in the response
        tools: List of tool definitions for function calling
        tool_choice: How to select tools ("auto" or "none")
        api_key: Override default API key
        api_base: Override default API base URL
        stream: Whether to stream the response
        top_p: Top-p sampling parameter
        model_id: Optional ARN for Bedrock inference profiles
        enable_thinking: Whether to enable thinking
        reasoning_effort: Level of reasoning effort

    Returns:
        Union[Dict[str, Any], AsyncGenerator]: API response or stream

    Raises:
        LLMRetryError: If API call fails after retries
        LLMError: For other API-related errors
    """
    # debug <timestamp>.json messages
    logger.info(f"Making LLM API call to model: {model_name} (Thinking: {enable_thinking}, Effort: {reasoning_effort})")
    
    # If streaming, bypass caching
    if stream:
        logger.info(f"📡 API Call (Streaming): Using model {model_name}")
        params = prepare_params(
            messages=messages,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            api_key=api_key,
            api_base=api_base,
            stream=stream,
            top_p=top_p,
            model_id=model_id,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort
        )
        # Proceed directly to API call for streaming, no caching
        # (Error handling and retry logic for streaming calls remain as is)
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(f"Attempt {attempt + 1}/{MAX_RETRIES} for streaming call")
                response = await litellm.acompletion(**params)
                logger.debug(f"Successfully received API stream from {model_name}")
                return response
            except (litellm.exceptions.RateLimitError, OpenAIError, json.JSONDecodeError) as e:
                last_error = e
                await handle_error(e, attempt, MAX_RETRIES)
            except Exception as e:
                logger.error(f"Unexpected error during streaming API call: {str(e)}", exc_info=True)
                raise LLMError(f"API call failed: {str(e)}")
        
        error_msg = f"Failed to make streaming API call after {MAX_RETRIES} attempts"
        if last_error:
            error_msg += f". Last error: {str(last_error)}"
        logger.error(error_msg, exc_info=True)
        raise LLMRetryError(error_msg)

    # --- Caching Logic for Non-Streaming Calls ---
    logger.info(f"📡 API Call (Non-Streaming): Using model {model_name}")

    # Prepare parameters once, as they might influence the cache key
    # (e.g., if prepare_params modifies model_name, temperature for reasoning_effort, etc.)
    params_for_api = prepare_params( # Renamed to avoid confusion with function-level params
        messages=messages,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        api_key=api_key,
        api_base=api_base,
        stream=stream, # Will be False here
        top_p=top_p,
        model_id=model_id,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort
    )

    redis_client = None
    cache_key = None

    try:
        redis_client = await redis_service.get_client()
    except Exception as e:
        logger.error(f"Failed to get Redis client: {e}. Caching will be skipped.", exc_info=True)

    if redis_client:
        try:
            # Use parameters that define the request, including those potentially modified by prepare_params
            # For example, 'model' from params_for_api, 'temperature' if modified by reasoning_effort, etc.
            key_params = {
                "model_name": params_for_api.get("model", model_name), # Use model from prepared_params
                "messages": messages, # Original messages
                "temperature": params_for_api.get("temperature", temperature), # Temperature from prepared_params
                "max_tokens": params_for_api.get("max_tokens") or params_for_api.get("max_completion_tokens"),
                "response_format": response_format,
                "tools": tools,
                "tool_choice": tool_choice,
                "top_p": top_p,
                # "reasoning_effort" is implicitly handled if it changes temperature or other params in prepare_params
                # However, if reasoning_effort itself should be a key component (even if it doesn't change other params), add it.
                # For now, assuming its effects are captured by changes in other params like temperature.
                "reasoning_effort": params_for_api.get("reasoning_effort"), # Explicitly add reasoning_effort
            }
            cache_key = _generate_cache_key(**key_params)
            
            cached_response_json = await redis_service.get(cache_key)
            if cached_response_json:
                logger.info(f"LLM response retrieved from cache for key: {cache_key}")
                try:
                    cached_response_dict = json.loads(cached_response_json)
                    # Ensure 'id' is present if ModelResponse expects it and it's not always in cached_response_dict
                    # This might require inspecting what litellm.ModelResponse strictly needs for instantiation
                    if 'id' not in cached_response_dict and hasattr(ModelResponse, "id"):
                        # If 'id' is a required field for ModelResponse and not in cache, 
                        # it might be better to treat as a cache miss or add a placeholder.
                        # For now, we assume 'id' might not be strictly necessary or is usually present.
                        # Alternatively, if 'id' should be unique per request, it shouldn't come from cache anyway,
                        # or the cache structure needs to account for it.
                        # LiteLLM ModelResponse objects *do* have an 'id' field.
                        # If it's not critical for the functioning of the app, it might be okay.
                        # Otherwise, a new 'id' could be generated, or this indicates an issue
                        # with what was originally cached.
                        # For safety, if 'id' is missing and important, one might log a warning
                        # or even invalidate this specific cache entry.
                        # The original code did ModelResponse(**cached_response_dict), let's stick to that
                        # and assume 'id' and other fields are handled correctly by litellm's Pydantic model.
                        pass

                    response = ModelResponse(**cached_response_dict)
                    return response
                except json.JSONDecodeError:
                    logger.warning(f"Error decoding cached JSON for key {cache_key}. Fetching fresh response.", exc_info=True)
                except TypeError as te:
                    logger.warning(f"Error reconstructing ModelResponse from cached data for key {cache_key}: {te}. Fetching fresh response.", exc_info=True)
                # If deserialization fails, proceed to API call (cache miss)
        except redis_service.redis.exceptions.RedisError as re: # More specific Redis error
            logger.error(f"Redis error during cache retrieval for key {cache_key}: {re}. Bypassing cache.", exc_info=True)
        except Exception as e:
            logger.warning(f"Error during cache check for key {cache_key}: {e}. Bypassing cache.", exc_info=True)
            cache_key = None # Invalidate cache_key if generation or initial check fails

    if cache_key:
        logger.info(f"LLM response not found in cache for key: {cache_key}. Proceeding with API call.")
    else:
        logger.info("Cache bypassed (Redis client unavailable or error in key generation). Proceeding with API call.")

    # API call parameters are already prepared in params_for_api
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"Attempt {attempt + 1}/{MAX_RETRIES}")
            # Use the already prepared params_for_api
            api_response = await litellm.acompletion(**params_for_api)
            logger.debug(f"Successfully received API response from {params_for_api.get('model', model_name)}")
            # logger.debug(f"Response: {api_response}") # Can be very verbose

            # Store in cache if cache_key was generated, redis_client is available, and response is valid
            if cache_key and redis_client and api_response:
                try:
                    response_to_cache_json = None
                    if hasattr(api_response, 'model_dump_json'):
                        response_to_cache_json = api_response.model_dump_json()
                    elif hasattr(api_response, 'dict'): 
                        response_to_cache_json = json.dumps(api_response.dict())
                    else:
                        # If neither, this will likely fail
                        logger.warning("Cannot serialize API response for caching: no model_dump_json or dict method.")
                    
                    if response_to_cache_json:
                        await redis_service.set(cache_key, response_to_cache_json, ex=redis_service.REDIS_KEY_TTL)
                        logger.info(f"LLM response stored in cache for key: {cache_key}")
                except redis_service.redis.exceptions.RedisError as re: # More specific Redis error
                    logger.error(f"Redis error during cache storage for key {cache_key}: {re}", exc_info=True)
                except Exception as e: # Catch other potential errors during serialization/storage
                    logger.error(f"Error during cache storage preparation for key {cache_key}: {e}", exc_info=True)
            
            return api_response

        except (litellm.exceptions.RateLimitError, OpenAIError, json.JSONDecodeError, litellm.exceptions.APIConnectionError) as e: # Added APIConnectionError
            last_error = e
            await handle_error(e, attempt, MAX_RETRIES)

        except Exception as e:
            logger.error(f"Unexpected error during API call: {str(e)}", exc_info=True)
            raise LLMError(f"API call failed: {str(e)}")

    error_msg = f"Failed to make API call after {MAX_RETRIES} attempts"
    if last_error:
        error_msg += f". Last error: {str(last_error)}"
    logger.error(error_msg, exc_info=True)
    raise LLMRetryError(error_msg)

# Helper function to generate cache key
def _generate_cache_key(
    model_name: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: Optional[int],
    response_format: Optional[Any],
    tools: Optional[List[Dict[str, Any]]],
    tool_choice: Optional[str], # tool_choice can be None if tools are not used
    top_p: Optional[float],
    reasoning_effort: Optional[str] # Added based on requirements
) -> str:
    """Generates a SHA256 hash key for caching LLM responses."""
    try:
        # Deterministic serialization for complex objects
        serialized_messages = json.dumps(messages, sort_keys=True)
        serialized_tools = json.dumps(tools, sort_keys=True) if tools else "None"
        serialized_response_format = json.dumps(response_format, sort_keys=True) if response_format else "None"

        key_string_parts = [
            str(model_name),
            serialized_messages,
            str(temperature),
            str(max_tokens),
            serialized_response_format,
            serialized_tools,
            str(tool_choice),
            str(top_p),
            str(reasoning_effort),
        ]
        
        # Filter out None parts if they were converted to "None" strings unnecessarily,
        # or ensure consistent string representation for all parts.
        # The current approach of str(None) becoming "None" is fine for consistency.
        
        key_string = ":".join(key_string_parts)
        
        return f"llm_cache:{hashlib.sha256(key_string.encode('utf-8')).hexdigest()}"
    except Exception as e:
        logger.error(f"Error generating cache key: {e}", exc_info=True)
        # Fallback to a less specific key or indicate error if necessary,
        # but for now, re-raise or handle by returning a key that won't be found.
        # For simplicity, let this error propagate to be caught by the caller's cache bypass logic.
        raise


# Initialize API keys on module import
setup_api_keys()

# Test code for OpenRouter integration
async def test_openrouter():
    """Test the OpenRouter integration with a simple query."""
    test_messages = [
        {"role": "user", "content": "Hello, can you give me a quick test response?"}
    ]

    try:
        # Test with standard OpenRouter model
        print("\n--- Testing standard OpenRouter model ---")
        response = await make_llm_api_call(
            model_name="openrouter/openai/gpt-4o-mini",
            messages=test_messages,
            temperature=0.7,
            max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")

        # Test with deepseek model
        print("\n--- Testing deepseek model ---")
        response = await make_llm_api_call(
            model_name="openrouter/deepseek/deepseek-r1-distill-llama-70b",
            messages=test_messages,
            temperature=0.7,
            max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")

        # Test with Mistral model
        print("\n--- Testing Mistral model ---")
        response = await make_llm_api_call(
            model_name="openrouter/mistralai/mixtral-8x7b-instruct",
            messages=test_messages,
            temperature=0.7,
            max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")

        return True
    except Exception as e:
        print(f"Error testing OpenRouter: {str(e)}")
        return False

async def test_bedrock():
    """Test the AWS Bedrock integration with a simple query."""
    test_messages = [
        {"role": "user", "content": "Hello, can you give me a quick test response?"}
    ]

    try:
        response = await make_llm_api_call(
            model_name="bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0",
            model_id="arn:aws:bedrock:us-west-2:935064898258:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            messages=test_messages,
            temperature=0.7,
            # Claude 3.7 has issues with max_tokens, so omit it
            # max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")

        return True
    except Exception as e:
        print(f"Error testing Bedrock: {str(e)}")
        return False

if __name__ == "__main__":
    import asyncio

    test_success = asyncio.run(test_bedrock())

    if test_success:
        print("\n✅ integration test completed successfully!")
    else:
        print("\n❌ Bedrock integration test failed!")
