from prometheus_client import CollectorRegistry, Counter, Histogram

# Use a dedicated registry to avoid duplicate metric registration
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "api_request_total",
    "Total API requests",
    ["method", "endpoint", "http_status"],
    registry=registry,
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["method", "endpoint"],
    registry=registry,
)

LLM_CALL_LATENCY = Histogram(
    "llm_api_latency_seconds",
    "Latency of LLM API calls",
    ["model"],
    registry=registry,
)

__all__ = [
    "registry",
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "LLM_CALL_LATENCY",
]
