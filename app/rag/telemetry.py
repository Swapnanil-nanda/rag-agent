import uuid
import time
import logging

logger = logging.getLogger("context_engine.telemetry")

class TelemetryTracer:
    def __init__(self):
        self.trace_id = uuid.uuid4().hex
        self.start_time = time.time()

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str = "openrouter") -> float:
        # Approximate cost calculation ($0.0001 / 1k tokens)
        total_tokens = prompt_tokens + completion_tokens
        return round((total_tokens / 1000.0) * 0.0001, 6)

    def log_query_span(self, question: str, response_length: int, chunks_count: int, model: str = "nvidia/nemotron"):
        latency_ms = round((time.time() - self.start_time) * 1000, 2)
        est_prompt_tokens = len(question.split()) * 4
        est_completion_tokens = max(1, response_length // 4)
        est_cost = self.calculate_cost(est_prompt_tokens, est_completion_tokens, model)
        
        telemetry_data = {
            "trace_id": self.trace_id,
            "latency_ms": latency_ms,
            "prompt_tokens": est_prompt_tokens,
            "completion_tokens": est_completion_tokens,
            "estimated_cost_usd": est_cost,
            "chunks_retrieved": chunks_count,
            "model": model
        }
        logger.info(f"[TELEMETRY_SPAN] {telemetry_data}")
        return telemetry_data
