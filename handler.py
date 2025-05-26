import runpod
import asyncio
import random
import time

# ────────────────────────────────────────────────────────────
# Globals
# ────────────────────────────────────────────────────────────
request_rate = 0
request_history = []
MAX_CONCURRENCY = 10
MIN_CONCURRENCY = 1
RATE_THRESHOLD = 50
WINDOW_SECONDS = 60
MAX_HISTORY = 1000

# ────────────────────────────────────────────────────────────
# Request Handler
# ────────────────────────────────────────────────────────────
async def process_request(job):
    """
    Handles incoming requests with simulated async processing.

    Args:
        job (dict): Job payload from RunPod Serverless

    Returns:
        dict: Result after simulated processing
    """
    global request_history

    job_input = job.get("input", {})
    delay = job_input.get("delay", 1)

    # Track timestamp for rate metrics
    request_history.append(time.time())

    # Simulate work (e.g. model inference, DB lookup, etc.)
    await asyncio.sleep(delay)

    return {
        "status": "completed",
        "input": job_input,
        "message": f"Processed after {delay}s"
    }

# ────────────────────────────────────────────────────────────
# Concurrency Modifier
# ────────────────────────────────────────────────────────────
def adjust_concurrency(current_concurrency):
    """
    Adjust the concurrency level based on request rate.

    Args:
        current_concurrency (int): The current concurrency level

    Returns:
        int: The new concurrency level
    """
    global request_rate
    update_request_rate()

    if request_rate > RATE_THRESHOLD and current_concurrency < MAX_CONCURRENCY:
        return current_concurrency + 1
    elif request_rate <= RATE_THRESHOLD and current_concurrency > MIN_CONCURRENCY:
        return current_concurrency - 1
    return current_concurrency

# ────────────────────────────────────────────────────────────
# Rate Estimator
# ────────────────────────────────────────────────────────────
def update_request_rate():
    """
    Update the global request rate based on recent request timestamps.
    """
    global request_rate, request_history

    now = time.time()
    recent = [t for t in request_history if t > now - WINDOW_SECONDS]
    request_rate = len(recent)

    # Avoid memory bloat
    if len(request_history) > MAX_HISTORY:
        request_history[:] = recent

# ────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    runpod.serverless.start({
        "handler": process_request,
        "concurrency_modifier": adjust_concurrency
    })
