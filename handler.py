import runpod
from runpod.serverless.utils import rp_upload
import json
import urllib.request
import urllib.parse
import time
import os
import requests
import base64
from io import BytesIO
import websocket
import uuid
import tempfile
import socket

# Optional: OpenAI moderation
import openai

# Time to wait between API check attempts in milliseconds
COMFY_API_AVAILABLE_INTERVAL_MS = 50
# Maximum number of API check attempts
COMFY_API_AVAILABLE_MAX_RETRIES = 500
# Websocket Reconnection Retries (when connection drops during recv)
WEBSOCKET_RECONNECT_ATTEMPTS = 2
WEBSOCKET_RECONNECT_DELAY_S = 3
# Host where ComfyUI is running
COMFY_HOST = "127.0.0.1:8188"
# Enforce a clean state after each job is done
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"

# OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Define categories to block
MODERATION_CATEGORY_BLACKLIST = set(
    os.getenv("MODERATION_CATEGORIES", os.environ.get("DEFAULT_MODERATION_CATEGORIES", "hate,harassment,self-harm,sexual,violence,underage")).split(",")
)

def moderate_prompt(prompt_text):
    try:
        response = openai.Moderation.create(input=prompt_text)
        result = response["results"][0]
        if result["flagged"]:
            blocked = [k for k, v in result["categories"].items() if v and k in MODERATION_CATEGORY_BLACKLIST]
            if blocked:
                return True, blocked
        return False, []
    except Exception as e:
        print(f"worker-comfyui - OpenAI Moderation API error: {e}")
        return False, []

def moderate_image(base64_image):
    try:
        response = openai.Moderation.create(input=base64_image)
        result = response["results"][0]
        if result["flagged"]:
            blocked = [k for k, v in result["categories"].items() if v and k in MODERATION_CATEGORY_BLACKLIST]
            if blocked:
                return True, blocked
        return False, []
    except Exception as e:
        print(f"worker-comfyui - OpenAI Image Moderation API error: {e}")
        return False, []

def handler(job):
    if not OPENAI_API_KEY:
        print("worker-comfyui - Skipping moderation: OPENAI_API_KEY not set.")
        moderation_enabled = False
    else:
        moderation_enabled = True
    job_input = job["input"]
    job_id = job["id"]

    validated_data, error_message = validate_input(job_input)
    if error_message:
        return {"error": error_message}

    workflow = validated_data["workflow"]
    input_images = validated_data.get("images")

    # Moderate prompt if text exists in workflow (assume common key 'text')
    if moderation_enabled:
        for node in workflow.values():
        if isinstance(node, dict):
            text = node.get("inputs", {}).get("text")
            if isinstance(text, str):
                flagged, categories = moderate_prompt(text)
                if flagged:
                    return {
                    "error": "Prompt flagged by moderation system",
                    "categories": categories,
                    "statusCode": 403
                }

    # Moderate input images (base64 string)
    if moderation_enabled and input_images:
        for image in input_images:
            base64_data = image.get("image", "")
            if "," in base64_data:
                base64_data = base64_data.split(",", 1)[1]
            flagged, categories = moderate_image(base64_data)
            if flagged:
                return {
                "error": f"Input image '{image.get('name', 'unnamed')}' flagged by moderation system",
                "categories": categories,
                "statusCode": 403
            }

    if not check_server(
        f"http://{COMFY_HOST}/",
        COMFY_API_AVAILABLE_MAX_RETRIES,
        COMFY_API_AVAILABLE_INTERVAL_MS,
    ):
        return {
            "error": f"ComfyUI server ({COMFY_HOST}) not reachable after multiple retries."
        }

    if input_images:
        upload_result = upload_images(input_images)
        if upload_result["status"] == "error":
            return {
                "error": "Failed to upload one or more input images",
                "details": upload_result["details"],
            }

        ws = None
    client_id = str(uuid.uuid4())
    prompt_id = None
    output_data = []
    errors = []

    try:
        ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"
        print(f"worker-comfyui - Connecting to websocket: {ws_url}")
        ws = websocket.WebSocket()
        ws.connect(ws_url, timeout=10)
        print(f"worker-comfyui - Websocket connected")

        try:
            queued_workflow = queue_workflow(workflow, client_id)
            prompt_id = queued_workflow.get("prompt_id")
            if not prompt_id:
                raise ValueError(f"Missing 'prompt_id' in queue response: {queued_workflow}")
            print(f"worker-comfyui - Queued workflow with ID: {prompt_id}")
        except requests.RequestException as e:
            raise ValueError(f"Error queuing workflow: {e}")

        print(f"worker-comfyui - Waiting for workflow execution ({prompt_id})...")
        execution_done = False
        while True:
            try:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message.get("type") == "status":
                        continue
                    elif message.get("type") == "executing":
                        data = message.get("data", {})
                        if data.get("node") is None and data.get("prompt_id") == prompt_id:
                            print(f"worker-comfyui - Execution finished for prompt {prompt_id}")
                            execution_done = True
                            break
                    elif message.get("type") == "execution_error":
                        data = message.get("data", {})
                        if data.get("prompt_id") == prompt_id:
                            error_details = f"Node Type: {data.get('node_type')}, Node ID: {data.get('node_id')}, Message: {data.get('exception_message')}"
                            print(f"worker-comfyui - Execution error received: {error_details}")
                            errors.append(f"Workflow execution error: {error_details}")
                            break
                else:
                    continue
            except websocket.WebSocketTimeoutException:
                continue
            except websocket.WebSocketConnectionClosedException as closed_err:
                try:
                    ws = _attempt_websocket_reconnect(ws_url, WEBSOCKET_RECONNECT_ATTEMPTS, WEBSOCKET_RECONNECT_DELAY_S, closed_err)
                    continue
                except websocket.WebSocketConnectionClosedException as reconn_failed_err:
                    raise reconn_failed_err

        if not execution_done and not errors:
            raise ValueError("Workflow monitoring loop exited without confirmation of completion or error.")

        print(f"worker-comfyui - Fetching history for prompt {prompt_id}...")
        history = get_history(prompt_id)
        if prompt_id not in history:
            error_msg = f"Prompt ID {prompt_id} not found in history after execution."
            print(f"worker-comfyui - {error_msg}")
            if not errors:
                return {"error": error_msg}
            else:
                errors.append(error_msg)
                return {"error": "Job processing failed, prompt ID not found in history.", "details": errors}

        prompt_history = history.get(prompt_id, {})
        outputs = prompt_history.get("outputs", {})
        if not outputs:
            warning_msg = f"No outputs found in history for prompt {prompt_id}."
            print(f"worker-comfyui - {warning_msg}")
            if not errors:
                errors.append(warning_msg)

        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for image_info in node_output["images"]:
                    filename = image_info.get("filename")
                    subfolder = image_info.get("subfolder", "")
                    img_type = image_info.get("type")
                    if img_type == "temp":
                        continue
                    if not filename:
                        errors.append(f"Missing filename in node {node_id} output.")
                        continue
                    image_bytes = get_image_data(filename, subfolder, img_type)
                    if image_bytes:
                        file_extension = os.path.splitext(filename)[1] or ".png"
                        if os.environ.get("BUCKET_ENDPOINT_URL"):
                            try:
                                with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                                    temp_file.write(image_bytes)
                                    temp_file_path = temp_file.name
                                s3_url = rp_upload.upload_image(job_id, temp_file_path)
                                os.remove(temp_file_path)
                                output_data.append({"filename": filename, "type": "s3_url", "data": s3_url})
                            except Exception as e:
                                errors.append(f"S3 upload error for {filename}: {e}")
                                if os.path.exists(temp_file_path):
                                    try:
                                        os.remove(temp_file_path)
                                    except OSError:
                                        pass
                        else:
                            try:
                                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                                output_data.append({"filename": filename, "type": "base64", "data": base64_image})
                            except Exception as e:
                                errors.append(f"Base64 encode error for {filename}: {e}")
                    else:
                        errors.append(f"Failed to fetch image data for {filename}.")

    except websocket.WebSocketException as e:
        return {"error": f"WebSocket communication error: {e}"}
    except requests.RequestException as e:
        return {"error": f"HTTP communication error with ComfyUI: {e}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}
    finally:
        if ws and ws.connected:
            ws.close()

    final_result = {}
    if output_data:
        final_result["images"] = output_data
    if errors:
        final_result["errors"] = errors
    if not output_data and errors:
        return {"error": "Job processing failed", "details": errors}
    elif not output_data and not errors:
        final_result["status"] = "success_no_images"
        final_result["images"] = []

    return final_result

if __name__ == "__main__":
    print("worker-comfyui - Starting handler...")
    runpod.serverless.start({"handler": handler})
