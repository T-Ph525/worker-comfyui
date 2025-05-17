"""handler.py — Concurrent ComfyUI Worker + /admin management API
================================================================
This single file exposes **two HTTP endpoints** when deployed on RunPod
Serverless:

* **/run**   → `handler` (generates images, moderates inputs, supports
  concurrency scaling)
* **/admin** → `admin_handler` (manages models & custom nodes via *comfy‑cli*)

Security
--------
Every `/admin` request **must** include a JSON field `api_key` matching the
`ADMIN_API_KEY` environment variable (set in RunPod’s *Secure ENV* tab).  If the
key is missing or wrong the call returns `{"error": "unauthorized"}`.

Supported admin actions
----------------------
`job["input"]` for the admin endpoint must include an `action` string:

| action           | required fields                | description                                               |
|------------------|--------------------------------|-----------------------------------------------------------|
| `download_model` | `url`, `relative_path`         | Download a checkpoint with `comfy model download`         |
| `install_node`   | `repo`                         | Install a custom node repo via `comfy node install`       |
| `list_models`    | _(none)_                       | JSON list from `comfy model list --json`                  |
| `list_nodes`     | _(none)_                       | JSON list from `comfy node list --json`                   |
| `remove_model`   | `name`                         | Delete file under `/root/comfy/ComfyUI/models`            |
| `remove_node`    | `repo_name`                    | Delete folder under `/root/comfy/ComfyUI/custom_nodes`    |

Extend functionality easily by adding more `elif` blocks inside
`admin_handler`.

Implementation notes
--------------------
* Uses `subprocess.run` with `check=True` for robust error propagation.
* Emits basic **progress_update** signals for long‑running download/install
  operations.
* Adaptive concurrency applies only to `/run`; admin calls execute serially and
  rarely.
"""

import asyncio, json, os, time, uuid, base64, tempfile, socket, urllib.parse, subprocess, shutil
from io import BytesIO
from typing import Dict, Any, Tuple, List

import requests, websocket, runpod, torch
from runpod.serverless.utils import rp_upload
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# ────────────────────────────────────────────────────────────────────────────────
# Config ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ------------------------------------------------------------------------------
COMFY_HOST               = os.getenv("COMFY_HOST", "127.0.0.1:8188")
COMFY_API_AVAILABLE_MS   = int(os.getenv("COMFY_POLL_INTERVAL", 50))
COMFY_API_MAX_RETRIES    = int(os.getenv("COMFY_POLL_RETRIES", 500))
WEBSOCKET_RETRY_ATTEMPTS = int(os.getenv("WEBSOCKET_RETRY", 2))
WEBSOCKET_RETRY_DELAY_S  = int(os.getenv("WEBSOCKET_RETRY_DELAY", 3))
AGE_MODEL_PATH           = os.getenv("AGE_MODEL_PATH", "/comfy/model/age-classifier")
AGE_THRESHOLD            = float(os.getenv("AGE_THRESHOLD", 0.5))
DISABLE_AGE_CHECK        = bool(os.getenv("DISABLE_AGE_CHECK"))
ADMIN_API_KEY            = os.getenv("ADMIN_API_KEY", "")

COMFY_ROOT = "/root/comfy/ComfyUI"
MODELS_DIR = os.path.join(COMFY_ROOT, "models")
NODES_DIR  = os.path.join(COMFY_ROOT, "custom_nodes")

# ────────────────────────────────────────────────────────────────────────────────
# Age‑classifier (lazy‑loaded) ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ------------------------------------------------------------------------------
_age_processor, _age_model = None, None

def _load_age_classifier():
    global _age_processor, _age_model
    if _age_model is None and not DISABLE_AGE_CHECK:
        _age_processor = AutoImageProcessor.from_pretrained(AGE_MODEL_PATH)
        _age_model     = AutoModelForImageClassification.from_pretrained(AGE_MODEL_PATH)
        _age_model.eval()
        print(f"worker-comfyui - Age‑classifier loaded from {AGE_MODEL_PATH}")


def check_underage(blob: bytes) -> bool:
    if DISABLE_AGE_CHECK:
        return False
    _load_age_classifier()
    img = Image.open(BytesIO(blob)).convert("RGB")
    inputs = _age_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = _age_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    child_idx = _age_model.config.label2id.get("child", 0)
    return probs[child_idx].item() >= AGE_THRESHOLD

# ────────────────────────────────────────────────────────────────────────────────
# Helper: WebSocket reconnect ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ------------------------------------------------------------------------------

def _attempt_ws_reconnect(url, attempts, delay, first_err):
    last = first_err
    for _ in range(attempts):
        try:
            ws = websocket.WebSocket(); ws.connect(url, timeout=10); return ws
        except (websocket.WebSocketException, ConnectionRefusedError, socket.timeout, OSError) as e:
            last = e; time.sleep(delay)
    raise websocket.WebSocketConnectionClosedException(f"Reconnect failed — {last}")

# ────────────────────────────────────────────────────────────────────────────────
# Input validation ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ------------------------------------------------------------------------------

def validate_input(inp: Any) -> Tuple[Dict[str, Any] | None, str | None]:
    if inp is None:
        return None, "Please provide input"
    if isinstance(inp, str):
        try: inp = json.loads(inp)
        except json.JSONDecodeError: return None, "Invalid JSON"
    wf = inp.get("workflow")
    if wf is None:
        return None, "Missing 'workflow'"
    imgs = inp.get("images")
    if imgs and (not isinstance(imgs, list) or not all("name" in i and "image" in i for i in imgs)):
        return None, "'images' must be list of {name,image}"
    return {"workflow": wf, "images": imgs}, None

# ────────────────────────────────────────────────────────────────────────────────
# ComfyUI REST helpers ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ------------------------------------------------------------------------------

def check_server() -> bool:
    for _ in range(COMFY_API_MAX_RETRIES):
        try:
            if requests.get(f"http://{COMFY_HOST}/", timeout=5).status_code == 200:
                return True
        except (requests.Timeout, requests.RequestException):
            pass
        time.sleep(COMFY_API_AVAILABLE_MS / 1000)
    return False


def queue_workflow(wf, cid):
    r = requests.post(f"http://{COMFY_HOST}/prompt", data=json.dumps({"prompt": wf, "client_id": cid}).encode(), headers={"Content-Type": "application/json"}, timeout=30)
    r.raise_for_status(); return r.json()

def get_history(pid):
    r = requests.get(f"http://{COMFY_HOST}/history/{pid}", timeout=30)
    r.raise_for_status(); return r.json()

def get_image_data(fn, sub, typ):
    q = urllib.parse.urlencode({"filename": fn, "subfolder": sub, "type": typ})
    try:
        r = requests.get(f"http://{COMFY_HOST}/view?{q}", timeout=60)
        r.raise_for_status(); return r.content
    except requests.RequestException:
        return None

# ────────────────────────────────────────────────────────────────────────────────
# Image upload + moderation ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ------------------------------------------------------------------------------

def upload_images(images: List[Dict[str, str]], send_update) -> Dict[str, Any]:
    if not images:
        return {"status": "success", "details": []}
    ok, err = [], []
    for im in images:
        name, uri = im["name"], im["image"]
        data = uri.split(",", 1)[1] if "," in uri else uri
        try:
            blob = base64.b64decode(data)
        except base64.binascii.Error as e:
            err.append(f"{name}: invalid base64 ({e})"); continue
        if check_underage(blob):
            err.append(f"{name}: underage content detected"); continue
        try:
            files = {"image": (name, BytesIO(blob), "image/png"), "overwrite": (None, "true")}
            requests.post(f"http://{COMFY_HOST}/upload/image", files=files, timeout=30).raise_for_status()
            ok.append(f"Uploaded {name}")
        except requests.RequestException as e:
            err.append(f"{name}: {e}")
    return {"status": "error", "details": err} if err else {"status": "success", "details": ok}

# ────────────────────────────────────────────────────────────────────────────────
# Primary /run job handler ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ------------------------------------------------------------------------------

def handle_job_sync(job: Dict[str, Any]):
    send = lambda m: runpod.serverless.progress_update(job, m)
    val, err = validate_input(job.get("input"))
    if err:
        return {"error": err}

    wf, images = val["workflow"], val.get("images")
    send("Input validated ✔")

    if not check_server():
        return {"error": f"ComfyUI ({COMFY_HOST}) unreachable"}
    send("ComfyUI reachable")

    if images:
        up = upload_images(images, send)
        if up["status"] == "error":
            return {"error": "Image moderation/upload failed", "details": up["details"]}
    send("Images moderated & uploaded")

    cid = str(uuid.uuid4()); ws_url = f"ws://{COMFY_HOST}/ws?clientId={cid}"
    ws, pid, outs, issues = None, None, [], []
    try:
        ws = websocket.WebSocket(); ws.connect(ws_url, timeout=10)
        pid = queue_workflow(wf, cid).get("prompt_id")
        if not pid:
            return {"error": "No prompt_id"}
        send(f"Workflow queued ({pid})")
        while True:
            try:
                msg = json.loads(ws.recv())
                if msg.get("type") == "executing" and msg["data"].get("node") is None and msg["data"].get("prompt_id") == pid:
                    break
                if msg.get("type") == "execution_error":
                    issues.append(msg.get("data", {}).get("exception_message", "error")); break
            except websocket.WebSocketConnectionClosedException as ce:
                ws = _attempt_ws_reconnect(ws_url, WEBSOCKET_RETRY_ATTEMPTS, WEBSOCKET_RETRY_DELAY_S, ce)
            except json.JSONDecodeError:
                continue
        send("Workflow execution finished")

        hist = get_history(pid).get(pid, {})
        for node in hist.get("outputs", {}).values():
            for img in node.get("images", []):
                if img.get("type") == "temp": continue
                raw = get_image_data(img["filename"], img.get("subfolder", ""), img["type"])
                if not raw:
                    issues.append(f"Missing {img['filename']}"); continue
                if os.getenv("BUCKET_ENDPOINT_URL"):
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img["filename"])[1] or ".png")
                    tmp.write(raw); tmp.close()
                    try:
                        url = rp_upload.upload
