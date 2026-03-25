import json
import os
import uuid
from time import sleep

import pytest
import requests


@pytest.fixture(scope="module", autouse=True)
def workflow(request):
    with open(str(request.path).replace(".py", ".json")) as f:
        data = f.read()
    data = data.replace("$TEST_CHECKPOINT", os.environ["PC_TEST_CHECKPOINT"])
    data = data.replace("$TEST_LORA", os.environ["PC_TEST_LORA"])
    return json.loads(data)


def assert_prompt(url, p):
    timeout = 60
    r = requests.post(f"{url}/prompt", json={"prompt": p, "client_id": str(uuid.uuid4())}).json()
    prompt_id = r["prompt_id"]
    r = {"status": "pending"}
    while r["status"] in ["pending", "in_progress"]:
        sleep(1)
        assert timeout > 0
        timeout -= 1
        r = requests.get(f"{url}/api/jobs/{prompt_id}").json()
    assert r["status"] == "completed"


@pytest.fixture
def comfyui():
    return os.environ.get("PC_TEST_COMFYUI", "http://localhost:8188")


def test_workflow(workflow, comfyui):
    prompt = "DEF(blue=green)a blue dog and a cat sitting [COUPLE(0 0.5, 0 1) red (cat,:1.3) COUPLE(0.5 1, 0 1) (blue:1.2) dog,:0.1]"
    workflow["1"]["inputs"]["text"] = prompt
    assert_prompt(comfyui, workflow)
