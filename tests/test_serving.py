"""API contracts and deployment gates; real checkpoint inference is a separate container check."""

from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
import numpy as np
from PIL import Image

from src.serving.api import app, decode_image, load_model, predict_image
from src.serving.deploy import stage_model


async def request(method, path, body=b"", content_type="image/png", raw=False):
    messages = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    await app({"type": "http", "http_version": "1.1", "scheme": "http", "method": method,
               "path": path, "raw_path": path.encode(), "query_string": b"",
               "headers": [(b"content-type", content_type.encode())],
               "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 8000)}, receive, send)
    payload = b"".join(m.get("body", b"") for m in messages[1:])
    return messages[0]["status"], payload.decode() if raw else json.loads(payload)


class ServingTests(unittest.IsolatedAsyncioTestCase):
    async def test_metrics_outcomes_latency_and_scrape_exclusion(self):
        from prometheus_client.parser import text_string_to_metric_families

        manifest = {"model_name": "fixture", "model_version": "7", "checkpoint_sha256": "abc",
                    "deployment_mode": "experimental", "inference": {
                        "max_upload_bytes": 1024, "max_image_pixels": 100}}
        with patch("src.serving.api.load_model", return_value=(object(), manifest)), \
                patch("src.serving.api.predict_image", return_value=[]) as inference:
            async with app.router.lifespan_context(app):
                image = BytesIO()
                Image.new("RGB", (4, 5)).save(image, format="PNG")
                await request("GET", "/health")
                await request("GET", "/metrics", raw=True)
                await request("POST", "/predict", image.getvalue())
                await request("POST", "/predict", b"corrupt")
                inference.side_effect = RuntimeError("fixture failure")
                with self.assertLogs("src.serving.api", level="ERROR"):
                    await request("POST", "/predict", image.getvalue())
                _, text = await request("GET", "/metrics", raw=True)
                samples = [s for family in text_string_to_metric_families(text) for s in family.samples]
                counts = {s.labels["outcome"]: s.value for s in samples if s.name == "va_prediction_requests_total"}
                self.assertEqual(counts, {"success": 1, "client_error": 1, "server_error": 1})
                self.assertEqual(sum(s.value for s in samples if s.name == "va_prediction_duration_seconds_count"), 3)
                self.assertEqual(next(s.value for s in samples if s.name == "va_model_ready"), 1)
                info = next(s for s in samples if s.name == "va_model_info")
                self.assertEqual(info.labels["model_version"], "7")
                self.assertEqual(info.labels["deployment_mode"], "experimental")
                self.assertIn("deployment_id", info.labels)

    async def test_readiness_input_errors_and_prediction_identity(self):
        manifest = {"model_name": "fixture", "model_version": "7", "checkpoint_sha256": "abc",
                    "deployment_mode": "experimental", "inference": {
                        "max_upload_bytes": 1024, "max_image_pixels": 100}}
        app.state.ready = False
        self.assertEqual((await request("GET", "/health"))[0], 503)
        with patch("src.serving.api.load_model", return_value=(object(), manifest)), \
                patch("src.serving.api.predict_image", return_value=[]) as inference:
            async with app.router.lifespan_context(app):
                self.assertEqual((await request("GET", "/health"))[1]["model_version"], "7")
                for body, content_type, expected in [(b"{}", "application/json", 415),
                                                     (b"bad", "image/png", 422),
                                                     (b"x" * 1025, "image/png", 413)]:
                    self.assertEqual((await request("POST", "/predict", body, content_type))[0], expected)
                self.assertFalse(inference.called)
                image = BytesIO()
                Image.new("RGB", (4, 5)).save(image, format="PNG")
                code, result = await request("POST", "/predict", image.getvalue())
                self.assertEqual(code, 200)
                self.assertEqual((result["width"], result["height"], result["detections"]), (4, 5, []))
                self.assertEqual(result["deployment_mode"], "experimental")
                self.assertEqual(result["inference_sha256"], hashlib.sha256(
                    json.dumps(manifest["inference"], sort_keys=True).encode()).hexdigest())
                inference.side_effect = RuntimeError("fixture failure")
                with self.assertLogs("src.serving.api", level="ERROR"):
                    code, result = await request("POST", "/predict", image.getvalue())
                self.assertEqual((code, result), (500, {"detail": "Model inference failed"}))
        self.assertEqual((await request("GET", "/health"))[0], 503)

    def test_decoded_pixel_bound_and_corrupt_checkpoint(self):
        image = BytesIO()
        Image.new("RGB", (11, 10)).save(image, format="PNG")
        with self.assertRaises(HTTPException) as raised:
            decode_image(image.getvalue(), 100)
        self.assertEqual(raised.exception.status_code, 413)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "manifest.json").write_text(json.dumps({"checkpoint_sha256": "wrong"}))
            (directory / "best.pt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checkpoint does not match"):
                load_model(directory)

    def test_boxes_return_to_original_coordinates(self):
        import torch

        boxes = SimpleNamespace(xyxy=torch.tensor([[0., 104., 416., 312.]]),
                                cls=torch.tensor([0.]), conf=torch.tensor([0.9]))
        model = MagicMock(names={0: "helmet"})
        model.predict.return_value = [SimpleNamespace(boxes=boxes)]
        settings = dict(image_size=416, pad_value=114, confidence=0.25, iou=0.7, max_detections=300)
        result = predict_image(model, np.zeros((500, 1000, 3), dtype=np.uint8), settings)
        self.assertEqual(result[0]["xyxy"], [0., 0., 1000., 500.])
        self.assertEqual(model.predict.call_args.kwargs["device"], "cpu")

    def test_unapproved_version_refused_before_artifact_download(self):
        client = MagicMock()
        client.get_model_version.return_value = SimpleNamespace(version="1")
        client.get_registered_model.return_value = SimpleNamespace(aliases={})
        project = Path(__file__).resolve().parents[1]
        with patch("src.serving.deploy.local_store", return_value=(client, None, {"model_name": "fixture"})):
            with self.assertRaisesRegex(ValueError, "not referenced by stable"):
                stage_model(project, project, "1")
            with self.assertRaisesRegex(ValueError, "explicit numeric"):
                stage_model(project, project, experimental=True)
        client.download_artifacts.assert_not_called()


if __name__ == "__main__":
    unittest.main()
