"""
Unit tests for CrowdSight models.

Runs without a GPU and without any dataset — uses random tensors.
All tests must be fast (<30 s total on CPU).

Run:
    pytest tests/ -v
"""

import pytest
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def random_image(batch: int = 1, h: int = 256, w: int = 256) -> torch.Tensor:
    """ImageNet-normalised random image tensor."""
    return torch.randn(batch, 3, h, w)


def count_from_density(density: torch.Tensor) -> float:
    return float(density.sum().item())


# ─────────────────────────────────────────────────────────────────────────────
#  Model registry
# ─────────────────────────────────────────────────────────────────────────────

def test_get_model_csrnet():
    from model import get_model
    m = get_model("csrnet", pretrained=False)
    assert m is not None


def test_get_model_mac_cnn():
    from model import get_model
    m = get_model("mac_cnn")
    assert m is not None


def test_get_model_unknown_raises():
    from model import get_model
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("does_not_exist")


# ─────────────────────────────────────────────────────────────────────────────
#  CSRNet
# ─────────────────────────────────────────────────────────────────────────────

class TestCSRNet:
    @pytest.fixture(scope="class")
    def model(self):
        from model import get_model
        m = get_model("csrnet", pretrained=False)
        m.eval()
        return m

    def test_output_stride(self, model):
        assert model.output_stride == 8

    def test_forward_shape(self, model):
        x = random_image(1, 256, 256)
        with torch.no_grad():
            y = model(x)
        # (B, 1, H/8, W/8)
        assert y.shape == (1, 1, 32, 32), f"Unexpected shape: {y.shape}"

    def test_forward_batch(self, model):
        x = random_image(2, 256, 256)
        with torch.no_grad():
            y = model(x)
        assert y.shape[0] == 2

    def test_non_square_input(self, model):
        x = random_image(1, 256, 320)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1, 32, 40)

    def test_output_non_negative(self, model):
        x = random_image(1, 256, 256)
        with torch.no_grad():
            y = model(x)
        assert float(y.min().item()) >= 0.0

    def test_num_params(self, model):
        # CSRNet should have >1M trainable params
        assert model.num_params > 1_000_000

    def test_count_is_sum(self, model):
        x = random_image(1, 256, 256)
        with torch.no_grad():
            density = model(x)
        count = count_from_density(density)
        assert isinstance(count, float)


# ─────────────────────────────────────────────────────────────────────────────
#  MAC-CNN
# ─────────────────────────────────────────────────────────────────────────────

class TestMACCNN:
    @pytest.fixture(scope="class")
    def model(self):
        from model import get_model
        m = get_model("mac_cnn")
        m.eval()
        return m

    def test_output_stride(self, model):
        assert hasattr(model, "output_stride")

    def test_forward_shape(self, model):
        s = model.output_stride
        x = random_image(1, 256, 256)
        with torch.no_grad():
            y = model(x)
        assert y.shape[2] == 256 // s
        assert y.shape[3] == 256 // s

    def test_output_non_negative(self, model):
        x = random_image(1, 256, 256)
        with torch.no_grad():
            y = model(x)
        assert float(y.min().item()) >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Losses
# ─────────────────────────────────────────────────────────────────────────────

class TestLosses:
    def test_mse_loss(self):
        from model.losses import MSELoss
        fn = MSELoss()
        pred = torch.rand(1, 1, 32, 32)
        gt   = torch.rand(1, 1, 32, 32)
        loss = fn(pred, gt)
        assert loss.item() >= 0.0

    def test_dmcount_loss_returns_tuple(self):
        from model.losses import DMCountLoss
        fn   = DMCountLoss()
        pred = torch.rand(1, 1, 32, 32)
        gt   = torch.rand(1, 1, 32, 32)
        loss, components = fn(pred, gt)
        assert loss.item() >= 0.0
        assert "ot" in components and "tv" in components and "count" in components

    def test_get_loss_factory(self):
        from model import get_loss
        assert get_loss("mse") is not None
        assert get_loss("dmcount") is not None


# ─────────────────────────────────────────────────────────────────────────────
#  Inference engine (no weights — random model)
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceEngine:
    @pytest.fixture(scope="class")
    def engine(self):
        from api.inference import CrowdInferenceEngine
        return CrowdInferenceEngine(weights_path=None, model_name="csrnet")

    def test_weights_not_loaded(self, engine):
        assert engine.weights_loaded is False

    def test_device_string(self, engine):
        assert engine.device_name in ("cpu", "cuda", "mps")

    def test_analyze_returns_required_keys(self, engine):
        from PIL import Image
        import numpy as np
        arr = (np.random.rand(256, 256, 3) * 255).astype("uint8")
        img = Image.fromarray(arr)
        result = engine.analyze(img)
        assert "count" in result
        assert "density_map" in result
        assert "inference_time_ms" in result

    def test_analyze_count_is_float(self, engine):
        from PIL import Image
        import numpy as np
        arr = (np.random.rand(128, 128, 3) * 255).astype("uint8")
        img = Image.fromarray(arr)
        result = engine.analyze(img)
        assert isinstance(result["count"], float)

    def test_analyze_heatmap_is_base64_png(self, engine):
        import base64
        from PIL import Image
        import numpy as np
        import io
        arr = (np.random.rand(128, 128, 3) * 255).astype("uint8")
        img = Image.fromarray(arr)
        result = engine.analyze(img)
        raw = base64.b64decode(result["density_map"])
        png = Image.open(io.BytesIO(raw))
        assert png.format == "PNG"

    def test_analyze_non_square_image(self, engine):
        from PIL import Image
        import numpy as np
        arr = (np.random.rand(480, 640, 3) * 255).astype("uint8")
        img = Image.fromarray(arr)
        result = engine.analyze(img)
        assert result["count"] >= 0.0
