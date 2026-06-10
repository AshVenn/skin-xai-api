import base64
import io
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

from app.schemas import TopKPrediction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "final_efficientnet_b0_xai_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_IMAGENET_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGENET_STD = [0.229, 0.224, 0.225]


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activations
        )

    def _save_activations(self, module, input_tensor, output_tensor):
        self.activations = output_tensor

        def _save_gradients(grad):
            self.gradients = grad

        output_tensor.register_hook(_save_gradients)

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        target_score = logits[:, target_class].sum()
        target_score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM failed: missing activations or gradients.")

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()

        cam_min = float(cam.min())
        cam_max = float(cam.max())

        if cam_max - cam_min < 1e-8:
            return np.zeros_like(cam, dtype=np.float32)

        cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.astype(np.float32)


class SkinDiseaseModelService:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        self.lock = threading.Lock()

        self.checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False,
        )

        self.num_classes = int(self.checkpoint["num_classes"])
        self.image_size = int(self.checkpoint.get("image_size", 224))

        self.idx_to_class: Dict[str, str] = self.checkpoint["idx_to_class"]
        self.class_to_idx: Dict[str, int] = self.checkpoint["class_to_idx"]

        preprocessing = self.checkpoint.get("preprocessing", {})
        self.mean = preprocessing.get("imagenet_mean", DEFAULT_IMAGENET_MEAN)
        self.std = preprocessing.get("imagenet_std", DEFAULT_IMAGENET_STD)

        self.model = self._build_model()
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        self.gradcam = GradCAM(
            model=self.model,
            target_layer=self.model.features[-1],
        )

    def _build_model(self) -> nn.Module:
        model = efficientnet_b0(weights=None)

        input_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features=input_features,
            out_features=self.num_classes,
        )

        model.load_state_dict(self.checkpoint["model_state_dict"])
        model.to(DEVICE)

        return model

    def _prepare_image(self, image_bytes: bytes) -> Tuple[Image.Image, torch.Tensor]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)

        return image, input_tensor

    def _softmax_top_k(
        self,
        logits: torch.Tensor,
        k: int = 3,
    ) -> Tuple[int, float, List[TopKPrediction]]:
        probabilities = torch.softmax(logits, dim=1)[0]
        top_probabilities, top_indices = torch.topk(probabilities, k=k)

        top_k = []

        for probability, class_index in zip(top_probabilities, top_indices):
            class_id = int(class_index.item())
            confidence = float(probability.item())

            top_k.append(
                TopKPrediction(
                    class_id=class_id,
                    class_name=self.idx_to_class[str(class_id)],
                    confidence=confidence,
                )
            )

        predicted_id = top_k[0].class_id
        predicted_confidence = top_k[0].confidence

        return predicted_id, predicted_confidence, top_k

    def _create_heatmap_overlay_base64(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
    ) -> str:
        resized_image = original_image.resize(
            (self.image_size, self.image_size)
        ).convert("RGB")

        image_np = np.array(resized_image).astype(np.uint8)

        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(
            image_np,
            1.0 - alpha,
            heatmap_color,
            alpha,
            0,
        )

        overlay_image = Image.fromarray(overlay)

        buffer = io.BytesIO()
        overlay_image.save(buffer, format="PNG")

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded

    def predict_with_gradcam(self, image_bytes: bytes) -> dict:
        with self.lock:
            original_image, input_tensor = self._prepare_image(image_bytes)

            logits = self.model(input_tensor)

            predicted_id, confidence, top_k = self._softmax_top_k(
                logits=logits,
                k=3,
            )

            heatmap = self.gradcam.generate(
                input_tensor=input_tensor,
                target_class=predicted_id,
            )

            heatmap_base64 = self._create_heatmap_overlay_base64(
                original_image=original_image,
                heatmap=heatmap,
                alpha=0.45,
            )

            return {
                "predicted_class_id": predicted_id,
                "predicted_class": self.idx_to_class[str(predicted_id)],
                "confidence": confidence,
                "top_k": top_k,
                "heatmap_base64": heatmap_base64,
            }


model_service = SkinDiseaseModelService()