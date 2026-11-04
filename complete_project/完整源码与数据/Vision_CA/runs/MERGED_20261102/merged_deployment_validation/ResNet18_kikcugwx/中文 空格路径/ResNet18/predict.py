"""读取完整照片，使用已冻结的手检测器与ResNet输出一个0—5指令。

所有权重从本程序旁的包目录读取；构建模型时不访问网络或预训练缓存。
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Union

from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import SSDLiteFeatureExtractorMobileNet, SSDLiteHead


IMAGE_NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
SUPPORTED_ARCHITECTURES = {"resnet18", "resnet34", "resnet50"}


def build_hand_detector() -> SSD:
    """离线构建训练使用的完整尾部SSDLite，避免预训练开关改变结构。"""
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    # weights_backbone=None的快捷工厂会缩窄尾部，因此显式构建完整MobileNet。
    features = models.mobilenet_v3_large(
        weights=None, reduced_tail=False, norm_layer=norm_layer
    ).features
    stages = [0] + [i for i, block in enumerate(features) if getattr(block, "_is_cn", False)] + [len(features) - 1]
    backbone = SSDLiteFeatureExtractorMobileNet(features, stages[-2], norm_layer).eval()
    with torch.inference_mode():
        channels = [feature.shape[1] for feature in backbone(torch.zeros(1, 3, 320, 320)).values()]
    if len(channels) != 6 or channels[1] != 960:
        raise RuntimeError(f"手检测器结构不匹配，实际通道为{channels}；请使用requirements.txt指定版本")
    anchors = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    head = SSDLiteHead(channels, anchors.num_anchors_per_location(), 2, norm_layer)
    # 下列变换和NMS参数与训练、正式评价使用的TorchVision工厂一致。
    return SSD(
        backbone, anchors, (320, 320), 2, head=head,
        score_thresh=0.001, nms_thresh=0.55, detections_per_img=300,
        topk_candidates=300, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5]
    ).eval()


def crop_hand(image: Image.Image, box: list[float]) -> Image.Image:
    """按手框长边各留10%边缘，越界补黑，再缩放成224×224。"""
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * 1.2
    if side <= 0:
        raise ValueError("检测器返回了无效的手框")
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    rectangle = (math.floor(center_x - side / 2), math.floor(center_y - side / 2),
                 math.ceil(center_x + side / 2), math.ceil(center_y + side / 2))
    return image.crop(rectangle).resize((224, 224), Image.Resampling.BILINEAR)


def aggregate_commands(labels: list[int]) -> int:
    """停止优先；0不抵消有效指令，不同有效动作冲突时输出0。"""
    if 3 in labels:
        return 3
    active = {int(label) for label in labels if label != 0}
    return next(iter(active)) if len(active) == 1 else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GesturePredictor:
    """一次加载模型后，可连续预测多张照片，避免每张照片重复加载权重。"""

    def __init__(self, config_path: Union[str, Path, None] = None,
                 device: str = "cpu", threads: int = 8):
        if threads < 1:
            raise ValueError("threads必须为正整数")
        torch.set_num_threads(threads)
        self.config_path = Path(config_path).resolve() if config_path else Path(__file__).resolve().parent / "config.json"
        self.root = self.config_path.parent
        self.config = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
        self.architecture = self.config["architecture"]
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"不支持的架构：{self.architecture}")
        self.mapping = self.config["mapping"]
        if set(self.mapping) != {str(label) for label in range(6)}:
            raise ValueError("mapping必须完整列出0—5六类")
        if any(not entry.get("name") or not entry.get("command") for entry in self.mapping.values()):
            raise ValueError("mapping的每一类必须包含name与command")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("device只支持cpu、cuda或auto")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前PyTorch环境无法使用CUDA，请改为--device cpu或安装匹配的GPU版本")
        self.device = torch.device(device)
        thresholds = self.config["thresholds"]
        self.detector_threshold = float(thresholds["detector"])
        self.classifier_threshold = float(thresholds["classifier"])
        if not (0 <= self.detector_threshold <= 1 and 0 <= self.classifier_threshold <= 1):
            raise ValueError("检测与分类阈值必须在0到1之间")
        weights = self.config["weights"]
        classifier_path = self._weight_path(weights["classifier"])
        detector_path = self._weight_path(weights["detector"])
        # 只加载包内真实冻结checkpoint，禁止缺失项或形状不符时部分加载。
        classifier_checkpoint = torch.load(classifier_path, map_location="cpu", weights_only=True)
        detector_checkpoint = torch.load(detector_path, map_location="cpu", weights_only=True)
        dropout = float(self.config["classifier_dropout"])
        if not math.isclose(float(classifier_checkpoint["params"]["dropout"]), dropout, abs_tol=1e-12):
            raise ValueError("配置的Dropout与冻结checkpoint不一致")
        self.classifier = getattr(models, self.architecture)(weights=None)
        self.classifier.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.classifier.fc.in_features, 6))
        self.classifier.load_state_dict(classifier_checkpoint["model"], strict=True)
        self.detector = build_hand_detector()
        self.detector.load_state_dict(detector_checkpoint["model"], strict=True)
        self.classifier.to(self.device).eval()
        self.detector.to(self.device).eval()

    def _weight_path(self, specification: dict) -> Path:
        """相对本包解析权重并核对身份，移动整个文件夹不会失效。"""
        relative = Path(specification["path"])
        if relative.is_absolute():
            raise ValueError("权重路径必须相对于程序包，不能依赖开发机器绝对路径")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("权重路径超出程序包目录")
        if not path.is_file():
            raise FileNotFoundError(f"权重文件不存在：{path}")
        expected = str(specification["sha256"]).lower()
        if _sha256(path) != expected:
            raise ValueError(f"权重校验失败：{relative}，请重新解压完整程序包")
        return path

    def predict(self, image: Union[str, Path, Image.Image]) -> int:
        """输入图片路径或PIL图片，返回Python整数0、1、2、3、4或5。"""
        return self.predict_details(image)["prediction"]

    def predict_details(self, image: Union[str, Path, Image.Image]) -> dict:
        """同时返回手框、各类概率和最终整数，便于排查实际照片。"""
        if isinstance(image, Image.Image):
            rgb_image = image.convert("RGB")
            input_name = None
        else:
            image_path = Path(image).expanduser().resolve()
            with Image.open(image_path) as source:
                # 与正式评价一致，不额外旋转EXIF方向或预裁整张照片。
                rgb_image = source.convert("RGB")
            input_name = str(image_path)
        hands = []
        with torch.inference_mode():
            result = self.detector([TF.to_tensor(rgb_image).to(self.device)])[0]
            keep = result["scores"] >= self.detector_threshold
            boxes = result["boxes"][keep].cpu().tolist()
            scores = result["scores"][keep].cpu().tolist()
            if boxes:
                crops = torch.stack([IMAGE_NORMALIZE(TF.to_tensor(crop_hand(rgb_image, box))) for box in boxes]).to(self.device)
                probabilities = self.classifier(crops).float().softmax(1).cpu()
                for box, score, values in zip(boxes, scores, probabilities):
                    confidence, label = values.max(0)
                    label = int(label.item())
                    hands.append({"bbox": box, "detector_score": score, "label": label,
                                  "confidence": float(confidence.item()),
                                  "accepted": bool(confidence.item() >= self.classifier_threshold),
                                  "probabilities": values.tolist()})
        accepted = [hand["label"] for hand in hands if hand["accepted"]]
        prediction = aggregate_commands(accepted)
        if not hands:
            reason = "没有手框达到固定检测阈值"
        elif not accepted:
            reason = "手势置信度未达到固定分类阈值"
        elif 3 in accepted:
            reason = "已接受的停止手势优先"
        elif len({label for label in accepted if label != 0}) > 1:
            reason = "不同有效动作冲突，输出0"
        elif prediction == 0:
            reason = "已接受的手势均属于其他类"
        else:
            reason = "已接受的有效动作一致"
        return {"architecture": self.architecture, "input": input_name,
                "image_size": list(rgb_image.size), "device": str(self.device),
                "thresholds": {"detector": self.detector_threshold, "classifier": self.classifier_threshold},
                "prediction": prediction, "command": self.mapping[str(prediction)]["command"],
                "class_name": self.mapping[str(prediction)]["name"], "reason": reason, "hands": hands}


def main() -> int:
    parser = argparse.ArgumentParser(description="输入完整图片，输出0—5中的一个手势指令整数")
    parser.add_argument("image", type=Path, help="待识别的JPG、PNG等图片路径")
    parser.add_argument("--config", type=Path, default=None, help="配置文件，默认读取程序旁config.json")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu", help="默认CPU，亦可明确选择CUDA")
    parser.add_argument("--threads", type=int, default=8, help="CPU线程数，默认8")
    parser.add_argument("--json", dest="json_path", type=Path, default=None, help="可选：把详细结果保存为JSON")
    args = parser.parse_args()
    try:
        predictor = GesturePredictor(args.config, device=args.device, threads=args.threads)
        details = predictor.predict_details(args.image)
        if args.json_path:
            # 保存诊断信息不能覆盖输入照片、配置或冻结权重。
            protected = {args.image.resolve(), predictor.config_path.resolve()}
            protected.update((predictor.root / value["path"]).resolve()
                             for value in predictor.config["weights"].values())
            if args.json_path.resolve() in protected:
                raise ValueError("JSON结果不能覆盖输入照片、配置或模型权重")
            args.json_path.parent.mkdir(parents=True, exist_ok=True)
            args.json_path.write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # 正常识别时标准输出仅这一行数字，便于其他程序直接读取。
        print(details["prediction"])
        return 0
    except Exception as error:
        print(f"识别失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
