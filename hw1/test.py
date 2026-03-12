import argparse
import sys
from pathlib import Path

import yaml

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.dataclass import (  # noqa: E402
    DataConfig,
    ModelConfig,
    OutputConfig,
    TestConfig,
)
from src.services.implement.test import Tester  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def load_config(config_path: str) -> TestConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    model_cfg = ModelConfig(
        backbone=raw["model"].get("backbone", "resnet50"),
        pretrained=False,
        num_classes=raw["model"].get("num_classes", 100),
        checkpoint=raw["model"].get("checkpoint", None),
    )
    data_raw = raw.get("data", {})
    data_cfg = DataConfig(
        test_dir=data_raw.get("test_dir", "data/test"),
        image_size=data_raw.get("image_size", 224),
        crop_size=data_raw.get("crop_size", 320),
        resize_size=data_raw.get("resize_size", 334),
        batch_size=data_raw.get("batch_size", 64),
        num_workers=data_raw.get("num_workers", 4),
    )
    output_cfg = OutputConfig(
        submission_file=raw.get("output", {}).get(
            "submission_file", "submission.csv"
        )
    )

    return TestConfig(model=model_cfg, data=data_cfg, output=output_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference and generate submission CSV"
    )
    parser.add_argument("--config", type=str, default="configs/test.yaml")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.config)

    tester = Tester()
    tester.setup(config)
    predictions = tester.predict()
    tester.save_submission(predictions, config.output.submission_file)


if __name__ == "__main__":
    main()
