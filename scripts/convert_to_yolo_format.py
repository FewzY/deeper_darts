"""
Convert DeepDarts labels.pkl to YOLO11 format.

This script converts the DeepDarts dataset from pandas DataFrame format to
YOLO11 txt annotations with proper train/val/test splits.

Usage:
    python scripts/convert_to_yolo_format.py

Author: Claude Code (SPARC Implementation)
Date: 2025-10-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import cv2
from tqdm import tqdm
import yaml
import argparse
from typing import Tuple, List, Set, Optional, Union


class DeepDartsConverter:
    """Convert DeepDarts format to YOLO11 format."""

    def __init__(
        self,
        labels_path: str = 'datasets/labels.pkl',
        images_dir: str = 'datasets/images',
        output_dir: str = 'datasets/yolo_format',
        keypoint_bbox_size: float = 0.025  # 2.5% following DeepDarts paper
    ):
        """
        Initialize converter.

        Args:
            labels_path: Path to labels.pkl file
            images_dir: Directory containing session folders with images
            output_dir: Output directory for YOLO format dataset
            keypoint_bbox_size: Bounding box size for keypoints (normalized)
        """
        self.labels_path = labels_path
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.keypoint_bbox_size = keypoint_bbox_size

        # Statistics
        self.stats = {
            'total': 0,
            'successful': 0,
            'skipped': 0,
            'errors': []
        }

        # Load labels
        print(f"Loading labels from {labels_path}...")
        self.df = pd.read_pickle(labels_path)
        print(f"✅ Loaded {len(self.df)} samples")

    def convert(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> None:
        """
        Convert dataset to YOLO format with train/val/test split.

        Args:
            train_ratio: Ratio of sessions for training
            val_ratio: Ratio of sessions for validation
            seed: Random seed for reproducibility
        """
        print("\n" + "="*60)
        print("🚀 YOLO11 Dataset Conversion")
        print("="*60)

        # Create output directories
        self._create_directories()

        # Split dataset by sessions (prevents data leakage)
        train_sessions, val_sessions, test_sessions = self._split_sessions(
            train_ratio, val_ratio, seed
        )

        # Convert each split
        print("\n📦 Converting splits...")
        self._convert_split(train_sessions, 'train')
        self._convert_split(val_sessions, 'val')
        self._convert_split(test_sessions, 'test')

        # Create data.yaml
        self._create_yaml()

        # Print final statistics
        self._print_final_stats()

    def _create_directories(self) -> None:
        """Create output directory structure."""
        print("\n📁 Creating directory structure...")

        for split in ['train', 'val', 'test']:
            img_dir = self.output_dir / 'images' / split
            label_dir = self.output_dir / 'labels' / split

            img_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

        print("✅ Directory structure created")

    def _split_sessions(
        self,
        train_ratio: float,
        val_ratio: float,
        seed: int
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Split dataset by session to avoid data leakage.

        Args:
            train_ratio: Ratio for training
            val_ratio: Ratio for validation
            seed: Random seed

        Returns:
            Tuple of (train_sessions, val_sessions, test_sessions)
        """
        sessions = self.df['img_folder'].unique()
        np.random.seed(seed)
        np.random.shuffle(sessions)

        n_total = len(sessions)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        train_sessions = set(sessions[:n_train])
        val_sessions = set(sessions[n_train:n_train + n_val])
        test_sessions = set(sessions[n_train + n_val:])

        print(f"\n📊 Dataset Split:")
        print(f"  Train: {len(train_sessions)} sessions ({train_ratio*100:.0f}%)")
        print(f"  Val:   {len(val_sessions)} sessions ({val_ratio*100:.0f}%)")
        print(f"  Test:  {len(test_sessions)} sessions ({(1-train_ratio-val_ratio)*100:.0f}%)")

        return train_sessions, val_sessions, test_sessions

    def _convert_split(self, sessions: Set[str], split_name: str) -> None:
        """
        Convert images and labels for a split.

        Args:
            sessions: Set of session names for this split
            split_name: Name of split ('train', 'val', or 'test')
        """
        df_split = self.df[self.df['img_folder'].isin(sessions)]

        print(f"\n🔄 Converting {split_name} split ({len(df_split)} images)...")

        split_stats = {'successful': 0, 'skipped': 0}

        for idx, row in tqdm(df_split.iterrows(), total=len(df_split),
                            desc=f"  {split_name}"):
            try:
                if self._convert_sample(row, split_name):
                    split_stats['successful'] += 1
                    self.stats['successful'] += 1
                else:
                    split_stats['skipped'] += 1
                    self.stats['skipped'] += 1
            except Exception as e:
                error_msg = f"Error in {row['img_name']}: {str(e)}"
                self.stats['errors'].append(error_msg)
                split_stats['skipped'] += 1
                self.stats['skipped'] += 1

        print(f"  ✅ Successful: {split_stats['successful']}")
        print(f"  ⚠️  Skipped: {split_stats['skipped']}")

    def _convert_sample(self, row: pd.Series, split: str) -> bool:
        """
        Convert a single sample.

        Args:
            row: DataFrame row containing sample data
            split: Split name ('train', 'val', or 'test')

        Returns:
            True if successful, False otherwise
        """
        img_folder = row['img_folder']
        img_name = row['img_name']
        bbox = row['bbox']  # [x, y, w, h]
        keypoints = row['xy']  # List of [x, y] normalized coordinates

        # Find image (handle case variations)
        img_path = self._find_image(img_folder, img_name)
        if img_path is None:
            return False

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            return False

        img_h, img_w = img.shape[:2]

        # Crop to dartboard
        x, y, w, h = bbox

        # Ensure bbox is within image bounds
        x = max(0, min(x, img_w))
        y = max(0, min(y, img_h))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            return False

        cropped = img[y:y+h, x:x+w]

        # Save cropped image
        new_img_name = f"{img_folder}_{img_name}"
        img_out_path = self.output_dir / 'images' / split / new_img_name
        cv2.imwrite(str(img_out_path), cropped)

        # Convert keypoints to YOLO format
        yolo_labels = self._keypoints_to_yolo(keypoints, bbox, img_w, img_h, w, h)

        # Write label file
        label_out_path = self.output_dir / 'labels' / split / f"{Path(new_img_name).stem}.txt"
        with open(label_out_path, 'w') as f:
            f.write('\n'.join(yolo_labels))

        return True

    def _find_image(self, folder: str, name: str) -> Optional[Path]:
        """
        Find image with potential case variations.

        Args:
            folder: Session folder name
            name: Image filename

        Returns:
            Path to image or None if not found
        """
        folder_path = self.images_dir / folder

        # Try exact name first
        img_path = folder_path / name
        if img_path.exists():
            return img_path

        # Try alternate extensions
        base_name = Path(name).stem
        for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            img_path = folder_path / f"{base_name}{ext}"
            if img_path.exists():
                return img_path

        return None

    def _keypoints_to_yolo(
        self,
        keypoints: List[List[float]],
        bbox: List[int],
        img_w: int,
        img_h: int,
        crop_w: int,
        crop_h: int
    ) -> List[str]:
        """
        Convert keypoints to YOLO format.

        Args:
            keypoints: List of [x, y] normalized coordinates
            bbox: Crop bounding box [x, y, w, h]
            img_w: Original image width
            img_h: Original image height
            crop_w: Crop width
            crop_h: Crop height

        Returns:
            List of YOLO format labels
        """
        x_crop, y_crop, _, _ = bbox
        yolo_labels = []

        for i, keypoint in enumerate(keypoints):
            x_norm, y_norm = keypoint

            # Convert to absolute coordinates
            x_abs = x_norm * img_w
            y_abs = y_norm * img_h

            # Convert to crop-relative coordinates
            x_rel = (x_abs - x_crop) / crop_w
            y_rel = (y_abs - y_crop) / crop_h

            # Skip if outside crop (with small tolerance)
            if not (-0.01 <= x_rel <= 1.01 and -0.01 <= y_rel <= 1.01):
                continue

            # Clamp to valid range
            x_rel = max(0, min(1, x_rel))
            y_rel = max(0, min(1, y_rel))

            # Determine class (0-3 for calibration, 4 for dart)
            class_id = i if i < 4 else 4

            # YOLO format: class x_center y_center width height
            yolo_label = (
                f"{class_id} "
                f"{x_rel:.6f} {y_rel:.6f} "
                f"{self.keypoint_bbox_size:.6f} {self.keypoint_bbox_size:.6f}"
            )
            yolo_labels.append(yolo_label)

        return yolo_labels

    def _create_yaml(self) -> None:
        """Create data.yaml configuration file for YOLO training."""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 5,
            'names': {
                0: 'calibration_5_20',
                1: 'calibration_13_6',
                2: 'calibration_17_3',
                3: 'calibration_8_11',
                4: 'dart_tip'
            }
        }

        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        print(f"\n📝 Created {yaml_path}")

    def _print_final_stats(self) -> None:
        """Print final conversion statistics."""
        print("\n" + "="*60)
        print("📊 Conversion Statistics")
        print("="*60)

        for split in ['train', 'val', 'test']:
            img_count = len(list((self.output_dir / 'images' / split).glob('*')))
            label_count = len(list((self.output_dir / 'labels' / split).glob('*.txt')))
            print(f"{split.capitalize():5s}: {img_count:5d} images, {label_count:5d} labels")

        total = self.stats['successful'] + self.stats['skipped']
        success_rate = (self.stats['successful'] / total * 100) if total > 0 else 0

        print(f"\nTotal:     {total:5d} samples")
        print(f"Successful: {self.stats['successful']:5d} ({success_rate:.1f}%)")
        print(f"Skipped:    {self.stats['skipped']:5d}")

        if self.stats['errors']:
            print(f"\n⚠️  Errors ({len(self.stats['errors'])}):")
            for error in self.stats['errors'][:5]:
                print(f"  - {error}")
            if len(self.stats['errors']) > 5:
                print(f"  ... and {len(self.stats['errors']) - 5} more")

        print("\n✅ Dataset conversion complete!")
        print(f"📁 Output: {self.output_dir.absolute()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Convert DeepDarts dataset to YOLO11 format'
    )
    parser.add_argument(
        '--labels-path',
        type=str,
        default='datasets/labels.pkl',
        help='Path to labels.pkl file'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        default='datasets/images',
        help='Directory containing image folders'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets/yolo_format',
        help='Output directory for YOLO format'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Ratio of sessions for training'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Ratio of sessions for validation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Create converter
    converter = DeepDartsConverter(
        labels_path=args.labels_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir
    )

    # Run conversion
    converter.convert(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
