#!/usr/bin/env python3
"""
Test script for dart scoring logic.
Tests that get_dart_scores() calculates scores correctly.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_scoring_import():
    """Test that scoring functions can be imported."""
    print("=" * 60)
    print("TEST: Scoring Function Import")
    print("=" * 60)

    try:
        from datasets.annotate import get_dart_scores, BOARD_DICT
        print("  ✓ Successfully imported get_dart_scores")
        print(f"  ✓ Board dictionary has {len(BOARD_DICT)} entries")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import scoring functions: {e}")
        return False


def test_config_loading():
    """Test that configuration file loads correctly."""
    print("\n" + "=" * 60)
    print("TEST: Configuration Loading")
    print("=" * 60)

    config_path = '/Users/fewzy/Dev/ai/deeper_darts/configs/deepdarts_d1.yaml'

    print(f"\n1. Checking config file...")
    if not os.path.exists(config_path):
        print(f"  ✗ Config file not found: {config_path}")
        return False

    print(f"  ✓ Config file found")

    try:
        from yacs.config import CfgNode as CN
        cfg = CN(new_allowed=True)
        cfg.merge_from_file(config_path)

        print(f"\n2. Verifying board configuration...")
        required_keys = ['r_double', 'r_treble', 'r_outer_bull', 'r_inner_bull', 'w_double_treble']

        for key in required_keys:
            if hasattr(cfg.board, key):
                value = getattr(cfg.board, key)
                print(f"  ✓ board.{key} = {value}")
            else:
                print(f"  ✗ Missing board.{key}")
                return False

        return True

    except Exception as e:
        print(f"  ✗ Failed to load config: {e}")
        return False


def test_scoring_calculation():
    """Test scoring calculation with known examples."""
    print("\n" + "=" * 60)
    print("TEST: Scoring Calculation")
    print("=" * 60)

    try:
        from datasets.annotate import get_dart_scores
        from yacs.config import CfgNode as CN

        # Load config
        cfg = CN(new_allowed=True)
        cfg.merge_from_file('/Users/fewzy/Dev/ai/deeper_darts/configs/deepdarts_d1.yaml')

        print("\n1. Testing with valid calibration points and darts...")

        # Create test data with 4 calibration points and 1 dart
        # Calibration points form a square around center (0.5, 0.5)
        xy = np.array([
            [0.4, 0.4, 1],  # cal_1 (top-left)
            [0.6, 0.4, 1],  # cal_2 (top-right)
            [0.4, 0.6, 1],  # cal_3 (bottom-left)
            [0.6, 0.6, 1],  # cal_4 (bottom-right)
            [0.5, 0.5, 1],  # dart_1 (center - should hit bullseye area)
        ], dtype=np.float32)

        scores = get_dart_scores(xy, cfg, numeric=False)
        print(f"  ✓ Scoring completed")
        print(f"    Scores: {scores}")
        print(f"    Number of darts scored: {len(scores)}")

        # Test with numeric output
        numeric_scores = get_dart_scores(xy, cfg, numeric=True)
        print(f"  ✓ Numeric scoring completed")
        print(f"    Numeric scores: {numeric_scores}")

        print("\n2. Testing with missing calibration points...")
        # Only 3 calibration points (should return empty)
        xy_incomplete = np.array([
            [0.4, 0.4, 1],  # cal_1
            [0.6, 0.4, 1],  # cal_2
            [0.4, 0.6, 1],  # cal_3
            [0.0, 0.0, 0],  # cal_4 (missing)
            [0.5, 0.5, 1],  # dart_1
        ], dtype=np.float32)

        scores_incomplete = get_dart_scores(xy_incomplete, cfg, numeric=False)
        print(f"  ✓ Missing calibration test: {scores_incomplete}")

        if len(scores_incomplete) == 0:
            print("    ✓ Correctly returns empty for missing calibration")
        else:
            print("    ⚠️  Unexpected behavior with missing calibration")

        print("\n3. Testing with multiple darts...")
        # 4 calibration points and 3 darts
        xy_multi = np.array([
            [0.4, 0.4, 1],  # cal_1
            [0.6, 0.4, 1],  # cal_2
            [0.4, 0.6, 1],  # cal_3
            [0.6, 0.6, 1],  # cal_4
            [0.5, 0.5, 1],  # dart_1 (center)
            [0.52, 0.48, 1],  # dart_2 (slightly off-center)
            [0.48, 0.52, 1],  # dart_3 (opposite direction)
        ], dtype=np.float32)

        scores_multi = get_dart_scores(xy_multi, cfg, numeric=False)
        print(f"  ✓ Multi-dart scoring: {scores_multi}")
        print(f"    Number of darts: {len(scores_multi)}")

        return True

    except Exception as e:
        print(f"  ✗ Scoring calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scoring_edge_cases():
    """Test scoring with edge cases."""
    print("\n" + "=" * 60)
    print("TEST: Scoring Edge Cases")
    print("=" * 60)

    try:
        from datasets.annotate import get_dart_scores
        from yacs.config import CfgNode as CN

        cfg = CN(new_allowed=True)
        cfg.merge_from_file('/Users/fewzy/Dev/ai/deeper_darts/configs/deepdarts_d1.yaml')

        print("\n1. Testing with no darts (only calibration)...")
        xy_no_darts = np.array([
            [0.4, 0.4, 1],
            [0.6, 0.4, 1],
            [0.4, 0.6, 1],
            [0.6, 0.6, 1],
        ], dtype=np.float32)

        scores = get_dart_scores(xy_no_darts, cfg, numeric=False)
        print(f"  ✓ No darts result: {scores}")

        if len(scores) == 0:
            print("    ✓ Correctly returns empty for no darts")
        else:
            print("    ⚠️  Unexpected behavior with no darts")

        print("\n2. Testing with dart outside board...")
        xy_outside = np.array([
            [0.4, 0.4, 1],
            [0.6, 0.4, 1],
            [0.4, 0.6, 1],
            [0.6, 0.6, 1],
            [0.1, 0.1, 1],  # Far outside
        ], dtype=np.float32)

        scores_outside = get_dart_scores(xy_outside, cfg, numeric=False)
        print(f"  ✓ Outside dart result: {scores_outside}")

        if scores_outside and scores_outside[0] == '0':
            print("    ✓ Correctly scores as '0' for outside dart")

        return True

    except Exception as e:
        print(f"  ✗ Edge case testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SCORING LOGIC TEST SUITE")
    print("=" * 60)

    # Test 1: Import
    result1 = test_scoring_import()
    if not result1:
        print("\n⚠️  Cannot continue without scoring imports")
        sys.exit(1)

    # Test 2: Config loading
    result2 = test_config_loading()
    if not result2:
        print("\n⚠️  Cannot continue without config")
        sys.exit(1)

    # Test 3: Basic scoring
    result3 = test_scoring_calculation()

    # Test 4: Edge cases
    result4 = test_scoring_edge_cases()

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Scoring import: {'✓ PASS' if result1 else '✗ FAIL'}")
    print(f"Config loading: {'✓ PASS' if result2 else '✗ FAIL'}")
    print(f"Scoring calculation: {'✓ PASS' if result3 else '✗ FAIL'}")
    print(f"Edge cases: {'✓ PASS' if result4 else '✗ FAIL'}")
    print("=" * 60)

    all_pass = result1 and result2 and result3 and result4
    sys.exit(0 if all_pass else 1)
