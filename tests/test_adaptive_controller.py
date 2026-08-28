"""
Unit tests for DeepDNS Adaptive Evidence Controller and Sequential CUSUM Detector.
"""

import pytest
import numpy as np
from src.inference.adaptive_controller import AdaptiveEvidenceController, AdaptiveDecision
from src.inference.cusum import SequentialCUSUMDetector


def test_adaptive_controller_attack_threshold_stopping():
    ctrl = AdaptiveEvidenceController(tau_attack=0.90, tau_benign=0.05, horizons=[5, 10, 15, 20, 25, 30])
    
    # Sequence with early attack confidence at K=10 (index 9)
    step_probs = np.full(30, 0.40)
    step_probs[9] = 0.95  # Step 10
    step_probs[10:] = 0.99

    dec = ctrl.evaluate_sequence_probabilities(step_probs)
    assert dec.predicted_label == 1
    assert dec.stopping_horizon == 10
    assert dec.is_early_stop is True
    assert "ATTACK_THRESHOLD_MET" in dec.stopping_reason


def test_adaptive_controller_benign_threshold_stopping():
    ctrl = AdaptiveEvidenceController(tau_attack=0.90, tau_benign=0.05, horizons=[5, 10, 15, 20, 25, 30])
    
    # Sequence with early benign confidence at K=5 (index 4)
    step_probs = np.full(30, 0.40)
    step_probs[4] = 0.01  # Step 5

    dec = ctrl.evaluate_sequence_probabilities(step_probs)
    assert dec.predicted_label == 0
    assert dec.stopping_horizon == 5
    assert dec.is_early_stop is True
    assert "BENIGN_THRESHOLD_MET" in dec.stopping_reason


def test_adaptive_controller_forced_stopping_at_kmax():
    ctrl = AdaptiveEvidenceController(tau_attack=0.95, tau_benign=0.01, horizons=[5, 10, 15, 20, 25, 30])
    
    # Ambiguous sequence staying in [0.40, 0.60] until K=30
    step_probs = np.full(30, 0.55)

    dec = ctrl.evaluate_sequence_probabilities(step_probs)
    assert dec.stopping_horizon == 30
    assert dec.is_early_stop is False
    assert dec.predicted_label == 1  # 0.55 >= 0.5
    assert "FORCED_HORIZON" in dec.stopping_reason


def test_adaptive_controller_batch_evaluation():
    ctrl = AdaptiveEvidenceController(tau_attack=0.90, tau_benign=0.10)
    
    # 4 sample sequences
    probs = np.full((4, 30), 0.50)
    probs[0, 4] = 0.95   # Early Atk at K=5
    probs[1, 9] = 0.02   # Early Ben at K=10
    probs[2, :] = 0.55
    probs[2, 29] = 0.92  # Forced Atk at K=30
    probs[3, :] = 0.40   # Forced Ben at K=30

    labels = np.array([1, 0, 1, 0])
    decisions, res = ctrl.evaluate_batch(probs, labels)

    assert len(decisions) == 4
    assert res.accuracy == 100.0
    assert res.f1_score == 1.0
    assert res.mean_stopping_horizon == (5 + 10 + 30 + 30) / 4


def test_cusum_detector_attack_alarm():
    detector = SequentialCUSUMDetector(h_attack=4.0, h_benign=4.0, min_horizon=5, max_horizon=30)
    
    # Strong sustained attack evidence starting from step 1 (p=0.99 -> log-odds = 4.60 >= 4.0 at K=5)
    step_probs = np.full(30, 0.99)
    dec = detector.evaluate_sequence(step_probs)

    assert dec.predicted_label == 1
    assert dec.stopping_horizon == 5
    assert dec.is_early_stop is True
    assert "CUSUM_ATTACK_ALARM" in dec.stopping_reason


def test_cusum_detector_benign_alarm():
    detector = SequentialCUSUMDetector(h_attack=4.0, h_benign=4.0, min_horizon=5, max_horizon=30)
    
    # Strong sustained benign evidence starting from step 1
    step_probs = np.full(30, 0.01)
    dec = detector.evaluate_sequence(step_probs)

    assert dec.predicted_label == 0
    assert dec.stopping_horizon == 5
    assert dec.is_early_stop is True
    assert "CUSUM_BENIGN_ALARM" in dec.stopping_reason


def test_adaptive_calibration_from_validation():
    # Synthetic validation step probabilities
    np.random.seed(42)
    val_atk = np.random.uniform(0.85, 0.99, (50, 30))
    val_ben = np.random.uniform(0.001, 0.05, (50, 30))
    val_probs = np.vstack([val_atk, val_ben])
    val_labels = np.array([1] * 50 + [0] * 50)

    calibrated_ctrl = AdaptiveEvidenceController.calibrate_from_validation(
        val_step_probabilities=val_probs,
        val_labels=val_labels,
        target_max_fpr=0.01,
        target_min_recall=0.98,
    )

    assert calibrated_ctrl.tau_attack >= 0.80
    assert calibrated_ctrl.tau_benign <= 0.15
