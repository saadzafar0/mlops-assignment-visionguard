"""Verify serving-compatible preprocessing, serialisation and evaluation policy."""

import numpy as np

from visionguard.evaluate import select_by_validation
from visionguard.metrics import binary_metrics, choose_threshold
from visionguard.models import build_baseline, build_transfer, unfreeze_last_block
from visionguard.pipeline import decode_and_resize
from visionguard.runtime import configure_runtime, tf

configure_runtime()


def test_decode_preserves_aspect_ratio_and_converts_grayscale():
    encoded = tf.io.encode_png(tf.ones((10, 20, 1), dtype=tf.uint8) * 200)
    pixels = decode_and_resize(encoded, 32).numpy()
    assert pixels.shape == (32, 32, 3)
    np.testing.assert_allclose(pixels[0], 0)
    assert pixels[16].mean() > 190


def test_mobile_preprocessing_and_last_block_only():
    model = build_transfer(image_size=32, weights=None, augment=False)
    preprocess = tf.keras.Model(model.input, model.get_layer("preprocessing").output)
    example = tf.ones((1, 32, 32, 3)) * 127.5
    np.testing.assert_allclose(preprocess(example), 0, atol=1e-6)
    names = unfreeze_last_block(model, 2)
    base = model.get_layer("backbone")
    assert len(names) == 2
    assert all(name.startswith("block_16_") or name == "Conv_1" for name in names)
    assert all(
        not layer.trainable
        for layer in base.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    )
    assert {layer.name for layer in base.layers if layer.trainable} == set(names)


def test_baseline_roundtrip_and_augmentation_is_disabled_at_inference(tmp_path):
    model = build_baseline(image_size=32)
    example = tf.ones((2, 32, 32, 3)) * 150
    expected = model(example, training=False).numpy()
    np.testing.assert_allclose(expected, model(example, training=False).numpy())
    path = tmp_path / "model.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)
    np.testing.assert_allclose(expected, restored(example, training=False).numpy(), atol=1e-6)
    assert expected.shape == (2, 1)


def test_roc_auc_uses_probabilities_and_threshold_cost_penalises_false_accusations():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.55, 0.9])
    result = choose_threshold(labels, probabilities, false_positive_cost=3)
    assert result["threshold"] > 0.6
    assert result["validation_cost"] == 1
    metrics = binary_metrics(labels, probabilities, result["threshold"])
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 1
    assert metrics["roc_auc"] == 0.75


def test_final_selection_ignores_test_performance():
    validation_winner = {
        "validation": {"roc_auc": 0.9},
        "threshold_selection": {"validation_cost": 20},
        "test": {"accuracy": 0.1},
    }
    test_winner = {
        "validation": {"roc_auc": 0.7},
        "threshold_selection": {"validation_cost": 30},
        "test": {"accuracy": 0.99},
    }
    assert select_by_validation([test_winner, validation_winner]) is validation_winner
