"""Export the unchanged champion, including exact decoding/padding, for TF Serving."""

import argparse
import hashlib
import json
from pathlib import Path

from visionguard.config import PROJECT_ROOT
from visionguard.preprocessing import decode_and_resize
from visionguard.runtime import configure_runtime, tf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/champion")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models/serving/1")
    args = parser.parse_args()
    configure_runtime()
    manifest = json.loads((args.model_dir / "manifest.json").read_text())
    model_path = args.model_dir / "model.keras"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != manifest["model_sha256"]:
        raise ValueError("Champion checksum mismatch")
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing SavedModel export")

    model = tf.keras.models.load_model(model_path, compile=False)

    class ServingModel(tf.Module):
        def __init__(self):
            super().__init__()
            # Track only resources used by the inference graph, not Keras's
            # cached training signatures and augmentation seed generators.
            self.model_weights = [variable.value for variable in model.variables]

        @tf.function(input_signature=[tf.TensorSpec([None], tf.string, name="image_bytes")])
        def serve(self, image_bytes):
            size = manifest["image_size"]
            pixels = tf.map_fn(
                lambda encoded: decode_and_resize(encoded, size),
                image_bytes,
                fn_output_signature=tf.TensorSpec((size, size, 3), tf.float32),
            )
            return {"probability_ai": model(pixels, training=False)}

    module = ServingModel()
    tf.saved_model.save(module, str(args.output), signatures={"serving_default": module.serve})
    (args.output.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"SavedModel exported to {args.output}; original Keras model remains intact.")


if __name__ == "__main__":
    main()
