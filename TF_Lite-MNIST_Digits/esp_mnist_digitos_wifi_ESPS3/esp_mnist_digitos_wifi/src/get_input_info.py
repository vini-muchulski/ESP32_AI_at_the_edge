import json
from pathlib import Path

import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parent / "model_simple_int8.tflite"
OUTPUT_PATH = Path(__file__).resolve().parent / "input_info.json"


def read_input_info(model_path: Path) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    details = interpreter.get_input_details()[0]
    shape = [int(x) for x in details["shape"]]
    quant_scale, quant_zero_point = details["quantization"]

    return {
        "model_path": str(model_path),
        "shape": shape,
        "size": [shape[1], shape[2]],
        "dtype": details["dtype"].__name__,
        "scale": float(quant_scale),
        "zero_point": int(quant_zero_point),
    }


def main() -> None:
    model_path = MODEL_PATH.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    info = read_input_info(model_path)
    print(json.dumps(info, indent=2))

    OUTPUT_PATH.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Saved input metadata to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
