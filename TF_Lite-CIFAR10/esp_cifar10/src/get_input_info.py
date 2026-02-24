import json
from pathlib import Path

import tensorflow as tf

base_dir = Path(__file__).resolve().parent
model_path = base_dir / "model_simple_int8.tflite"
output_path = base_dir / "input_info.json"

it = tf.lite.Interpreter(model_path=str(model_path))
it.allocate_tensors()
d = it.get_input_details()[0]

info = {
    "model": str(model_path.name),
    "size": [int(d["shape"][1]), int(d["shape"][2])],
    "scale": float(d["quantization"][0]),
    "zero_point": int(d["quantization"][1]),
    "dtype": d["dtype"].__name__,
}

print(info)
output_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
