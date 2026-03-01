# ESP32 Sine Model API (TensorFlow Lite Micro)

This project runs a TensorFlow Lite Micro sine model on ESP32 and exposes an HTTP API to predict `sin(angle)`.

## What this project includes

- ESP32 firmware with TensorFlow Lite Micro: `src/main.cpp`
- Model source file: `src/model_sine.tflite`
- Generated embedded header: `src/modelo_seno_float32.h`
- Web inference test script: `src/teste_inferencia_web.py`
- Kaggle notebook (training/export reference): [tensorflow-lite-sine-prediction](https://www.kaggle.com/code/vinimuchulski/tensorflow-lite-sine-prediction)

## Requirements

- ESP32 board (`esp32doit-devkit-v1` in `platformio.ini`)
- VS Code + **PlatformIO IDE** extension
- Python 3.10+ (for test script)

## Quick Start (VS Code + PlatformIO)

1. Open this folder in VS Code.
2. Install the **PlatformIO IDE** extension.
3. Connect your ESP32 board via USB.
4. In `platformio.ini`, verify:
   - environment `[env:esp32doit-devkit-v1]`
   - serial monitor speed (`115200`)
5. Update Wi-Fi credentials in `src/main.cpp` (`ssid` and `password`).
6. Build and upload:
   - `PlatformIO: Build`
   - `PlatformIO: Upload`
7. Open serial monitor (`PlatformIO: Monitor`) at `115200` baud.

## Generate `modelo_seno_float32.h`

`main.cpp` includes `src/modelo_seno_float32.h`. If you only have `src/model_sine.tflite`, generate the header first:

```bash
cd src
xxd -i model_sine.tflite > modelo_seno_float32.h
sed -i 's/model_sine_tflite/modelo_seno_float32_tflite/g' modelo_seno_float32.h
sed -i '1i #include <pgmspace.h>\n' modelo_seno_float32.h
sed -i 's/\[\] =/[] PROGMEM =/' modelo_seno_float32.h
```

You can also run the same commands from `src/comands.txt`.

## Wi-Fi and API

Firmware starts an HTTP server on port `80`.

- `GET /seno?angulo=VALUE`
  - Input: angle in degrees
  - Output: JSON with angle and predicted sine value
- `GET /`
  - Simple help page with usage examples

Example:

```text
http://<ESP32_IP>/seno?angulo=30
```

## Python Usage

```bash
cd src
python teste_inferencia_web.py
```
