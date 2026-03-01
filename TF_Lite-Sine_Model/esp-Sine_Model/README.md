# ESP32 Sine API (TensorFlow Lite Micro)

This project runs a TensorFlow Lite Micro sine model on an ESP32-S3 and exposes an HTTP API to predict `sin(angle)`.

## Project Files

- Firmware: `src/main.cpp`
- TFLite model: `src/model_sine.tflite`
- Generated model header: `src/model_sine_float32.h`
- Header generation commands: `src/comands.txt`
- Python API test client: `src/test_infer_web.py`
- PlatformIO configuration: `platformio.ini`

## Requirements

- ESP32-S3 board (`esp32-s3-devkitc-1`)
- VS Code + PlatformIO extension (or PlatformIO Core CLI)
- Python 3.10+
- Python package: `requests`

## Setup

1. Open the project in VS Code.
2. Connect the ESP32 board over USB.
3. Set Wi-Fi credentials in `src/main.cpp`:
   - `const char* ssid = "...";`
   - `const char* password = "...";`

## Generate `model_sine_float32.h`

`main.cpp` includes `src/model_sine_float32.h`. Generate it from `src/model_sine.tflite` before building.

Option 1: run commands from `src/comands.txt`:

```bash
cd src
xxd -i model_sine.tflite > model_sine_float32.h
sed -i '1i #include <pgmspace.h>\n' model_sine_float32.h
sed -i 's/\[\] =/[] PROGMEM =/' model_sine_float32.h
```

Option 2: copy and run the same commands manually.

## Build, Upload, Monitor

```bash
platformio run
platformio run -t upload
platformio device monitor -b 115200
```

## HTTP API

The firmware starts an HTTP server on port `80`.

- `GET /sine?angle=VALUE`
  - Input: angle in degrees
  - Output JSON:
    - `angle_degrees` (float)
    - `sine` (float)
- `GET /`
  - Simple usage help page

Example:

```text
http://<ESP32_IP>/sine?angle=30
```

## Python Test Client

Update `ESP32_IP` in `src/test_infer_web.py`, then run:

```bash
cd src
python3 test_infer_web.py
```
