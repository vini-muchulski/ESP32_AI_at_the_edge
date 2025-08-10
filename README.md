
# ESP32 AI at the Edge

A set of practical examples for running **AI on ESP32 microcontrollers**, using **TensorFlow Lite for Microcontrollers (TFLM)** and **ESP-DL**. The projects range from the classic *Hello World* (predicting the sine of an angle) to classifiers (MNIST and CIFAR-10) and on-device face detection.

## Repository Structure

```
ESP32_AI_at_the_edge/
├─ ESP-DL-face-detect/           # Face detection using ESP-DL
├─ TF_Lite-CIFAR10/              # CIFAR-10 classifier with TFLite Micro
├─ TF_Lite-Cifar10_MobileNetv2/  # CIFAR-10 with MobileNetV2 backbone (TFLM)
├─ TF_Lite-MNIST_Digits/         # MNIST digit classifier with TFLM
├─ TF_Lite-Sine_Model/           # Hello World: sine model (TFLM)
└─ .gitignore
```

> The names above are the exact repository directories. ([GitHub](https://github.com/vini-muchulski/ESP32_AI_at_the_edge))

## Prerequisites

*   **TensorFlow Lite for Microcontrollers** (for examples like *Hello World/sine*). ([Google AI for Developers](https://ai.google.dev/edge/litert/microcontrollers/get_started))
*   The TensorFlow Lite projects already include the TFLite library in the `libs` folder.
*   All code includes an option to use the ESP32's Wi-Fi to send data/images via an API to the model on the ESP32 for inference and to return the result. Check the `SSID` and `Password` variables within the code to be uploaded to the ESP32.
*   **ESP-IDF** - For the `ESP-DL-face-detect` project, you will use `idf.py` to configure, build, flash, and monitor. ([docs.espressif.com](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/tools/idf-py.html))

## Included Projects

### TF_Lite-Sine_Model (Hello World)

A minimal TFLM example: a model predicts sine values, useful for validating the toolchain and the *train → convert → deploy* workflow.

### TF_Lite-MNIST_Digits

A digit classifier (0–9) running on the ESP32 with TFLite. Ideal for testing a classification pipeline, quantization, and simple I/O. Use the Python code.

### TF_Lite-CIFAR10

Image classification into 10 classes with TFLM. Useful for studying lightweight preprocessing and classification on resource-constrained microcontrollers.

### TF_Lite-Cifar10_MobileNetv2

A variant of the CIFAR-10 project using the MobileNetV2 backbone, offering better accuracy with a moderate computational cost, optimized for the memory/compute constraints of ESP devices.

### ESP-DL-face-detect

Face detection using **ESP-DL**—a lightweight framework from Espressif for inference on the ESP32. To learn more: ([GitHub](https://github.com/espressif/esp-dl), [docs.espressif.com](https://docs.espressif.com/projects/esp-dl/en/latest/getting_started/readme.html))

## Hardware

*   I used the ESP32 for the Sine project.
*   I used the ESP32-S3 for the embedded CNN projects due to the presence of PSRAM in this type of microcontroller.

## 📄 License

Apache-2.0
