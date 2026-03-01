/*
 * TensorFlow Lite Micro inference on ESP32-C3
 * Sine model compatible with FLOAT32 or INT8/UINT8
 * Wi-Fi API for sine calculation
 */

#include <WiFi.h>
#include <WebServer.h>

#include <Arduino.h>
#include <math.h>  // for M_PI

const char* ssid = "ssid";
const char* password = "password";

// Web server on port 80
WebServer server(80);

#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Converted model include
#include "model_sine_float32.h"

// TFLM memory arena
constexpr int kTensorArenaSize = 12 * 1024;  // 12 kB
static uint8_t tensor_arena[kTensorArenaSize];

// Global variables
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = nullptr;
  tflite::AllOpsResolver resolver;  // all operators
  tflite::MicroInterpreter* interpreter = nullptr;

  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Quantization parameters (valid only for INT8/UINT8 models)
  float in_scale = 1.0f;
  int32_t in_zp = 0;
  float out_scale = 1.0f;
  int32_t out_zp = 0;
  bool is_quant = false;
}

float inferSine(float x) {
  // 1) Write input tensor
  if (is_quant) {
    int32_t q_in = static_cast<int32_t>(roundf(x / in_scale) + in_zp);

    // Clamp to numeric type range
    if (input->type == kTfLiteInt8) {
      q_in = max(-128, min(127, q_in));
      input->data.int8[0] = static_cast<int8_t>(q_in);
    } else {  // UINT8
      q_in = max(0, min(255, q_in));
      input->data.uint8[0] = static_cast<uint8_t>(q_in);
    }
  } else {
    input->data.f[0] = x;  // original float32 model
  }

  // 2) Invoke
  if (interpreter->Invoke() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke() failed.");
    return NAN;
  }

  // 3) Read output
  float y;
  if (is_quant) {
    int32_t q_out = (output->type == kTfLiteInt8)
                        ? output->data.int8[0]
                        : output->data.uint8[0];
    y = (q_out - out_zp) * out_scale;
  } else {
    y = output->data.f[0];
  }

  return y;
}

// API handler for sine calculation
void handleSine() {
  if (!server.hasArg("angle")) {
    server.send(400, "application/json", "{\"error\":\"Missing 'angle' query parameter\"}");
    return;
  }

  float angle_degrees = server.arg("angle").toFloat();
  float angle_rad = (angle_degrees * M_PI) / 180.0f;
  float sine_result = inferSine(angle_rad);

  if (isnan(sine_result)) {
    server.send(500, "application/json", "{\"error\":\"Model inference failed\"}");
    Serial.printf("Inference error for angle: %.2f degrees\n", angle_degrees);
    return;
  }

  Serial.printf("sin(%.2f deg) = %.6f\n", angle_degrees, sine_result);

  String response_json = "{";
  response_json += "\"angle_degrees\":" + String(angle_degrees, 2) + ",";
  response_json += "\"sine\":" + String(sine_result, 6);
  response_json += "}";

  server.send(200, "application/json", response_json);
}

// API root help page
void handleRoot() {
  String html = "<html><body>";
  html += "<h1>ESP32 Sine API</h1>";
  html += "<p>To calculate sine, use:</p>";
  html += "<p><strong>GET /sine?angle=VALUE</strong></p>";
  html += "<p>Example: <a href='/sine?angle=30'>/sine?angle=30</a></p>";
  html += "<p>Example: <a href='/sine?angle=45'>/sine?angle=45</a></p>";
  html += "<p>Example: <a href='/sine?angle=90'>/sine?angle=90</a></p>";
  html += "</body></html>";

  server.send(200, "text/html", html);
}

// 404 handler
void handleNotFound() {
  server.send(404, "application/json", "{\"error\":\"Endpoint not found\"}");
}

void runInitialInferenceTest() {
  Serial.println("Running initial inference test...");

  // Test angles (radians)
  constexpr float angles[] = { M_PI/3, M_PI/6, M_PI/4, M_PI/2, M_PI };
  constexpr int num_angles = sizeof(angles) / sizeof(angles[0]);

  for (int i = 0; i < num_angles; ++i) {
    float x = angles[i];
    float y = inferSine(x);

    if (!isnan(y)) {
      float angle_degrees = (x * 180.0f) / M_PI;
      Serial.printf("sin(%.2f deg) = %.6f\n", angle_degrees, y);
    } else {
      Serial.printf("Inference error for x=%f\n", x);
    }
  }
  Serial.println("----------------------------");
  Serial.println("Initial test completed.\n");
}

void setup() {
  Serial.begin(115200);
  delay(200);

  // 1) Load model from flash
  model = tflite::GetModel(model_sine_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model v%d != Schema v%d", model->version(), TFLITE_SCHEMA_VERSION);
    while (true);
  }

  // 2) Interpreter + tensor allocation
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed.");
    while (true);
  }

  // 3) Input/output tensor pointers
  input = interpreter->input(0);
  output = interpreter->output(0);

  // 4) Check if model is quantized
  is_quant = (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8);

  if (is_quant) {
    in_scale = input->params.scale;
    in_zp = input->params.zero_point;
    out_scale = output->params.scale;
    out_zp = output->params.zero_point;
  }

  // 5) Quick logs
  Serial.printf("INPUT  type=%d  scale=%f  zp=%d\n",
                input->type, in_scale, in_zp);
  Serial.printf("OUTPUT type=%d  scale=%f  zp=%d\n",
                output->type, out_scale, out_zp);
  Serial.println("TensorFlow Lite model loaded successfully.");

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.printf("Wi-Fi connected. IP: %s\n", WiFi.localIP().toString().c_str());

  // Configure API routes
  server.on("/", handleRoot);
  server.on("/sine", handleSine);
  server.onNotFound(handleNotFound);

  // Run initial inference test
  runInitialInferenceTest();

  // Start server
  server.begin();
  Serial.println("HTTP server started.");
  Serial.println("Use: GET /sine?angle=VALUE");
  Serial.println("Example: http://" + WiFi.localIP().toString() + "/sine?angle=30");
}

void loop() {
  // Process HTTP requests
  server.handleClient();

  // Small delay to avoid overload
  delay(2);
}
