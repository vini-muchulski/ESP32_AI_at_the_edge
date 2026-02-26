#include <Arduino.h>
#include <WiFi.h>
#include <WiFiServer.h>
#include <WiFiClient.h>
#include <cmath>
#include <climits>

// Check whether headers exist
#ifdef __has_include
  #if __has_include("model_simple_int8.h")
    #include "model_simple_int8.h"
    #define HAS_MODEL_DATA
  #endif
  #if __has_include("image_data.h")
    #include "image_data.h"
    #define HAS_IMAGE_DATA
  #endif
#endif

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// WiFi settings are injected from .env via PlatformIO extra script.
#ifndef WIFI_SSID
#define WIFI_SSID "WIFI_SSID"
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "WIFI_PASSWORD"
#endif

const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;
const int serverPort = 80;

// WiFi server
WiFiServer server(serverPort);

// Model data from external files (if available)
#ifdef HAS_MODEL_DATA
extern const unsigned char model_simple_int8_tflite[];
extern const unsigned int model_simple_int8_tflite_len;
#endif

// Model management structure
struct InferenceModel {
    tflite::ErrorReporter* error_reporter;
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input_tensor;
    TfLiteTensor* output_tensor;
    uint8_t* tensor_arena;
    uint8_t* model_buffer;
    bool initialized;
    
    // MobileNetV2 224x224x3 INT8 needs a larger arena on ESP32-S3.
    static constexpr int kTensorArenaSize = 3 * 1024 * 1024;
    // Keep model in Flash by default to preserve PSRAM for tensors.
    static constexpr bool kCopyModelToPSRAM = false;
    static constexpr int kInputWidth = 224;
    static constexpr int kInputHeight = 224;
    static constexpr int kInputChannels = 3;
    static constexpr int kImageSize = kInputWidth * kInputHeight * kInputChannels;
    static constexpr int kMaxRequestBytes = 1200 * 1024;
};

// Global model instance
InferenceModel model_ctx = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, false};

// Inference result structure
struct InferenceResult {
    int predicted_class;
    float confidence;
    bool success;
    String error_message;
};

struct RequestTimings {
    uint32_t receive_ms;
    uint32_t parse_ms;
    uint32_t inference_ms;
    uint32_t total_ms;
    int content_length;
    int body_length;
};

// Function declarations
void cleanup_model();
bool connect_wifi();
void handle_client();
String parse_json_array(const String& json_data, int16_t* image_array);
void write_input_tensor(const int16_t* image_data);
String create_json_response(const InferenceResult& result, const RequestTimings* timings = nullptr);

// Connect to WiFi
bool connect_wifi() {
    Serial.println("=== Connecting to WiFi ===");
    Serial.printf("SSID: %s\n", ssid);
    
    WiFi.begin(ssid, password);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(1000);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi connected!");
        Serial.printf("IP: %s\n", WiFi.localIP().toString().c_str());
        Serial.printf("Port: %d\n", serverPort);
        return true;
    } else {
        Serial.println("\nWiFi connection failed!");
        return false;
    }
}

// Memory cleanup
void cleanup_model() {
    if (model_ctx.model_buffer) {
        free(model_ctx.model_buffer);
        model_ctx.model_buffer = nullptr;
    }
    if (model_ctx.tensor_arena) {
        free(model_ctx.tensor_arena);
        model_ctx.tensor_arena = nullptr;
    }
    model_ctx.initialized = false;
}

// Allocate memory, preferring PSRAM
void* allocate_memory(size_t size) {
    void* ptr = heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (ptr == nullptr) {
        ptr = malloc(size);
    }
    return ptr;
}

// Load model
bool load_model() {
#ifndef HAS_MODEL_DATA
    Serial.println("ERROR: model header not found (expected model_simple_int8.h)");
    return false;
#endif

    Serial.println("[1] Loading model...");
    
    if (InferenceModel::kCopyModelToPSRAM) {
        model_ctx.model_buffer = static_cast<uint8_t*>(
            allocate_memory(model_simple_int8_tflite_len));

        if (model_ctx.model_buffer != nullptr) {
            Serial.printf("Copying model (%d bytes) into memory...\n", model_simple_int8_tflite_len);
            memcpy(model_ctx.model_buffer, model_simple_int8_tflite, model_simple_int8_tflite_len);
            model_ctx.model = tflite::GetModel(model_ctx.model_buffer);
        } else {
            Serial.println("Model copy to PSRAM failed, using Flash...");
            model_ctx.model = tflite::GetModel(model_simple_int8_tflite);
        }
    } else {
        Serial.println("Using model directly from Flash...");
        model_ctx.model = tflite::GetModel(model_simple_int8_tflite);
    }
    
    if (model_ctx.model == nullptr) {
        Serial.println("ERROR: Failed to load model");
        return false;
    }
    
    if (model_ctx.model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("ERROR: Incompatible version: %d vs %d\n",
                     model_ctx.model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    
    Serial.println("Model loaded successfully");
    return true;
}

// Initialize interpreter
bool initialize_interpreter() {
    Serial.println("[2] Initializing interpreter...");
    
    // Allocate tensor arena
    model_ctx.tensor_arena = static_cast<uint8_t*>(
        allocate_memory(InferenceModel::kTensorArenaSize));
    
    if (model_ctx.tensor_arena == nullptr) {
        Serial.printf("ERROR: Allocation of %d bytes failed\n", InferenceModel::kTensorArenaSize);
        return false;
    }
    
    // Configure op resolver
    static tflite::MicroMutableOpResolver<12> op_resolver;
    op_resolver.AddConv2D();
    op_resolver.AddDepthwiseConv2D();
    op_resolver.AddMaxPool2D();
    op_resolver.AddReshape();
    op_resolver.AddFullyConnected();
    op_resolver.AddSoftmax();
    op_resolver.AddQuantize();
    op_resolver.AddDequantize();
    op_resolver.AddMean();
    op_resolver.AddMul();
    op_resolver.AddAdd();
    
    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        model_ctx.model, op_resolver, model_ctx.tensor_arena, InferenceModel::kTensorArenaSize);
    model_ctx.interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = model_ctx.interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.printf("ERROR: AllocateTensors failed (code: %d)\n", allocate_status);
        return false;
    }
    
    // Get tensor pointers
    model_ctx.input_tensor = model_ctx.interpreter->input(0);
    model_ctx.output_tensor = model_ctx.interpreter->output(0);
    
    if (model_ctx.input_tensor == nullptr || model_ctx.output_tensor == nullptr) {
        Serial.println("ERROR: Null tensor pointers");
        return false;
    }
    
    Serial.printf("Arena used: %d/%d bytes\n", 
                  model_ctx.interpreter->arena_used_bytes(), InferenceModel::kTensorArenaSize);
    Serial.println("Interpreter initialized successfully");
    return true;
}

// Initialize full model pipeline
bool initialize_model() {
    Serial.println("=== Initializing Model ===");
    
    // Initialize error reporter
    static tflite::MicroErrorReporter micro_error_reporter;
    model_ctx.error_reporter = &micro_error_reporter;
    
    if (!load_model()) {
        return false;
    };
    
    if (!initialize_interpreter()) {
        cleanup_model();
        return false;
    }
    
    model_ctx.initialized = true;
    Serial.println("=== Model initialized successfully ===\n");
    return true;
}

void write_input_tensor(const int16_t* image_data) {
    if (model_ctx.input_tensor->type == kTfLiteUInt8) {
        for (int i = 0; i < InferenceModel::kImageSize; ++i) {
            int32_t value = image_data[i];
            value = max(0, min(255, value));
            model_ctx.input_tensor->data.uint8[i] = static_cast<uint8_t>(value);
        }
        return;
    }

    for (int i = 0; i < InferenceModel::kImageSize; ++i) {
        int32_t value = image_data[i];
        value = max(-128, min(127, value));
        model_ctx.input_tensor->data.int8[i] = static_cast<int8_t>(value);
    }
}

// Run inference
InferenceResult run_inference(const int16_t* image_data) {
    InferenceResult result = {-1, 0.0f, false, ""};
    
    if (!model_ctx.initialized) {
        result.error_message = "Model not initialized";
        Serial.println("ERROR: Model not initialized");
        return result;
    }
    
    // Input is already quantized on the client side.
    write_input_tensor(image_data);
    
    // Run inference
    TfLiteStatus invoke_status = model_ctx.interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        result.error_message = "Inference execution failed";
        Serial.printf("ERROR: Invoke failed (code: %d)\n", invoke_status);
        return result;
    }
    
    // Analyze result
    int best_index = 0;
    float max_score = -INFINITY;
    const int output_size = model_ctx.output_tensor->dims->data[1];

    const float output_scale = model_ctx.output_tensor->params.scale;
    const int32_t output_zero_point = model_ctx.output_tensor->params.zero_point;

    for (int i = 0; i < output_size; ++i) {
        float dequantized_score = 0.0f;
        if (model_ctx.output_tensor->type == kTfLiteInt8) {
            dequantized_score =
                (static_cast<float>(model_ctx.output_tensor->data.int8[i]) - output_zero_point) * output_scale;
        } else if (model_ctx.output_tensor->type == kTfLiteUInt8) {
            dequantized_score =
                (static_cast<float>(model_ctx.output_tensor->data.uint8[i]) - output_zero_point) * output_scale;
        } else {
            dequantized_score = model_ctx.output_tensor->data.f[i];
        }

        if (dequantized_score > max_score) {
            max_score = dequantized_score;
            best_index = i;
        }
    }

    result.predicted_class = best_index;
    result.confidence = max_score;
    result.success = true;
    
    return result;
}


// Build JSON response
String create_json_response(const InferenceResult& result, const RequestTimings* timings) {
    String response = "{\n";
    response += "  \"success\": " + String(result.success ? "true" : "false") + ",\n";
    response += "  \"predicted_class\": " + String(result.predicted_class) + ",\n";
    response += "  \"confidence\": " + String(result.confidence, 6) + ",\n";
    response += "  \"error_message\": \"" + result.error_message + "\",\n";
    response += "  \"heap_free\": " + String(esp_get_free_heap_size()) + ",\n";
    response += "  \"model_initialized\": " + String(model_ctx.initialized ? "true" : "false");
    if (timings != nullptr) {
        response += ",\n  \"receive_ms\": " + String(timings->receive_ms);
        response += ",\n  \"parse_ms\": " + String(timings->parse_ms);
        response += ",\n  \"inference_ms\": " + String(timings->inference_ms);
        response += ",\n  \"total_ms\": " + String(timings->total_ms);
        response += ",\n  \"content_length\": " + String(timings->content_length);
        response += ",\n  \"body_length\": " + String(timings->body_length) + "\n";
    } else {
        response += "\n";
    }
    response += "}";
    return response;
}

// Handle HTTP clients - corrected version
void handle_client() {
    WiFiClient client = server.available();
    if (!client) return;
    
    Serial.println("=== Client connected ===");
    
    // Configure client timeout
    client.setTimeout(15000);
    
    String request = "";
    String headers = "";
    String body = "";
    bool reading_body = false;
    int content_length = 0;
    bool body_complete = true;
    RequestTimings timings = {0, 0, 0, 0, 0, 0};
    const uint32_t request_start_ms = millis();
    
    // Read HTTP request with timeout
    unsigned long start_time = millis();
    const unsigned long timeout_ms = 10000; // 10 seconds
    
    while (client.connected() && (millis() - start_time < timeout_ms)) {
        if (!client.available()) {
            delay(1);
            continue;
        }
        
        String line = client.readStringUntil('\n');
        line.trim();
        
        if (!reading_body) {
            if (line.length() == 0) {
                reading_body = true;
                break; // Stop reading headers
            }
            
            if (request.length() == 0) {
                request = line;
            }
            
            headers += line + "\n";
            
            if (line.startsWith("Content-Length:")) {
                content_length = line.substring(15).toInt();
            }
        }
    }
    
    // Read body if POST has valid Content-Length
    const uint32_t body_read_start_ms = millis();
    if (request.startsWith("POST /predict") &&
        content_length > 0 &&
        content_length < InferenceModel::kMaxRequestBytes) {
        body.reserve(content_length + 100);
        
        unsigned long last_progress_ms = millis();
        const unsigned long body_timeout_ms = 30000; // timeout without read progress

        while (body.length() < content_length && client.connected()) {
            int available_bytes = client.available();
            if (available_bytes > 0) {
                int remaining = content_length - body.length();
                int to_read = min(available_bytes, remaining);
                while (to_read-- > 0) {
                    body += static_cast<char>(client.read());
                }
                last_progress_ms = millis();
            } else if ((millis() - last_progress_ms) > body_timeout_ms) {
                body_complete = false;
                break;
            } else {
                delay(1);
            }
        }
        if (body.length() != content_length) {
            body_complete = false;
        }
    }
    timings.receive_ms = millis() - body_read_start_ms;
    timings.content_length = content_length;
    timings.body_length = body.length();
    
    Serial.println("Request: " + request);
    Serial.println("Content-Length: " + String(content_length));
    Serial.println("Body length: " + String(body.length()));
    
    String response_body = "";
    String content_type = "text/html";
    
    // Process request
    if (request.startsWith("POST /predict")) {
        content_type = "application/json";

        if (content_length <= 0) {
            InferenceResult bad_request;
            bad_request.success = false;
            bad_request.error_message = "Missing or invalid Content-Length";
            bad_request.predicted_class = -1;
            bad_request.confidence = 0.0f;
            timings.total_ms = millis() - request_start_ms;
            response_body = create_json_response(bad_request, &timings);
            goto send_response;
        }

        if (content_length >= InferenceModel::kMaxRequestBytes) {
            InferenceResult too_large;
            too_large.success = false;
            too_large.error_message = "Payload too large";
            too_large.predicted_class = -1;
            too_large.confidence = 0.0f;
            timings.total_ms = millis() - request_start_ms;
            response_body = create_json_response(too_large, &timings);
            goto send_response;
        }

        if (!body_complete) {
            InferenceResult incomplete;
            incomplete.success = false;
            incomplete.error_message =
                "Incomplete request body: expected " + String(content_length) +
                ", received " + String(body.length());
            incomplete.predicted_class = -1;
            incomplete.confidence = 0.0f;
            timings.total_ms = millis() - request_start_ms;
            response_body = create_json_response(incomplete, &timings);
            goto send_response;
        }
        
        // Process inference
        int16_t* image_data = static_cast<int16_t*>(allocate_memory(sizeof(int16_t) * InferenceModel::kImageSize));
        if (image_data == nullptr) {
            InferenceResult oom_result;
            oom_result.success = false;
            oom_result.error_message = "Failed to allocate image buffer";
            oom_result.predicted_class = -1;
            oom_result.confidence = 0.0f;
            timings.total_ms = millis() - request_start_ms;
            response_body = create_json_response(oom_result, &timings);
            goto send_response;
        }

        const uint32_t parse_start_ms = millis();
        String parse_error = parse_json_array(body, image_data);
        timings.parse_ms = millis() - parse_start_ms;
        
        InferenceResult result;
        if (parse_error.length() > 0) {
            result.success = false;
            result.error_message = parse_error;
            result.predicted_class = -1;
            result.confidence = 0.0f;
            Serial.println("Parsing error: " + parse_error);
        } else {
            Serial.println("=== RUNNING INFERENCE ===");
            const uint32_t inference_start_ms = millis();
            result = run_inference(image_data);
            timings.inference_ms = millis() - inference_start_ms;
            
            if (result.success) {
                Serial.println("=== RESULT ===");
                Serial.printf("Prediction class: %d\n", result.predicted_class);
                Serial.printf("Confidence: %.6f\n", result.confidence);
                Serial.println("==================");
            } else {
                Serial.println("Inference failed: " + result.error_message);
            }
        }
        free(image_data);
        timings.total_ms = millis() - request_start_ms;
        Serial.printf(
            "Timing(ms): receive=%lu parse=%lu inference=%lu total=%lu\n",
            static_cast<unsigned long>(timings.receive_ms),
            static_cast<unsigned long>(timings.parse_ms),
            static_cast<unsigned long>(timings.inference_ms),
            static_cast<unsigned long>(timings.total_ms)
        );
        response_body = create_json_response(result, &timings);
        
    } else if (request.startsWith("GET /status")) {
        content_type = "application/json";
        
        // System status
        InferenceResult status_result;
        status_result.success = model_ctx.initialized;
        status_result.predicted_class = -1;
        status_result.confidence = 0.0f;
        status_result.error_message = model_ctx.initialized ? "" : "Model not initialized";
        
        response_body = create_json_response(status_result);
        
    } else {
        // Help page
        response_body = "<!DOCTYPE html><html><body>";
        response_body += "<h1>CIFAR-10 MobileNetV2 API</h1>";
        response_body += "<h2>Endpoints:</h2>";
        response_body += "<p><b>POST /predict</b> - Run inference</p>";
        response_body += "<p>Body JSON: {\"q_pixels\": [array of 150528 quantized values]}</p>";
        response_body += "<p><b>GET /status</b> - System status</p>";
        response_body += "<p>IP: " + WiFi.localIP().toString() + "</p>";
        response_body += "</body></html>";
    }

send_response:
    // Send HTTP response with proper headers
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: " + content_type);
    client.println("Access-Control-Allow-Origin: *");
    client.println("Access-Control-Allow-Methods: GET, POST, OPTIONS");
    client.println("Access-Control-Allow-Headers: Content-Type");
    client.println("Connection: close");
    client.println("Content-Length: " + String(response_body.length()));
    client.println(); // Important blank line
    client.print(response_body);
    client.flush(); // Ensure all data is sent
    
    // Short pause before closing
    delay(100);
    
    client.stop();
    Serial.println("Client disconnected\n");
}

// Parse JSON array - more robust version
String parse_json_array(const String& json_data, int16_t* image_array) {
    // Find the "q_pixels" array
    int start_index = json_data.indexOf("\"q_pixels\":");
    if (start_index == -1) {
        return "Field 'q_pixels' not found";
    }
    
    start_index = json_data.indexOf('[', start_index);
    if (start_index == -1) {
        return "Pixels array not found";
    }
    
    int end_index = json_data.indexOf(']', start_index);
    if (end_index == -1) {
        return "Array end not found";
    }
    
    // Parse numeric values in one pass to keep parsing linear for large payloads.
    int pixel_count = 0;
    bool in_number = false;
    int sign = 1;
    long value = 0;

    for (int i = start_index + 1; i < end_index; ++i) {
        char c = json_data.charAt(i);

        if (c >= '0' && c <= '9') {
            if (!in_number) {
                in_number = true;
                sign = 1;
                value = 0;
            }
            value = value * 10 + (c - '0');
            continue;
        }

        if (c == '-' && !in_number) {
            in_number = true;
            sign = -1;
            value = 0;
            continue;
        }

        if (c == '+' && !in_number) {
            in_number = true;
            sign = 1;
            value = 0;
            continue;
        }

        if (c == ',' || c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (in_number) {
                if (pixel_count >= InferenceModel::kImageSize) {
                    return "Input has more values than expected";
                }
                long final_value = sign * value;
                if (final_value < SHRT_MIN || final_value > SHRT_MAX) {
                    return "Value out of int16 range at index " + String(pixel_count);
                }
                image_array[pixel_count++] = static_cast<int16_t>(final_value);
                in_number = false;
                sign = 1;
                value = 0;
            }
            continue;
        }

        return "Invalid character in q_pixels payload";
    }

    if (in_number) {
        if (pixel_count >= InferenceModel::kImageSize) {
            return "Input has more values than expected";
        }
        long final_value = sign * value;
        if (final_value < SHRT_MIN || final_value > SHRT_MAX) {
            return "Value out of int16 range at index " + String(pixel_count);
        }
        image_array[pixel_count++] = static_cast<int16_t>(final_value);
    }
    
    if (pixel_count != InferenceModel::kImageSize) {
        return "Array must contain exactly " + String(InferenceModel::kImageSize) +
               " values (224x224x3), received: " + String(pixel_count);
    }
    
    return ""; // Success
}


void setup() {
    Serial.begin(115200);
    delay(2000);
    
    Serial.println("\n=== CIFAR-10 MobileNetV2 TensorFlow Lite WiFi API ===");
    Serial.printf("Initial free heap: %d bytes\n", esp_get_free_heap_size());
    Serial.printf("Available PSRAM: %d bytes\n", ESP.getPsramSize());
    
    // Connect WiFi
    if (!connect_wifi()) {
        Serial.println("WiFi connection failed - restarting...");
        ESP.restart();
    }
    
    // Initialize model
    if (!initialize_model()) {
        Serial.println("Model initialization failed!");
        return;
    }
    
    // Start server
    server.begin();
    Serial.println("\n=== HTTP server started ===");
    Serial.println("Available endpoints:");
    Serial.println("POST /predict - Run inference");
    Serial.println("GET /status - System status");
    Serial.println("GET / - Help page");
    Serial.println("============================\n");
}

void loop() {
    // Check WiFi connection
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi disconnected - trying to reconnect...");
        if (!connect_wifi()) {
            delay(5000);
            return;
        }
    }
    
    // Handle clients
    handle_client();
    
    delay(10); // Short delay to avoid overload
}
