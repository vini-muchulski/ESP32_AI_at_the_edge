#include <Arduino.h>
#include <WiFi.h>
#include <WiFiServer.h>
#include <WiFiClient.h>
#include <cmath>
#include <climits>

// Check whether headers exist
#ifdef __has_include
  #if __has_include("mnist_model_data.h")
    #include "mnist_model_data.h"
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
extern unsigned char mnist_cnn_small_int8_tflite[];
extern unsigned int mnist_cnn_small_int8_tflite_len;
#endif

// Mock test data if image_data.h is missing
#ifndef HAS_IMAGE_DATA
const uint8_t mnist_sample[784] PROGMEM = {0}; // Blank image for testing
#endif

// Model management structure
struct MNISTModel {
    tflite::ErrorReporter* error_reporter;
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input_tensor;
    TfLiteTensor* output_tensor;
    uint8_t* tensor_arena;
    uint8_t* model_buffer;
    bool initialized;
    
    static constexpr int kTensorArenaSize = 120 * 1024;
    static constexpr int kImageSize = 28 * 28;
};

// Global model instance
MNISTModel mnist_model = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, false};

// Inference result structure
struct InferenceResult {
    int predicted_class;
    float confidence;
    bool success;
    String error_message;
};

// Function declarations
void cleanup_model();
bool connect_wifi();
void handle_client();
String parse_json_array(String json_data, int16_t* image_array);
void write_input_tensor(const int16_t* image_data);
String create_json_response(const InferenceResult& result);

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
    if (mnist_model.model_buffer) {
        free(mnist_model.model_buffer);
        mnist_model.model_buffer = nullptr;
    }
    if (mnist_model.tensor_arena) {
        free(mnist_model.tensor_arena);
        mnist_model.tensor_arena = nullptr;
    }
    mnist_model.initialized = false;
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
    Serial.println("ERROR: mnist_model_data.h not found!");
    return false;
#endif

    Serial.println("[1] Loading model...");
    
    // Try loading the model into PSRAM
    mnist_model.model_buffer = static_cast<uint8_t*>(
        allocate_memory(mnist_cnn_small_int8_tflite_len));
    
    if (mnist_model.model_buffer != nullptr) {
        Serial.printf("Copying model (%d bytes) into memory...\n", mnist_cnn_small_int8_tflite_len);
        memcpy(mnist_model.model_buffer, mnist_cnn_small_int8_tflite, mnist_cnn_small_int8_tflite_len);
        mnist_model.model = tflite::GetModel(mnist_model.model_buffer);
    } else {
        Serial.println("Using model directly from Flash...");
        mnist_model.model = tflite::GetModel(mnist_cnn_small_int8_tflite);
    }
    
    if (mnist_model.model == nullptr) {
        Serial.println("ERROR: Failed to load model");
        return false;
    }
    
    if (mnist_model.model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("ERROR: Incompatible version: %d vs %d\n",
                     mnist_model.model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    
    Serial.println("Model loaded successfully");
    return true;
}

// Initialize interpreter
bool initialize_interpreter() {
    Serial.println("[2] Initializing interpreter...");
    
    // Allocate tensor arena
    mnist_model.tensor_arena = static_cast<uint8_t*>(
        allocate_memory(MNISTModel::kTensorArenaSize));
    
    if (mnist_model.tensor_arena == nullptr) {
        Serial.printf("ERROR: Allocation of %d bytes failed\n", MNISTModel::kTensorArenaSize);
        return false;
    }
    
    // Configure op resolver
    static tflite::MicroMutableOpResolver<10> op_resolver;
    op_resolver.AddConv2D();
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
        mnist_model.model, op_resolver, mnist_model.tensor_arena, MNISTModel::kTensorArenaSize);
    mnist_model.interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = mnist_model.interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.printf("ERROR: AllocateTensors failed (code: %d)\n", allocate_status);
        return false;
    }
    
    // Get tensor pointers
    mnist_model.input_tensor = mnist_model.interpreter->input(0);
    mnist_model.output_tensor = mnist_model.interpreter->output(0);
    
    if (mnist_model.input_tensor == nullptr || mnist_model.output_tensor == nullptr) {
        Serial.println("ERROR: Null tensor pointers");
        return false;
    }
    
    Serial.printf("Arena used: %d/%d bytes\n", 
                  mnist_model.interpreter->arena_used_bytes(), MNISTModel::kTensorArenaSize);
    Serial.println("Interpreter initialized successfully");
    return true;
}

// Initialize full model pipeline
bool initialize_mnist_model() {
    Serial.println("=== Initializing MNIST Model ===");
    
    // Initialize error reporter
    static tflite::MicroErrorReporter micro_error_reporter;
    mnist_model.error_reporter = &micro_error_reporter;
    
    if (!load_model()) {
        return false;
    };
    
    if (!initialize_interpreter()) {
        cleanup_model();
        return false;
    }
    
    mnist_model.initialized = true;
    Serial.println("=== Model initialized successfully ===\n");
    return true;
}

void write_input_tensor(const int16_t* image_data) {
    if (mnist_model.input_tensor->type == kTfLiteUInt8) {
        for (int i = 0; i < MNISTModel::kImageSize; ++i) {
            int32_t value = image_data[i];
            value = max(0, min(255, value));
            mnist_model.input_tensor->data.uint8[i] = static_cast<uint8_t>(value);
        }
        return;
    }

    for (int i = 0; i < MNISTModel::kImageSize; ++i) {
        int32_t value = image_data[i];
        value = max(-128, min(127, value));
        mnist_model.input_tensor->data.int8[i] = static_cast<int8_t>(value);
    }
}

// Run inference
InferenceResult run_inference(const int16_t* image_data) {
    InferenceResult result = {-1, 0.0f, false, ""};
    
    if (!mnist_model.initialized) {
        result.error_message = "Model not initialized";
        Serial.println("ERROR: Model not initialized");
        return result;
    }
    
    // Input is already quantized on the client side.
    write_input_tensor(image_data);
    
    // Run inference
    TfLiteStatus invoke_status = mnist_model.interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        result.error_message = "Inference execution failed";
        Serial.printf("ERROR: Invoke failed (code: %d)\n", invoke_status);
        return result;
    }
    
    // Analyze result
    int best_index = 0;
    float max_score = -INFINITY;
    const int output_size = mnist_model.output_tensor->dims->data[1];

    const float output_scale = mnist_model.output_tensor->params.scale;
    const int32_t output_zero_point = mnist_model.output_tensor->params.zero_point;

    for (int i = 0; i < output_size; ++i) {
        float dequantized_score = 0.0f;
        if (mnist_model.output_tensor->type == kTfLiteInt8) {
            dequantized_score =
                (static_cast<float>(mnist_model.output_tensor->data.int8[i]) - output_zero_point) * output_scale;
        } else if (mnist_model.output_tensor->type == kTfLiteUInt8) {
            dequantized_score =
                (static_cast<float>(mnist_model.output_tensor->data.uint8[i]) - output_zero_point) * output_scale;
        } else {
            dequantized_score = mnist_model.output_tensor->data.f[i];
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
String create_json_response(const InferenceResult& result) {
    String response = "{\n";
    response += "  \"success\": " + String(result.success ? "true" : "false") + ",\n";
    response += "  \"predicted_class\": " + String(result.predicted_class) + ",\n";
    response += "  \"predicted_digit\": " + String(result.predicted_class) + ",\n";
    response += "  \"confidence\": " + String(result.confidence, 6) + ",\n";
    response += "  \"error_message\": \"" + result.error_message + "\",\n";
    response += "  \"heap_free\": " + String(esp_get_free_heap_size()) + ",\n";
    response += "  \"model_initialized\": " + String(mnist_model.initialized ? "true" : "false") + "\n";
    response += "}";
    return response;
}

// Handle HTTP clients - corrected version
void handle_client() {
    WiFiClient client = server.available();
    if (!client) return;
    
    Serial.println("=== Client connected ===");
    
    // Configure client timeout
    client.setTimeout(5000);
    
    String request = "";
    String headers = "";
    String body = "";
    bool reading_body = false;
    int content_length = 0;
    
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
    
    // Read body if POST has Content-Length
    if (content_length > 0 && content_length < 50000) { // Safety limit
        body.reserve(content_length + 100);
        
        unsigned long body_start = millis();
        while (body.length() < content_length && client.connected() && 
               (millis() - body_start < 5000)) {
            if (client.available()) {
                char c = client.read();
                body += c;
            } else {
                delay(1);
            }
        }
    }
    
    Serial.println("Request: " + request);
    Serial.println("Body length: " + String(body.length()));
    
    String response_body = "";
    String content_type = "text/html";
    
    // Process request
    if (request.startsWith("POST /predict")) {
        content_type = "application/json";
        
        // Process inference
        int16_t image_data[MNISTModel::kImageSize];
        String parse_error = parse_json_array(body, image_data);
        
        InferenceResult result;
        if (parse_error.length() > 0) {
            result.success = false;
            result.error_message = parse_error;
            result.predicted_class = -1;
            result.confidence = 0.0f;
            Serial.println("Parsing error: " + parse_error);
        } else {
            Serial.println("=== RUNNING INFERENCE ===");
            result = run_inference(image_data);
            
            if (result.success) {
                Serial.println("=== RESULT ===");
                Serial.printf("Prediction class: %d\n", result.predicted_class);
                Serial.printf("Confidence: %.6f\n", result.confidence);
                Serial.println("==================");
            } else {
                Serial.println("Inference failed: " + result.error_message);
            }
        }
        
        response_body = create_json_response(result);
        
    } else if (request.startsWith("GET /status")) {
        content_type = "application/json";
        
        // System status
        InferenceResult status_result;
        status_result.success = mnist_model.initialized;
        status_result.predicted_class = -1;
        status_result.confidence = 0.0f;
        status_result.error_message = mnist_model.initialized ? "" : "Model not initialized";
        
        response_body = create_json_response(status_result);
        
    } else {
        // Help page
        response_body = "<!DOCTYPE html><html><body>";
        response_body += "<h1>Fashion-MNIST API</h1>";
        response_body += "<h2>Endpoints:</h2>";
        response_body += "<p><b>POST /predict</b> - Run inference</p>";
        response_body += "<p>Body JSON: {\"q_pixels\": [array of 784 quantized values]}</p>";
        response_body += "<p><b>GET /status</b> - System status</p>";
        response_body += "<p>IP: " + WiFi.localIP().toString() + "</p>";
        response_body += "</body></html>";
    }
    
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
String parse_json_array(String json_data, int16_t* image_array) {
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
    
    String array_content = json_data.substring(start_index + 1, end_index);
    array_content.trim();
    
    // Parse values with improved error handling
    int pixel_count = 0;
    int current_pos = 0;
    
    while (current_pos < array_content.length() && pixel_count < MNISTModel::kImageSize) {
        // Skip whitespace
        while (current_pos < array_content.length() && 
               (array_content.charAt(current_pos) == ' ' || 
                array_content.charAt(current_pos) == '\t' ||
                array_content.charAt(current_pos) == '\n' ||
                array_content.charAt(current_pos) == '\r')) {
            current_pos++;
        }
        
        if (current_pos >= array_content.length()) break;
        
        // Find next comma or end
        int comma_pos = array_content.indexOf(',', current_pos);
        String value_str;
        
        if (comma_pos == -1) {
            value_str = array_content.substring(current_pos);
        } else {
            value_str = array_content.substring(current_pos, comma_pos);
        }
        
        value_str.trim();
        
        // Validate numeric token
        bool is_valid_number = true;
        for (int i = 0; i < value_str.length(); i++) {
            char c = value_str.charAt(i);
            if (!isdigit(c) && c != '-' && c != '+') {
                is_valid_number = false;
                break;
            }
        }
        
        if (!is_valid_number || value_str.length() == 0) {
            return "Invalid value at index " + String(pixel_count) + ": '" + value_str + "'";
        }
        
        int pixel_value = value_str.toInt();
        image_array[pixel_count] = static_cast<int16_t>(pixel_value);
        pixel_count++;
        
        if (comma_pos == -1) break;
        current_pos = comma_pos + 1;
    }
    
    if (pixel_count != MNISTModel::kImageSize) {
        return "Array must contain exactly " + String(MNISTModel::kImageSize) +
               " pixels (28x28), received: " + String(pixel_count);
    }
    
    return ""; // Success
}


void setup() {
    Serial.begin(115200);
    delay(2000);
    
    Serial.println("\n=== Fashion-MNIST TensorFlow Lite WiFi API ===");
    Serial.printf("Initial free heap: %d bytes\n", esp_get_free_heap_size());
    Serial.printf("Available PSRAM: %d bytes\n", ESP.getPsramSize());
    
    // Connect WiFi
    if (!connect_wifi()) {
        Serial.println("WiFi connection failed - restarting...");
        ESP.restart();
    }
    
    // Initialize model
    if (!initialize_mnist_model()) {
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
