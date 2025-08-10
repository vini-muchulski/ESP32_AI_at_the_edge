#include <Arduino.h>
#include <WiFi.h>
#include <WiFiServer.h>
#include <WiFiClient.h>
#include <cmath>
#include <climits>

#ifdef __has_include
  #if __has_include("cifar10_model_data.h")
    #include "cifar10_model_data.h"
    #define HAS_MODEL_DATA
  #endif
#endif

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

const char* ssid = "REDE WIFI";
const char* password = "PASSWORD";

const int serverPort = 80;

WiFiServer server(serverPort);

#ifdef HAS_MODEL_DATA
extern unsigned char cifar10_simple_int8_tflite[];
extern unsigned int cifar10_simple_int8_tflite_len;
#endif

struct CIFAR10Model {
    tflite::ErrorReporter* error_reporter;
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input_tensor;
    TfLiteTensor* output_tensor;
    uint8_t* tensor_arena;
    uint8_t* model_buffer;
    bool initialized;

    static constexpr int kTensorArenaSize = 150 * 1024;
    static constexpr int kImageSize = 32 * 32 * 3;
};

CIFAR10Model cifar10_model = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, false};

struct InferenceResult {
    int predicted_class;
    float confidence;
    bool success;
    String error_message;
};

void cleanup_model();
bool connect_wifi();
void handle_client();
String parse_json_array(String json_data, uint8_t* image_array);
String create_json_response(const InferenceResult& result);
bool initialize_cifar10_model();
InferenceResult run_inference(const uint8_t* image_data);

bool connect_wifi() {
    Serial.println("=== Conectando ao WiFi ===");
    Serial.printf("SSID: %s\n", ssid);

    WiFi.begin(ssid, password);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(1000);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi conectado!");
        Serial.printf("IP: %s\n", WiFi.localIP().toString().c_str());
        Serial.printf("Porta: %d\n", serverPort);
        return true;
    } else {
        Serial.println("\nFalha na conexão WiFi!");
        return false;
    }
}

void cleanup_model() {
    if (cifar10_model.model_buffer) {
        free(cifar10_model.model_buffer);
        cifar10_model.model_buffer = nullptr;
    }
    if (cifar10_model.tensor_arena) {
        free(cifar10_model.tensor_arena);
        cifar10_model.tensor_arena = nullptr;
    }
    cifar10_model.initialized = false;
}

void* allocate_memory(size_t size) {
    void* ptr = heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (ptr == nullptr) {
        ptr = malloc(size);
    }
    return ptr;
}

bool load_model() {
#ifndef HAS_MODEL_DATA
    Serial.println("ERRO: cifar10_model_data.h não encontrado!");
    return false;
#endif

    Serial.println("[1] Carregando modelo...");

    cifar10_model.model_buffer = static_cast<uint8_t*>(
        allocate_memory(cifar10_simple_int8_tflite_len));

    if (cifar10_model.model_buffer != nullptr) {
        Serial.printf("Copiando modelo (%d bytes) para memória...\n", cifar10_simple_int8_tflite_len);
        memcpy(cifar10_model.model_buffer, cifar10_simple_int8_tflite, cifar10_simple_int8_tflite_len);
        cifar10_model.model = tflite::GetModel(cifar10_model.model_buffer);
    } else {
        Serial.println("Usando modelo diretamente da Flash...");
        cifar10_model.model = tflite::GetModel(cifar10_simple_int8_tflite);
    }

    if (cifar10_model.model == nullptr) {
        Serial.println("ERRO: Falha ao carregar modelo");
        return false;
    }

    if (cifar10_model.model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("ERRO: Versão incompatível: %d vs %d\n",
                     cifar10_model.model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    Serial.println("Modelo carregado com sucesso");
    return true;
}

bool initialize_interpreter() {
    Serial.println("[2] Inicializando interpretador...");

    cifar10_model.tensor_arena = static_cast<uint8_t*>(
        allocate_memory(CIFAR10Model::kTensorArenaSize));

    if (cifar10_model.tensor_arena == nullptr) {
        Serial.printf("ERRO: Falha na alocação de %d bytes\n", CIFAR10Model::kTensorArenaSize);
        return false;
    }

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

    static tflite::MicroInterpreter static_interpreter(
        cifar10_model.model, op_resolver, cifar10_model.tensor_arena, CIFAR10Model::kTensorArenaSize);
    cifar10_model.interpreter = &static_interpreter;

    TfLiteStatus allocate_status = cifar10_model.interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.printf("ERRO: AllocateTensors falhou (código: %d)\n", allocate_status);
        return false;
    }

    cifar10_model.input_tensor = cifar10_model.interpreter->input(0);
    cifar10_model.output_tensor = cifar10_model.interpreter->output(0);

    if (cifar10_model.input_tensor == nullptr || cifar10_model.output_tensor == nullptr) {
        Serial.println("ERRO: Ponteiros de tensor nulos");
        return false;
    }

    Serial.printf("Arena usada: %lu/%d bytes\n",
                  cifar10_model.interpreter->arena_used_bytes(), CIFAR10Model::kTensorArenaSize);
    Serial.println("Interpretador inicializado com sucesso");
    return true;
}

bool initialize_cifar10_model() {
    Serial.println("=== Inicializando Modelo CIFAR-10 ===");

    static tflite::MicroErrorReporter micro_error_reporter;
    cifar10_model.error_reporter = &micro_error_reporter;

    if (!load_model()) {
        return false;
    }

    if (!initialize_interpreter()) {
        cleanup_model();
        return false;
    }

    cifar10_model.initialized = true;
    Serial.println("=== Modelo inicializado com sucesso ===\n");
    return true;
}

void preprocess_image(const uint8_t* image_data) {
    const float input_scale = cifar10_model.input_tensor->params.scale;
    const int32_t input_zero_point = cifar10_model.input_tensor->params.zero_point;

    for (int i = 0; i < CIFAR10Model::kImageSize; ++i) {
        float normalized_pixel = image_data[i] / 255.0f;
        int32_t quantized_value = static_cast<int32_t>(
            roundf(normalized_pixel / input_scale) + input_zero_point);
        quantized_value = max(-128, min(127, quantized_value));
        cifar10_model.input_tensor->data.int8[i] = static_cast<int8_t>(quantized_value);
    }
}

InferenceResult run_inference(const uint8_t* image_data) {
    InferenceResult result = {-1, 0.0f, false, ""};

    if (!cifar10_model.initialized) {
        result.error_message = "Modelo não inicializado";
        Serial.println("ERRO: " + result.error_message);
        return result;
    }

    preprocess_image(image_data);

    TfLiteStatus invoke_status = cifar10_model.interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        result.error_message = "Falha na execução da inferência";
        Serial.printf("ERRO: Invoke falhou (código: %d)\n", invoke_status);
        return result;
    }

    int best_index = 0;
    int8_t max_score = SCHAR_MIN;
    const int output_size = cifar10_model.output_tensor->dims->data[1];

    for (int i = 0; i < output_size; ++i) {
        if (cifar10_model.output_tensor->data.int8[i] > max_score) {
            max_score = cifar10_model.output_tensor->data.int8[i];
            best_index = i;
        }
    }

    const float output_scale = cifar10_model.output_tensor->params.scale;
    const int32_t output_zero_point = cifar10_model.output_tensor->params.zero_point;
    float confidence = (static_cast<float>(max_score) - output_zero_point) * output_scale;

    result.predicted_class = best_index;
    result.confidence = confidence;
    result.success = true;

    return result;
}

String create_json_response(const InferenceResult& result) {
    String response = "{\n";
    response += "  \"success\": " + String(result.success ? "true" : "false") + ",\n";
    response += "  \"predicted_class\": " + String(result.predicted_class) + ",\n";
    response += "  \"confidence\": " + String(result.confidence, 6) + ",\n";
    response += "  \"error_message\": \"" + result.error_message + "\",\n";
    response += "  \"heap_free\": " + String(esp_get_free_heap_size()) + ",\n";
    response += "  \"model_initialized\": " + String(cifar10_model.initialized ? "true" : "false") + "\n";
    response += "}";
    return response;
}

void handle_client() {
    WiFiClient client = server.available();
    if (!client) return;

    Serial.println("=== Cliente conectado ===");
    client.setTimeout(5000);

    String request = "";
    String headers = "";
    String body = "";
    bool reading_body = false;
    int content_length = 0;
    unsigned long start_time = millis();

    while (client.connected() && (millis() - start_time < 10000)) {
        if (!client.available()) {
            delay(1);
            continue;
        }
        String line = client.readStringUntil('\n');
        line.trim();

        if (!reading_body) {
            if (line.length() == 0) {
                reading_body = true;
                break;
            }
            if (request.length() == 0) request = line;
            if (line.startsWith("Content-Length:")) content_length = line.substring(15).toInt();
        }
    }

    if (content_length > 0 && content_length < 50000) {
        body.reserve(content_length + 1);
        unsigned long body_start = millis();
        while (body.length() < content_length && client.connected() && (millis() - body_start < 5000)) {
            if (client.available()) body += (char)client.read();
            else delay(1);
        }
    }

    Serial.println("Requisição: " + request);
    Serial.println("Body length: " + String(body.length()));

    String response_body = "";
    String content_type = "text/html";

    if (request.startsWith("POST /predict")) {
        content_type = "application/json";
        uint8_t image_data[CIFAR10Model::kImageSize];
        String parse_error = parse_json_array(body, image_data);
        InferenceResult result;

        if (parse_error.length() > 0) {
            result.success = false;
            result.error_message = parse_error;
            result.predicted_class = -1;
            result.confidence = 0.0f;
            Serial.println("ERRO no parsing: " + parse_error);
        } else {
            Serial.println("=== EXECUTANDO INFERÊNCIA ===");
            result = run_inference(image_data);
            if (result.success) {
                Serial.println("=== RESULTADO ===");
                Serial.printf("Predição: %d\n", result.predicted_class);
                Serial.printf("Confiança: %.6f\n", result.confidence);
                Serial.println("==================");
            } else {
                Serial.println("Falha na inferência: " + result.error_message);
            }
        }
        response_body = create_json_response(result);
    } else if (request.startsWith("GET /status")) {
        content_type = "application/json";
        InferenceResult status_result;
        status_result.success = cifar10_model.initialized;
        status_result.predicted_class = -1;
        status_result.confidence = 0.0f;
        status_result.error_message = cifar10_model.initialized ? "" : "Modelo não inicializado";
        response_body = create_json_response(status_result);
    } else {
        response_body = "<!DOCTYPE html><html><body><h1>CIFAR-10 API</h1><h2>Endpoints:</h2><p><b>POST /predict</b> - Body: {\"pixels\": [array de 3072 valores (32x32x3) 0-255]}</p><p><b>GET /status</b> - Status do sistema</p><p>IP: " + WiFi.localIP().toString() + "</p></body></html>";
    }

    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: " + content_type);
    client.println("Access-Control-Allow-Origin: *");
    client.println("Connection: close");
    client.println("Content-Length: " + String(response_body.length()));
    client.println();
    client.print(response_body);
    client.flush();
    delay(100);
    client.stop();
    Serial.println("Cliente desconectado\n");
}

String parse_json_array(String json_data, uint8_t* image_array) {
    int start_index = json_data.indexOf("\"pixels\":");
    if (start_index == -1) return "Campo 'pixels' não encontrado";

    start_index = json_data.indexOf('[', start_index);
    if (start_index == -1) return "Array de pixels não encontrado";

    int end_index = json_data.indexOf(']', start_index);
    if (end_index == -1) return "Fim do array não encontrado";

    String array_content = json_data.substring(start_index + 1, end_index);
    int pixel_count = 0;
    int current_pos = 0;

    while (current_pos < array_content.length() && pixel_count < CIFAR10Model::kImageSize) {
        while (current_pos < array_content.length() && isspace(array_content.charAt(current_pos))) {
            current_pos++;
        }
        if (current_pos >= array_content.length()) break;

        int comma_pos = array_content.indexOf(',', current_pos);
        String value_str = (comma_pos == -1) ?
            array_content.substring(current_pos) :
            array_content.substring(current_pos, comma_pos);
        value_str.trim();

        bool is_valid_number = (value_str.length() > 0);
        for (int i = 0; i < value_str.length() && is_valid_number; i++) {
            if (!isdigit(value_str.charAt(i))) is_valid_number = false;
        }

        if (!is_valid_number) {
            return "Valor inválido no índice " + String(pixel_count) + ": '" + value_str + "'";
        }

        int pixel_value = value_str.toInt();
        image_array[pixel_count] = (uint8_t)constrain(pixel_value, 0, 255);
        pixel_count++;

        if (comma_pos == -1) break;
        current_pos = comma_pos + 1;
    }

    if (pixel_count != CIFAR10Model::kImageSize) {
        return "Array deve ter " + String(CIFAR10Model::kImageSize) + " valores, recebido: " + String(pixel_count);
    }
    return "";
}

void setup() {
    Serial.begin(115200);
    delay(2000);

    Serial.println("\n=== CIFAR-10 TensorFlow Lite WiFi API ===");
    Serial.printf("Free heap inicial: %lu bytes\n", esp_get_free_heap_size());
    Serial.printf("PSRAM disponível: %u bytes\n", ESP.getPsramSize());

    if (!connect_wifi()) {
        Serial.println("Falha na conexão WiFi - reiniciando...");
        ESP.restart();
    }

    if (!initialize_cifar10_model()) {
        Serial.println("Falha na inicialização do modelo! Parando.");
        while(true) { delay(1000); }
    }

    server.begin();
    Serial.println("\n=== Servidor HTTP iniciado ===");
}

void loop() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi desconectado - tentando reconectar...");
        connect_wifi();
    }
    handle_client();
    delay(10);
}