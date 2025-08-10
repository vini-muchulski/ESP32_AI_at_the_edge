#include <Arduino.h>
#include <WiFi.h>
#include <WiFiServer.h>
#include <WiFiClient.h>
#include <cmath>
#include <climits>

// Verificar se os headers existem
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

// Configurações WiFi - ALTERE AQUI
const char* ssid = "Starlink";
const char* password = "diversao";
const int serverPort = 80;

// Servidor WiFi
WiFiServer server(serverPort);

// Model data from external files (se existirem)
#ifdef HAS_MODEL_DATA
extern unsigned char mnist_cnn_small_int8_tflite[];
extern unsigned int mnist_cnn_small_int8_tflite_len;
#endif

// Dados de teste mockados se não tiver image_data.h
#ifndef HAS_IMAGE_DATA
const uint8_t mnist_sample[784] PROGMEM = {0}; // Imagem vazia para teste
#endif

// Estrutura para gerenciar o modelo
struct MNISTModel {
    tflite::ErrorReporter* error_reporter;
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input_tensor;
    TfLiteTensor* output_tensor;
    uint8_t* tensor_arena;
    uint8_t* model_buffer;
    bool initialized;
    
    static constexpr int kTensorArenaSize = 80 * 1024;
    static constexpr int kImageSize = 28 * 28;
};

// Instância global do modelo
MNISTModel mnist_model = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, false};

// Estrutura para resultado da inferência
struct InferenceResult {
    int predicted_digit;
    float confidence;
    bool success;
    String error_message;
};

// Declarações das funções
void cleanup_model();
bool connect_wifi();
void handle_client();
String parse_json_array(String json_data, uint8_t* image_array);
String create_json_response(const InferenceResult& result);

// Função para conectar ao WiFi
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

// Função para limpeza de memória
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

// Função para alocar memória preferindo PSRAM
void* allocate_memory(size_t size) {
    void* ptr = heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (ptr == nullptr) {
        ptr = malloc(size);
    }
    return ptr;
}

// Função para carregar o modelo
bool load_model() {
#ifndef HAS_MODEL_DATA
    Serial.println("ERRO: mnist_model_data.h não encontrado!");
    return false;
#endif

    Serial.println("[1] Carregando modelo...");
    
    // Tentar carregar modelo no PSRAM
    mnist_model.model_buffer = static_cast<uint8_t*>(
        allocate_memory(mnist_cnn_small_int8_tflite_len));
    
    if (mnist_model.model_buffer != nullptr) {
        Serial.printf("Copiando modelo (%d bytes) para memória...\n", mnist_cnn_small_int8_tflite_len);
        memcpy(mnist_model.model_buffer, mnist_cnn_small_int8_tflite, mnist_cnn_small_int8_tflite_len);
        mnist_model.model = tflite::GetModel(mnist_model.model_buffer);
    } else {
        Serial.println("Usando modelo diretamente da Flash...");
        mnist_model.model = tflite::GetModel(mnist_cnn_small_int8_tflite);
    }
    
    if (mnist_model.model == nullptr) {
        Serial.println("ERRO: Falha ao carregar modelo");
        return false;
    }
    
    if (mnist_model.model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("ERRO: Versão incompatível: %d vs %d\n",
                     mnist_model.model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    
    Serial.println("Modelo carregado com sucesso");
    return true;
}

// Função para inicializar o interpretador
bool initialize_interpreter() {
    Serial.println("[2] Inicializando interpretador...");
    
    // Alocar tensor arena
    mnist_model.tensor_arena = static_cast<uint8_t*>(
        allocate_memory(MNISTModel::kTensorArenaSize));
    
    if (mnist_model.tensor_arena == nullptr) {
        Serial.printf("ERRO: Falha na alocação de %d bytes\n", MNISTModel::kTensorArenaSize);
        return false;
    }
    
    // Configurar op resolver
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
    
    // Criar interpretador
    static tflite::MicroInterpreter static_interpreter(
        mnist_model.model, op_resolver, mnist_model.tensor_arena, MNISTModel::kTensorArenaSize);
    mnist_model.interpreter = &static_interpreter;
    
    // Alocar tensores
    TfLiteStatus allocate_status = mnist_model.interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.printf("ERRO: AllocateTensors falhou (código: %d)\n", allocate_status);
        return false;
    }
    
    // Obter ponteiros dos tensores
    mnist_model.input_tensor = mnist_model.interpreter->input(0);
    mnist_model.output_tensor = mnist_model.interpreter->output(0);
    
    if (mnist_model.input_tensor == nullptr || mnist_model.output_tensor == nullptr) {
        Serial.println("ERRO: Ponteiros de tensor nulos");
        return false;
    }
    
    Serial.printf("Arena usada: %d/%d bytes\n", 
                  mnist_model.interpreter->arena_used_bytes(), MNISTModel::kTensorArenaSize);
    Serial.println("Interpretador inicializado com sucesso");
    return true;
}

// Função para inicializar o modelo completo
bool initialize_mnist_model() {
    Serial.println("=== Inicializando Modelo MNIST ===");
    
    // Inicializar error reporter
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
    Serial.println("=== Modelo inicializado com sucesso ===\n");
    return true;
}

// Função para preprocessar a imagem
void preprocess_image(const uint8_t* image_data) {
    const float input_scale = mnist_model.input_tensor->params.scale;
    const int32_t input_zero_point = mnist_model.input_tensor->params.zero_point;
    
    for (int i = 0; i < MNISTModel::kImageSize; ++i) {
        uint8_t pixel = image_data[i];
        float normalized_pixel = pixel / 255.0f;
        int32_t quantized_value = static_cast<int32_t>(
            roundf(normalized_pixel / input_scale) + input_zero_point);
        quantized_value = max(-128, min(127, quantized_value));
        mnist_model.input_tensor->data.int8[i] = static_cast<int8_t>(quantized_value);
    }
}

// Função para fazer inferência
InferenceResult run_inference(const uint8_t* image_data) {
    InferenceResult result = {-1, 0.0f, false, ""};
    
    if (!mnist_model.initialized) {
        result.error_message = "Modelo não inicializado";
        Serial.println("ERRO: Modelo não inicializado");
        return result;
    }
    
    // Preprocessar imagem
    preprocess_image(image_data);
    
    // Executar inferência
    TfLiteStatus invoke_status = mnist_model.interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        result.error_message = "Falha na execução da inferência";
        Serial.printf("ERRO: Invoke falhou (código: %d)\n", invoke_status);
        return result;
    }
    
    // Analisar resultado
    int best_index = 0;
    int8_t max_score = SCHAR_MIN;
    const int output_size = mnist_model.output_tensor->dims->data[1];
    
    for (int i = 0; i < output_size; ++i) {
        if (mnist_model.output_tensor->data.int8[i] > max_score) {
            max_score = mnist_model.output_tensor->data.int8[i];
            best_index = i;
        }
    }
    
    // Converter score para float
    const float output_scale = mnist_model.output_tensor->params.scale;
    const int32_t output_zero_point = mnist_model.output_tensor->params.zero_point;
    float confidence = (static_cast<float>(max_score) - output_zero_point) * output_scale;
    
    result.predicted_digit = best_index;
    result.confidence = confidence;
    result.success = true;
    
    return result;
}


// Função para criar resposta JSON
String create_json_response(const InferenceResult& result) {
    String response = "{\n";
    response += "  \"success\": " + String(result.success ? "true" : "false") + ",\n";
    response += "  \"predicted_digit\": " + String(result.predicted_digit) + ",\n";
    response += "  \"confidence\": " + String(result.confidence, 6) + ",\n";
    response += "  \"error_message\": \"" + result.error_message + "\",\n";
    response += "  \"heap_free\": " + String(esp_get_free_heap_size()) + ",\n";
    response += "  \"model_initialized\": " + String(mnist_model.initialized ? "true" : "false") + "\n";
    response += "}";
    return response;
}

// Função para lidar com clientes HTTP
// Função para lidar com clientes HTTP - VERSÃO CORRIGIDA
void handle_client() {
    WiFiClient client = server.available();
    if (!client) return;
    
    Serial.println("=== Cliente conectado ===");
    
    // Configurar timeout do cliente
    client.setTimeout(5000);
    
    String request = "";
    String headers = "";
    String body = "";
    bool reading_body = false;
    int content_length = 0;
    
    // Ler requisição HTTP com timeout
    unsigned long start_time = millis();
    const unsigned long timeout_ms = 10000; // 10 segundos
    
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
                break; // Parar de ler headers
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
    
    // Ler body se POST e tiver Content-Length
    if (content_length > 0 && content_length < 50000) { // Limite de segurança
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
    
    Serial.println("Requisição: " + request);
    Serial.println("Body length: " + String(body.length()));
    
    String response_body = "";
    String content_type = "text/html";
    
    // Processar requisição
    if (request.startsWith("POST /predict")) {
        content_type = "application/json";
        
        // Processar inferência
        uint8_t image_data[784];
        String parse_error = parse_json_array(body, image_data);
        
        InferenceResult result;
        if (parse_error.length() > 0) {
            result.success = false;
            result.error_message = parse_error;
            result.predicted_digit = -1;
            result.confidence = 0.0f;
            Serial.println("ERRO no parsing: " + parse_error);
        } else {
            Serial.println("=== EXECUTANDO INFERÊNCIA ===");
            result = run_inference(image_data);
            
            if (result.success) {
                Serial.println("=== RESULTADO ===");
                Serial.printf("Predição: %d\n", result.predicted_digit);
                Serial.printf("Confiança: %.6f\n", result.confidence);
                Serial.println("==================");
            } else {
                Serial.println("Falha na inferência: " + result.error_message);
            }
        }
        
        response_body = create_json_response(result);
        
    } else if (request.startsWith("GET /status")) {
        content_type = "application/json";
        
        // Status do sistema
        InferenceResult status_result;
        status_result.success = mnist_model.initialized;
        status_result.predicted_digit = -1;
        status_result.confidence = 0.0f;
        status_result.error_message = mnist_model.initialized ? "" : "Modelo não inicializado";
        
        response_body = create_json_response(status_result);
        
    } else {
        // Página de ajuda
        response_body = "<!DOCTYPE html><html><body>";
        response_body += "<h1>MNIST API</h1>";
        response_body += "<h2>Endpoints:</h2>";
        response_body += "<p><b>POST /predict</b> - Fazer inferência</p>";
        response_body += "<p>Body JSON: {\"pixels\": [array de 784 valores 0-255]}</p>";
        response_body += "<p><b>GET /status</b> - Status do sistema</p>";
        response_body += "<p>IP: " + WiFi.localIP().toString() + "</p>";
        response_body += "</body></html>";
    }
    
    // Enviar resposta HTTP com headers apropriados
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: " + content_type);
    client.println("Access-Control-Allow-Origin: *");
    client.println("Access-Control-Allow-Methods: GET, POST, OPTIONS");
    client.println("Access-Control-Allow-Headers: Content-Type");
    client.println("Connection: close");
    client.println("Content-Length: " + String(response_body.length()));
    client.println(); // Linha vazia importante
    client.print(response_body);
    client.flush(); // Garantir que todos os dados foram enviados
    
    // Pequena pausa antes de fechar
    delay(100);
    
    client.stop();
    Serial.println("Cliente desconectado\n");
}

// Função para parsear array JSON - VERSÃO MAIS ROBUSTA
String parse_json_array(String json_data, uint8_t* image_array) {
    // Procurar pelo array "pixels"
    int start_index = json_data.indexOf("\"pixels\":");
    if (start_index == -1) {
        return "Campo 'pixels' não encontrado";
    }
    
    start_index = json_data.indexOf('[', start_index);
    if (start_index == -1) {
        return "Array de pixels não encontrado";
    }
    
    int end_index = json_data.indexOf(']', start_index);
    if (end_index == -1) {
        return "Fim do array não encontrado";
    }
    
    String array_content = json_data.substring(start_index + 1, end_index);
    array_content.trim();
    
    // Parsear valores com melhor tratamento de erros
    int pixel_count = 0;
    int current_pos = 0;
    
    while (current_pos < array_content.length() && pixel_count < 784) {
        // Pular espaços em branco
        while (current_pos < array_content.length() && 
               (array_content.charAt(current_pos) == ' ' || 
                array_content.charAt(current_pos) == '\t' ||
                array_content.charAt(current_pos) == '\n' ||
                array_content.charAt(current_pos) == '\r')) {
            current_pos++;
        }
        
        if (current_pos >= array_content.length()) break;
        
        // Encontrar próxima vírgula ou fim
        int comma_pos = array_content.indexOf(',', current_pos);
        String value_str;
        
        if (comma_pos == -1) {
            value_str = array_content.substring(current_pos);
        } else {
            value_str = array_content.substring(current_pos, comma_pos);
        }
        
        value_str.trim();
        
        // Verificar se é número válido
        bool is_valid_number = true;
        for (int i = 0; i < value_str.length(); i++) {
            char c = value_str.charAt(i);
            if (!isdigit(c) && c != '-' && c != '+') {
                is_valid_number = false;
                break;
            }
        }
        
        if (!is_valid_number || value_str.length() == 0) {
            return "Valor inválido no índice " + String(pixel_count) + ": '" + value_str + "'";
        }
        
        int pixel_value = value_str.toInt();
        
        // Validar range do pixel (0-255)
        if (pixel_value < 0) pixel_value = 0;
        if (pixel_value > 255) pixel_value = 255;
        
        image_array[pixel_count] = (uint8_t)pixel_value;
        pixel_count++;
        
        if (comma_pos == -1) break;
        current_pos = comma_pos + 1;
    }
    
    if (pixel_count != 784) {
        return "Array deve ter exatamente 784 pixels (28x28), recebido: " + String(pixel_count);
    }
    
    return ""; // Sucesso
}


void setup() {
    Serial.begin(115200);
    delay(2000);
    
    Serial.println("\n=== MNIST TensorFlow Lite WiFi API ===");
    Serial.printf("Free heap inicial: %d bytes\n", esp_get_free_heap_size());
    Serial.printf("PSRAM disponível: %d bytes\n", ESP.getPsramSize());
    
    // Conectar WiFi
    if (!connect_wifi()) {
        Serial.println("Falha na conexão WiFi - reiniciando...");
        ESP.restart();
    }
    
    // Inicializar modelo
    if (!initialize_mnist_model()) {
        Serial.println("Falha na inicialização do modelo!");
        return;
    }
    
    // Iniciar servidor
    server.begin();
    Serial.println("\n=== Servidor HTTP iniciado ===");
    Serial.println("Endpoints disponíveis:");
    Serial.println("POST /predict - Fazer inferência");
    Serial.println("GET /status - Status do sistema");
    Serial.println("GET / - Página de ajuda");
    Serial.println("============================\n");
}

void loop() {
    // Verificar conexão WiFi
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi desconectado - tentando reconectar...");
        if (!connect_wifi()) {
            delay(5000);
            return;
        }
    }
    
    // Lidar com clientes
    handle_client();
    
    delay(10); // Pequeno delay para não sobrecarregar
}