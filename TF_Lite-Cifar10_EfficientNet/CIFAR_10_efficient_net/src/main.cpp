#include <Arduino.h>
#include <WiFi.h>
#include <WiFiServer.h>
#include <WiFiClient.h>
#include <cmath>
#include <climits>
#include <memory>

#include "cifar10_model_data.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

const char *ssid = "";
const char *password = "";
const int serverPort = 80;

WiFiServer server(serverPort);

struct CIFAR10Model
{
    tflite::ErrorReporter *error_reporter;
    const tflite::Model *model;
    tflite::MicroInterpreter *interpreter;
    TfLiteTensor *input_tensor;
    TfLiteTensor *output_tensor;
    uint8_t *tensor_arena;
    uint8_t *model_buffer;
    bool initialized;

    static constexpr int kInputWidth = 224;
    static constexpr int kInputHeight = 224;
    static constexpr int kInputChannels = 3;
    static constexpr int kImageSize = kInputWidth * kInputHeight * kInputChannels;
    static constexpr int kTensorArenaSize = 6 * 1024 * 1024;
};

CIFAR10Model cifar10_model = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, false};

struct InferenceResult
{
    int predicted_class;
    float confidence;
    bool success;
    String error_message;
};

void cleanup_model();
bool connect_wifi();
String parse_json_stream(WiFiClient &client, int content_length, uint8_t *image_array);
void handle_client();
String parse_json_array(const String &json_data, uint8_t *image_array);
String create_json_response(const InferenceResult &result);
bool initialize_cifar10_model();
InferenceResult run_inference(const uint8_t *image_data);

bool connect_wifi()
{
    Serial.println("=== Conectando ao WiFi ===");
    Serial.printf("SSID: %s\n", ssid);
    WiFi.begin(ssid, password);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30)
    {
        delay(1000);
        Serial.print(".");
        attempts++;
    }
    if (WiFi.status() == WL_CONNECTED)
    {
        Serial.println("\nWiFi conectado!");
        Serial.printf("IP: %s\n", WiFi.localIP().toString().c_str());
        Serial.printf("Porta: %d\n", serverPort);
        return true;
    }
    Serial.println("\nFalha na conexão WiFi!");
    return false;
}

void cleanup_model()
{
    if (cifar10_model.model_buffer)
    {
        free(cifar10_model.model_buffer);
        cifar10_model.model_buffer = nullptr;
    }
    if (cifar10_model.tensor_arena)
    {
        free(cifar10_model.tensor_arena);
        cifar10_model.tensor_arena = nullptr;
    }
    cifar10_model.initialized = false;
}

void *allocate_memory(size_t size)
{
    void *ptr = heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (ptr == nullptr)
    {
        ptr = malloc(size);
    }
    return ptr;
}

bool load_model()
{
    Serial.println("[1] Carregando modelo...");

    cifar10_model.model_buffer = nullptr;
    Serial.println("Usando modelo diretamente da Flash (PROGMEM)...");
    cifar10_model.model = tflite::GetModel(cifar10_EfficientNetB0_small_finetuned_int8_tflite);

    if (cifar10_model.model == nullptr)
    {
        Serial.println("ERRO: Falha ao carregar modelo.");
        return false;
    }
    if (cifar10_model.model->version() != TFLITE_SCHEMA_VERSION)
    {
        Serial.printf("ERRO: Versão do modelo incompatível (%d vs %d)\n",
                      cifar10_model.model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    Serial.println("Modelo carregado com sucesso.");
    return true;
}

bool initialize_interpreter()
{
    Serial.println("[2] Inicializando interpretador...");

    cifar10_model.tensor_arena = static_cast<uint8_t *>(allocate_memory(CIFAR10Model::kTensorArenaSize));
    if (cifar10_model.tensor_arena == nullptr)
    {
        Serial.printf("ERRO: Falha na alocação da arena de tensores (%d bytes).\n", CIFAR10Model::kTensorArenaSize);
        return false;
    }

    // Resolver com 20 operadores (capacidade extra para garantir)
    static tflite::MicroMutableOpResolver<30> op_resolver;

    // Operadores de convolução
    op_resolver.AddConv2D();
    op_resolver.AddDepthwiseConv2D();

    // Operadores de pooling
    op_resolver.AddMaxPool2D();
    op_resolver.AddAveragePool2D();
    op_resolver.AddMean();

    // Operadores de reshape e resize
    op_resolver.AddReshape();
    op_resolver.AddResizeNearestNeighbor();
    op_resolver.AddPad();
    op_resolver.AddShape();
    op_resolver.AddStridedSlice();

    // Operadores de camada densa e ativação
    op_resolver.AddFullyConnected();
    op_resolver.AddSoftmax();
    op_resolver.AddRelu();
    op_resolver.AddRelu6();
    op_resolver.AddLogistic();

    // Operadores aritméticos
    op_resolver.AddAdd();
    op_resolver.AddSub();
    op_resolver.AddMul();

    // Operadores de quantização
    op_resolver.AddQuantize();
    op_resolver.AddDequantize();

    op_resolver.AddPack();
    op_resolver.AddUnpack();
    op_resolver.AddCast();
    op_resolver.AddExpandDims();
    op_resolver.AddSqueeze();
    op_resolver.AddSlice();
    op_resolver.AddConcatenation();
    op_resolver.AddTranspose();
    op_resolver.AddGather();

    static tflite::MicroInterpreter static_interpreter(
        cifar10_model.model,
        op_resolver,
        cifar10_model.tensor_arena,
        CIFAR10Model::kTensorArenaSize);
    cifar10_model.interpreter = &static_interpreter;

    TfLiteStatus allocate_status = cifar10_model.interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        Serial.printf("ERRO: AllocateTensors falhou (código: %d).\n", allocate_status);
        return false;
    }

    cifar10_model.input_tensor = cifar10_model.interpreter->input(0);
    cifar10_model.output_tensor = cifar10_model.interpreter->output(0);

    if (cifar10_model.input_tensor == nullptr || cifar10_model.output_tensor == nullptr)
    {
        Serial.println("ERRO: Ponteiros de tensor de entrada/saída nulos.");
        return false;
    }

    Serial.printf("Arena usada: %lu/%d bytes (%.1f%%)\n",
                  cifar10_model.interpreter->arena_used_bytes(),
                  CIFAR10Model::kTensorArenaSize,
                  (cifar10_model.interpreter->arena_used_bytes() * 100.0f) / CIFAR10Model::kTensorArenaSize);
    Serial.println("Interpretador inicializado com sucesso.");
    return true;
}

bool initialize_cifar10_model()
{
    Serial.println("=== Inicializando Modelo CIFAR-10 ===");
    static tflite::MicroErrorReporter micro_error_reporter;
    cifar10_model.error_reporter = &micro_error_reporter;
    if (!load_model())
        return false;
    if (!initialize_interpreter())
    {
        cleanup_model();
        return false;
    }
    cifar10_model.initialized = true;
    Serial.println("=== Modelo inicializado com sucesso ===\n");
    return true;
}

void load_int8_input(const uint8_t *src)
{
    memcpy(cifar10_model.input_tensor->data.int8, src, CIFAR10Model::kImageSize);
}

void preprocess_image(const uint8_t *image_data)
{
    const float s = cifar10_model.input_tensor->params.scale;
    const int32_t zp = cifar10_model.input_tensor->params.zero_point;

    for (int i = 0; i < CIFAR10Model::kImageSize; ++i)
    {
        // CORREÇÃO: Converter de [0, 255] para [-1, 1] primeiro
        float normalized = (static_cast<float>(image_data[i]) / 127.5f) - 1.0f;

        // Agora quantizar o valor normalizado
        int32_t q = lrintf(normalized / s) + zp;
        q = std::max<int32_t>(SCHAR_MIN, std::min<int32_t>(SCHAR_MAX, q));
        cifar10_model.input_tensor->data.int8[i] = static_cast<int8_t>(q);
    }
}

InferenceResult run_inference(const uint8_t *image_data)
{
    InferenceResult result = {-1, 0.0f, false, ""};
    if (!cifar10_model.initialized)
    {
        result.error_message = "Modelo não inicializado.";
        Serial.println("ERRO: " + result.error_message);
        return result;
    }
    // preprocess_image(image_data);
    load_int8_input(image_data);
    if (cifar10_model.interpreter->Invoke() != kTfLiteOk)
    {
        result.error_message = "Falha na execução da inferência.";
        Serial.println("ERRO: " + result.error_message);
        return result;
    }
    int best_index = 0;
    int8_t max_score = SCHAR_MIN;
    const int output_size = cifar10_model.output_tensor->dims->data[1];
    for (int i = 0; i < output_size; ++i)
    {
        if (cifar10_model.output_tensor->data.int8[i] > max_score)
        {
            max_score = cifar10_model.output_tensor->data.int8[i];
            best_index = i;
        }
    }
    const float output_scale = cifar10_model.output_tensor->params.scale;
    const int32_t output_zero_point = cifar10_model.output_tensor->params.zero_point;
    result.confidence = (static_cast<float>(max_score) - output_zero_point) * output_scale;
    result.predicted_class = best_index;
    result.success = true;
    return result;
}

String create_json_response(const InferenceResult &result)
{
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

/* ---------- parse_json_stream COM CRONÔMETRO ---------- */
String parse_json_stream(WiFiClient &client, int content_length, uint8_t *image_array)
{
    uint32_t t_start = millis(); // ⏱ inicio
    constexpr size_t BUF = 2048;
    char buf[BUF];
    String num;
    int pixels = 0;
    bool open = false;
    int remaining = content_length;
    int processed = 0;
    unsigned long last_data = millis();

    while (remaining > 0 && client.connected())
    {
        int n = client.read(reinterpret_cast<uint8_t *>(buf), std::min<int>(BUF, remaining));
        if (n > 0)
        {
            last_data = millis();
            processed += n;
            remaining -= n;
            Serial.printf("RX %d/%d bytes\r\n", processed, content_length);

            for (int i = 0; i < n; ++i)
            {
                char c = buf[i];
                if (!open)
                {
                    if (c == '[')
                        open = true;
                    continue;
                }
                if (c >= '0' && c <= '9')
                {
                    num += c;
                    continue;
                }
                if (c == ',' || c == ']')
                {
                    if (num.length())
                    {
                        if (pixels >= CIFAR10Model::kImageSize)
                            return "Limite excedido.";
                        image_array[pixels++] = uint8_t(constrain(num.toInt(), 0, 255));
                        num = "";
                    }
                }
            }
        }
        else if (millis() - last_data > 5000)
        {
            Serial.printf("Tempo parse JSON: %ums (timeout)\n", millis() - t_start);
            return "Timeout após " + String(processed) + " bytes.";
        }
    }

    if (remaining != 0)
    {
        Serial.printf("Tempo parse JSON: %ums (incompleto)\n", millis() - t_start);
        return "Body incompleto.";
    }
    if (pixels != CIFAR10Model::kImageSize)
    {
        Serial.printf("Tempo parse JSON: %ums (pixels)\n", millis() - t_start);
        return "Pixels faltando.";
    }

    Serial.println("\nRX completo");
    Serial.printf("Tempo parse JSON: %ums\n", millis() - t_start); // ⏱ fim
    return "";
}

/* ---------- read_raw COM CRONÔMETRO ---------- */
static String read_raw(WiFiClient &c, int len, uint8_t *dst)
{
    uint32_t t_start = millis(); // ⏱ inicio
    constexpr size_t REPORT = 16 * 1024;
    constexpr uint32_t IDLEMS = 30000;
    size_t done = 0, next = REPORT;
    uint32_t last = millis();

    while (done < (size_t)len)
    {
        if (!c.connected())
        {
            Serial.printf("Tempo read_raw: %ums (perda)\n", millis() - t_start);
            return "Conexão perdida.";
        }
        int avail = c.available();
        if (avail)
        {
            int n = c.read(dst + done, std::min<int>(avail, len - done));
            if (n <= 0)
                continue;
            done += n;
            if (done >= next)
            {
                Serial.printf("RX %u/%u bytes (%.1f%%)\n", done, len, done * 100.0f / len);
                next += REPORT;
            }
            last = millis();
        }
        else if (millis() - last > IDLEMS)
        {
            Serial.printf("Tempo read_raw: %ums (timeout)\n", millis() - t_start);
            return "Timeout após " + String(done) + " bytes.";
        }
        delay(1);
        yield();
    }
    Serial.printf("Tempo read_raw: %ums\n", millis() - t_start); // ⏱ fim
    return "";
}

/* ---------- handle_client COM CRONÔMETROS ---------- */
void handle_client()
{
    WiFiClient client = server.available();
    if (!client)
        return;
    client.setNoDelay(true);
    uint32_t t_total = millis(); // ⏱ total
    Serial.println("=== Cliente conectado ===");

    String req, line;
    int content_len = 0;
    uint32_t t_headers_start = millis();

    while (client.connected())
    {
        line = client.readStringUntil('\n');
        line.trim();
        if (line.isEmpty())
            break;
        if (req.isEmpty())
            req = line;
        if (line.startsWith("Content-Length:"))
            content_len = line.substring(15).toInt();
    }
    Serial.printf("Headers: %ums\n", millis() - t_headers_start); // ⏱ headers

    Serial.println("Requisição: " + req);
    Serial.printf("Content-Length: %d bytes\n", content_len);

    String ctype = "text/html", body;
    InferenceResult res{-1, 0, false, ""};

    /* --- POST /predict_bin ------------------------------------------------ */
    if (req.startsWith("POST /predict_bin"))
    {
        ctype = "application/json";
        if (content_len != CIFAR10Model::kImageSize)
        {
            res.success = false;
            res.error_message = "Content-Length deve ser 150528.";
        }
        else
        {
            std::unique_ptr<uint8_t[]> img(new (std::nothrow) uint8_t[CIFAR10Model::kImageSize]);
            if (!img)
            {
                res.success = false;
                res.error_message = "Falha de memória.";
            }
            else
            {
                uint32_t t_rx_start = millis();
                String err = read_raw(client, content_len, img.get());
                uint32_t t_rx_end = millis();
                Serial.printf("Tempo RX bin: %ums\n", t_rx_end - t_rx_start); // ⏱ RX bin

                if (err.isEmpty())
                {
                    Serial.println("=== EXECUTANDO INFERÊNCIA ===");
                    uint32_t t_inf_start = millis();
                    res = run_inference(img.get());
                    uint32_t t_inf_end = millis();
                    Serial.printf("Tempo inferência: %ums\n", t_inf_end - t_inf_start); // ⏱ inf
                }
                else
                {
                    res.success = false;
                    res.error_message = err;
                }
            }
        }
        body = create_json_response(res);
    }
    /* --- POST /predict (JSON) -------------------------------------------- */
    else if (req.startsWith("POST /predict"))
    {
        ctype = "application/json";
        if (content_len <= 0 || content_len > 2000000)
        {
            res.success = false;
            res.error_message = "Content-Length inválido.";
        }
        else
        {
            std::unique_ptr<uint8_t[]> img(new (std::nothrow) uint8_t[CIFAR10Model::kImageSize]);
            if (!img)
            {
                res.success = false;
                res.error_message = "Falha de memória.";
            }
            else
            {
                uint32_t t_parse_start = millis();
                String err = parse_json_stream(client, content_len, img.get());
                uint32_t t_parse_end = millis();
                Serial.printf("Tempo parse JSON: %ums\n", t_parse_end - t_parse_start); // ⏱ parse

                if (err.isEmpty())
                {
                    Serial.println("=== EXECUTANDO INFERÊNCIA ===");
                    uint32_t t_inf_start = millis();
                    res = run_inference(img.get());
                    uint32_t t_inf_end = millis();
                    Serial.printf("Tempo inferência: %ums\n", t_inf_end - t_inf_start); // ⏱ inf
                }
                else
                {
                    res.success = false;
                    res.error_message = err;
                }
            }
        }
        body = create_json_response(res);
    }
    /* --- GET /status ou raiz --------------------------------------------- */
    else if (req.startsWith("GET /status"))
    {
        ctype = "application/json";
        res.success = cifar10_model.initialized;
        res.error_message = res.success ? "" : "Modelo não inicializado";
        body = create_json_response(res);
    }
    else
    {
        body =
            "<!DOCTYPE html><html><body><h1>CIFAR-10 EfficientNetB0 API</h1>"
            "<p><b>POST /predict_bin</b> 150 528 B raw</p>"
            "<p><b>POST /predict</b> JSON pixels[]</p>"
            "<p><b>GET /status</b></p>"
            "<p>IP: " +
            WiFi.localIP().toString() + "</p></body></html>";
    }

    uint32_t t_send_start = millis();
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: " + ctype);
    client.println("Access-Control-Allow-Origin: *");
    client.println("Connection: close");
    client.println("Content-Length: " + String(body.length()));
    client.println();
    client.print(body);                                             // write único
    Serial.printf("Tempo send(): %ums\n", millis() - t_send_start); // ⏱ send

    client.stop();
    Serial.printf("Total conexão: %ums\n\n", millis() - t_total); // ⏱ total
}

String parse_json_array(const String &json_data, uint8_t *image_array)
{
    int start_index = json_data.indexOf("\"pixels\":");
    if (start_index == -1)
        return "Campo 'pixels' não encontrado.";
    start_index = json_data.indexOf('[', start_index);
    if (start_index == -1)
        return "Array de pixels não encontrado.";
    int end_index = json_data.indexOf(']', start_index);
    if (end_index == -1)
        return "Fim do array não encontrado.";

    String array_content = json_data.substring(start_index + 1, end_index);
    int pixel_count = 0;
    int current_pos = 0;

    while (current_pos < array_content.length() && pixel_count < CIFAR10Model::kImageSize)
    {
        int next_comma = array_content.indexOf(',', current_pos);
        String value_str = (next_comma == -1) ? array_content.substring(current_pos) : array_content.substring(current_pos, next_comma);
        value_str.trim();

        if (value_str.isEmpty())
        {
            current_pos = (next_comma == -1) ? array_content.length() : next_comma + 1;
            continue;
        }

        int pixel_value = value_str.toInt();
        image_array[pixel_count++] = (uint8_t)constrain(pixel_value, 0, 255);
        if (next_comma == -1)
            break;
        current_pos = next_comma + 1;
    }

    if (pixel_count != CIFAR10Model::kImageSize)
    {
        return "Array deve ter " + String(CIFAR10Model::kImageSize) + " valores, recebido: " + String(pixel_count);
    }
    return "";
}

void setup()
{
    Serial.begin(115200);
    delay(2000);
    WiFi.setSleep(false);
    Serial.println("\n=== CIFAR-10 TensorFlow Lite WiFi API ===");
    Serial.printf("Free heap inicial: %u bytes\n", esp_get_free_heap_size());
    Serial.printf("PSRAM disponível: %u bytes\n", ESP.getPsramSize());
    if (!connect_wifi())
    {
        Serial.println("Falha na conexão WiFi. Reiniciando...");
        ESP.restart();
    }
    if (!initialize_cifar10_model())
    {
        Serial.println("Falha na inicialização do modelo. Parando.");
        while (true)
            delay(1000);
    }
    server.begin();
    Serial.println("\n=== Servidor HTTP iniciado ===");
}

void loop()
{
    if (WiFi.status() != WL_CONNECTED)
    {
        Serial.println("WiFi desconectado. Tentando reconectar...");
        if (!connect_wifi())
        {
            delay(5000);
        }
    }
    handle_client();
}