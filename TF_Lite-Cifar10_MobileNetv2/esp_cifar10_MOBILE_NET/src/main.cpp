#include <Arduino.h>
#include <WiFi.h>
#include <WiFiServer.h>
#include <WiFiClient.h>
#include <cmath>
#include <climits>

#include "mobilenetv2_model_data.h"

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

  static constexpr int kInputWidth = 96;
  static constexpr int kInputHeight = 96;
  static constexpr int kInputChannels = 3;
  static constexpr int kImageSize = kInputWidth * kInputHeight * kInputChannels;
  static constexpr int kTensorArenaSize = 450 * 1024;
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
void handle_client();
String parse_json_array(String json_data, uint8_t *image_array);
String create_json_response(const InferenceResult &result);
bool initialize_cifar10_model();
InferenceResult run_inference(const uint8_t *image_data);

bool connect_wifi()
{
  Serial.println("=== Conectando ao WiFi ===");
  Serial.printf("SSID: %s\n", ssid);

  WiFi.setSleep(false);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);
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
  else
  {
    Serial.println("\nFalha na conexão WiFi!");
    return false;
  }
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

// Função load_model()

bool load_model()
{

  Serial.println("[1] Carregando modelo...");
  cifar10_model.model = tflite::GetModel(cifar10_mobilenetv2_finetuned_int8_tflite);

  if (cifar10_model.model == nullptr)
  {
    Serial.println("ERRO: Falha ao carregar modelo");
    return false;
  }

  if (cifar10_model.model->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.printf("ERRO: Versão incompatível: %d vs %d\n",
                  cifar10_model.model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  Serial.println("Modelo carregado com sucesso");
  return true;
}

bool initialize_interpreter()
{
  Serial.println("[2] Inicializando interpretador...");

  cifar10_model.tensor_arena = static_cast<uint8_t *>(
      allocate_memory(CIFAR10Model::kTensorArenaSize));

  if (cifar10_model.tensor_arena == nullptr)
  {
    Serial.printf("ERRO: Falha na alocação de %d bytes\n", CIFAR10Model::kTensorArenaSize);
    return false;
  }

  static tflite::MicroMutableOpResolver<12> op_resolver;
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
  op_resolver.AddDepthwiseConv2D();

  static tflite::MicroInterpreter static_interpreter(
      cifar10_model.model, op_resolver, cifar10_model.tensor_arena, CIFAR10Model::kTensorArenaSize);
  cifar10_model.interpreter = &static_interpreter;

  TfLiteStatus allocate_status = cifar10_model.interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    Serial.printf("ERRO: AllocateTensors falhou (código: %d)\n", allocate_status);
    return false;
  }

  cifar10_model.input_tensor = cifar10_model.interpreter->input(0);
  cifar10_model.output_tensor = cifar10_model.interpreter->output(0);

  if (cifar10_model.input_tensor == nullptr || cifar10_model.output_tensor == nullptr)
  {
    Serial.println("ERRO: Ponteiros de tensor nulos");
    return false;
  }

  Serial.printf("Arena usada: %lu/%d bytes\n",
                cifar10_model.interpreter->arena_used_bytes(), CIFAR10Model::kTensorArenaSize);
  Serial.println("Interpretador inicializado com sucesso");
  return true;
}

bool initialize_cifar10_model()
{
  Serial.println("=== Inicializando Modelo CIFAR-10 ===");

  static tflite::MicroErrorReporter micro_error_reporter;
  cifar10_model.error_reporter = &micro_error_reporter;

  if (!load_model())
  {
    return false;
  }

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


InferenceResult run_inference(const uint8_t *image_data)
{
  InferenceResult result = {-1, 0.0f, false, ""};
  if (!cifar10_model.initialized)
  {
    result.error_message = "Modelo não inicializado";
    return result;
  }
  load_int8_input(image_data);
  if (cifar10_model.interpreter->Invoke() != kTfLiteOk)
  {
    result.error_message = "Falha na inferência";
    return result;
  }
  int best_index = 0;
  int8_t max_score = SCHAR_MIN;
  const int n = cifar10_model.output_tensor->dims->data[1];
  for (int i = 0; i < n; ++i)
    if (cifar10_model.output_tensor->data.int8[i] > max_score)
    {
      max_score = cifar10_model.output_tensor->data.int8[i];
      best_index = i;
    }
  const float os = cifar10_model.output_tensor->params.scale;
  const int32_t ozp = cifar10_model.output_tensor->params.zero_point;
  result.predicted_class = best_index;
  result.confidence = (static_cast<float>(max_score) - ozp) * os;
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


String parse_json_array_streaming(WiFiClient &client, uint8_t *image_array, int remaining_bytes)
{
  // Pular até encontrar '['
  bool found_bracket = false;
  int bytes_read = 0;

  while (client.connected() && bytes_read < remaining_bytes && !found_bracket)
  {
    if (client.available())
    {
      char c = client.read();
      bytes_read++;
      if (c == '[')
        found_bracket = true;
    }
    else
    {
      delay(1);
    }
  }

  if (!found_bracket)
    return "Array '[' não encontrado";

  // Ler valores diretamente
  int pixel_count = 0;
  String number_str = "";
  number_str.reserve(4); // Máximo 255 = 3 dígitos

  while (client.connected() && bytes_read < remaining_bytes)
  {
    if (!client.available())
    {
      delay(1);
      continue;
    }

    char c = client.read();
    bytes_read++;

    if (isdigit(c))
    {
      number_str += c;
    }
    else if ((c == ',' || c == ']') && number_str.length() > 0)
    {
      int value = number_str.toInt();
      image_array[pixel_count++] = constrain(value, 0, 255);
      number_str = "";

      if (c == ']' || pixel_count >= CIFAR10Model::kImageSize)
        break;
    }
  }

  if (pixel_count != CIFAR10Model::kImageSize)
  {
    return "Esperado 27648 pixels, recebido: " + String(pixel_count);
  }

  return ""; // Sucesso
}


static String read_raw(WiFiClient &c, int len, uint8_t *dst) {
  size_t done = 0;
  uint32_t last = millis();
  uint8_t pct_last = 0;
  while (done < (size_t)len) {
    int avail = c.available();
    if (avail > 0) {
      int to_read = std::min(avail, len - (int)done);
      int n = c.read(dst + done, to_read);
      if (n > 0) {
        done += n;
        uint8_t pct = (uint8_t)((done * 100U) / (uint32_t)len);
        if (pct >= pct_last + 5 || done == (size_t)len) {
          pct_last = pct;
          Serial.printf("RX %u%% (%u/%d)\n", pct, (unsigned)done, len);
        }
        last = millis();
      }
    } else {
      if (millis() - last > 5000) return "Timeout de leitura.";
      delay(1);
    }
  }
  return "";
}


void handle_client()
{
  WiFiClient client = server.available();
  if (!client) return;

  client.setNoDelay(true);
  client.setTimeout(30000);

  const uint32_t t_total = millis();
  const IPAddress rip = client.remoteIP();
  const uint16_t rport = client.remotePort();
  Serial.printf(">>> Conectado: %s:%u\n", rip.toString().c_str(), rport);

  String req, line;
  int content_len = 0;
  bool expect_continue = false;

  while (client.connected()) {
    line = client.readStringUntil('\n');
    line.trim();
    if (line.isEmpty()) break;
    if (req.isEmpty()) req = line;
    if (line.startsWith("Content-Length:")) content_len = line.substring(15).toInt();
    if (line.startsWith("Expect:") && line.indexOf("100-continue") >= 0) expect_continue = true;
  }

  if (expect_continue) {
    client.print("HTTP/1.1 100 Continue\r\n\r\n");
    client.flush();
  }

  String ctype = "text/html";
  String body;
  InferenceResult res{-1, 0.0f, false, ""};

  if (req.startsWith("POST /predict_bin")) {
    ctype = "application/json";
    if (content_len != CIFAR10Model::kImageSize) {
      res.success = false;
      res.error_message = "Content-Length inválido";
    } else {
      std::unique_ptr<uint8_t[]> img(new (std::nothrow) uint8_t[CIFAR10Model::kImageSize]);
      if (!img) {
        res.success = false;
        res.error_message = "Falha de memória";
      } else {
        const uint32_t t_rx0 = millis();
        String err = read_raw(client, content_len, img.get());
        const uint32_t t_rx = millis() - t_rx0;
        if (err.length()) {
          res.success = false;
          res.error_message = err;
          Serial.printf("RX falhou em %lums: %s\n", t_rx, err.c_str());
        } else {
          Serial.printf("RX ok em %lums\n", t_rx);
          const uint32_t t_inf0 = millis();
          res = run_inference(img.get());
          Serial.printf("Inferência em %lums\n", millis() - t_inf0);
        }
      }
    }
    body = create_json_response(res);
  } else if (req.startsWith("GET /status")) {
    ctype = "application/json";
    res.success = cifar10_model.initialized;
    body = create_json_response(res);
  } else {
    body = "<!DOCTYPE html><html><body><h1>CIFAR-10 MobileNetV2 API</h1><p><b>POST /predict_bin</b> 27648 B raw</p><p><b>GET /status</b></p></body></html>";
  }

  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: " + ctype);
  client.println("Access-Control-Allow-Origin: *");
  client.println("Connection: close");
  client.println("Content-Length: " + String(body.length()));
  client.println();
  client.print(body);
  client.flush();
  client.stop();

  Serial.printf("<<< Desconectado: %s:%u | Total %lums | Heap %u\n",
                rip.toString().c_str(), rport, millis() - t_total, esp_get_free_heap_size());
}




String parse_json_array(String json_data, uint8_t *image_array)
{
  int start_index = json_data.indexOf("\"pixels\":");
  if (start_index == -1)
    return "Campo 'pixels' não encontrado";

  start_index = json_data.indexOf('[', start_index);
  if (start_index == -1)
    return "Array de pixels não encontrado";

  int end_index = json_data.indexOf(']', start_index);
  if (end_index == -1)
    return "Fim do array não encontrado";

  String array_content = json_data.substring(start_index + 1, end_index);
  int pixel_count = 0;
  int current_pos = 0;

  while (current_pos < array_content.length() && pixel_count < CIFAR10Model::kImageSize)
  {
    while (current_pos < array_content.length() && isspace(array_content.charAt(current_pos)))
    {
      current_pos++;
    }
    if (current_pos >= array_content.length())
      break;

    int comma_pos = array_content.indexOf(',', current_pos);
    String value_str = (comma_pos == -1) ? array_content.substring(current_pos) : array_content.substring(current_pos, comma_pos);
    value_str.trim();

    bool is_valid_number = (value_str.length() > 0);
    for (int i = 0; i < value_str.length() && is_valid_number; i++)
    {
      if (!isdigit(value_str.charAt(i)))
        is_valid_number = false;
    }

    if (!is_valid_number)
    {
      return "Valor inválido no índice " + String(pixel_count) + ": '" + value_str + "'";
    }

    int pixel_value = value_str.toInt();
    image_array[pixel_count] = (uint8_t)constrain(pixel_value, 0, 255);
    pixel_count++;

    if (comma_pos == -1)
      break;
    current_pos = comma_pos + 1;
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

  Serial.println("\n=== CIFAR-10 TensorFlow Lite WiFi API ===");
  Serial.printf("Free heap inicial: %lu bytes\n", esp_get_free_heap_size());
  Serial.printf("PSRAM disponível: %u bytes\n", ESP.getPsramSize());

  if (!connect_wifi())
  {
    Serial.println("Falha na conexão WiFi - reiniciando...");
    ESP.restart();
  }

  if (!initialize_cifar10_model())
  {
    Serial.println("Falha na inicialização do modelo! Parando.");
    while (true)
    {
      delay(1000);
    }
  }

  server.begin();
  Serial.println("\n=== Servidor HTTP iniciado ===");
}

void loop()
{
  if (WiFi.status() != WL_CONNECTED)
  {
    Serial.println("WiFi desconectado - tentando reconectar...");
    connect_wifi();
  }
  handle_client();
  delay(10);
}
