/******************************************************************************
 * Inferência TensorFlow Lite Micro no ESP32-C3
 * Modelo de seno — compatível com FLOAT32 **ou** INT8/UINT8
 * API WiFi para cálculo de seno
 ******************************************************************************/

#include <WiFi.h>
#include <WebServer.h> 

#include <Arduino.h>
#include <math.h>                     // para M_PI

const char* ssid = "Starlink";
const char* password = "diversao";

// Servidor web na porta 80
WebServer server(80);

// ───────── TensorFlow Lite Micro ──────────────────────────────────────────────
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ───────── Modelo convertido (escolha aqui) ──────────────────────────────────
#include "modelo_seno_float32.h"  

#define modelo_seno_tflite      modelo_seno_float32_tflite
#define modelo_seno_tflite_len  modelo_seno_float32_tflite_len

// ───────── Arena de memória do TFLM ──────────────────────────────────────────
constexpr int   kTensorArenaSize = 12 * 1024;   // 12 kB
static   uint8_t tensor_arena[kTensorArenaSize];

// ───────── Variáveis globais ─────────────────────────────────────────────────
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter*     error_reporter = &micro_error_reporter;

  const tflite::Model*       model          = nullptr;
  tflite::AllOpsResolver     resolver;                  // todas as ops
  tflite::MicroInterpreter*  interpreter     = nullptr;

  TfLiteTensor*              input          = nullptr;
  TfLiteTensor*              output         = nullptr;

  // Parâmetros de quantização (válidos só se o modelo for INT8/UINT8)
  float     in_scale   = 1.0f;
  int32_t   in_zp      = 0;
  float     out_scale  = 1.0f;
  int32_t   out_zp     = 0;
  bool      is_quant   = false;
}

// ═════════════════════════════════════════════════════════════════════════════
// FUNÇÃO DE INFERÊNCIA
// ═════════════════════════════════════════════════════════════════════════════
float inferirSeno(float x) {
  // ─── 1. Escreve no tensor de entrada ──────────────────────────────────
  if (is_quant) {
    int32_t q_in = static_cast<int32_t>(roundf(x / in_scale) + in_zp);

    // Clampa se exceder range do tipo
    if (input->type == kTfLiteInt8) {
      q_in = max(-128, min(127, q_in));
      input->data.int8[0] = static_cast<int8_t>(q_in);
    } else {                         // UINT8
      q_in = max(0, min(255, q_in));
      input->data.uint8[0] = static_cast<uint8_t>(q_in);
    }
  } else {
    input->data.f[0] = x;            // modelo float32 original
  }

  // ─── 2. Invoke ─────────────────────────────────────────────────────────
  if (interpreter->Invoke() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke() falhou.");
    return NAN; // Retorna NaN em caso de erro
  }

  // ─── 3. Lê a saída ─────────────────────────────────────────────────────
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

// ═════════════════════════════════════════════════════════════════════════════
// HANDLERS DA API
// ═════════════════════════════════════════════════════════════════════════════

// Handler para calcular seno
void handleSeno() {
  // Verifica se o parâmetro 'angulo' foi enviado
  if (!server.hasArg("angulo")) {
    server.send(400, "application/json", "{\"erro\":\"Parâmetro 'angulo' não encontrado\"}");
    return;
  }

  // Converte o parâmetro para float
  float angulo_graus = server.arg("angulo").toFloat();
  
  // Converte graus para radianos
  float angulo_rad = (angulo_graus * M_PI) / 180.0;
  
  // Faz a inferência
  float seno_resultado = inferirSeno(angulo_rad);
  
  // Verifica se houve erro na inferência
  if (isnan(seno_resultado)) {
    server.send(500, "application/json", "{\"erro\":\"Erro na inferência do modelo\"}");
    Serial.printf("Erro na inferência para ângulo: %.2f graus\n", angulo_graus);
    return;
  }
  
  // Imprime no terminal
  Serial.printf("sin(%.2f°) = %.6f\n", angulo_graus, seno_resultado);
  
  // Monta resposta JSON
  String resposta = "{";
  resposta += "\"angulo_graus\":" + String(angulo_graus, 2) + ",";
  resposta += "\"seno\":" + String(seno_resultado, 6);
  resposta += "}";
  
  // Envia resposta
  server.send(200, "application/json", resposta);
}

// Handler para página de ajuda
void handleRoot() {
  String html = "<html><body>";
  html += "<h1>API de Cálculo de Seno - ESP32</h1>";
  html += "<p>Para calcular o seno de um ângulo, use:</p>";
  html += "<p><strong>GET /seno?angulo=VALOR</strong></p>";
  html += "<p>Exemplo: <a href='/seno?angulo=30'>/seno?angulo=30</a></p>";
  html += "<p>Exemplo: <a href='/seno?angulo=45'>/seno?angulo=45</a></p>";
  html += "<p>Exemplo: <a href='/seno?angulo=90'>/seno?angulo=90</a></p>";
  html += "</body></html>";
  
  server.send(200, "text/html", html);
}

// Handler para 404
void handleNotFound() {
  server.send(404, "application/json", "{\"erro\":\"Endpoint não encontrado\"}");
}

// ═════════════════════════════════════════════════════════════════════════════
// FUNÇÃO DE TESTE INICIAL
// ═════════════════════════════════════════════════════════════════════════════
void testeInicialInferencia() {
  Serial.println("Executando teste inicial de inferência...");
  
  // Ângulos de teste (rad)
  constexpr float angles[] = { M_PI/3, M_PI/6, M_PI/4, M_PI/2, M_PI };
  constexpr int   num_angles = sizeof(angles) / sizeof(angles[0]);
  
  for (int i = 0; i < num_angles; ++i) {
    float x = angles[i];
    float y = inferirSeno(x);
    
    // Verifica se houve erro na inferência
    if (!isnan(y)) {
      float angulo_graus = (x * 180.0) / M_PI;
      Serial.printf("sin(%.2f°) = %.6f\n", angulo_graus, y);
    } else {
      Serial.printf("Erro na inferência para x=%f\n", x);
    }
  }
  Serial.println("----------------------------");
  Serial.println("Teste inicial concluído!\n");
}

// ═════════════════════════════════════════════════════════════════════════════
// SETUP
// ═════════════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(200);

  // 1) Carrega o modelo da flash
  model = tflite::GetModel(modelo_seno_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
        "Modelo v%d ≠ Schema v%d", model->version(), TFLITE_SCHEMA_VERSION);
    while (true);
  }

  // 2) Intérprete + alocação de tensores
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() falhou.");
    while (true);
  }

  // 3) Ponteiros de entrada/saída
  input  = interpreter->input(0);
  output = interpreter->output(0);

  // 4) Checa se é quantizado
  is_quant = (input->type == kTfLiteInt8  || input->type == kTfLiteUInt8);

  if (is_quant) {
    in_scale   = input->params.scale;
    in_zp      = input->params.zero_point;
    out_scale  = output->params.scale;
    out_zp     = output->params.zero_point;
  }

  // 5) Log rápido
  Serial.printf("INPUT  type=%d  scale=%f  zp=%d\n",
                input->type,  in_scale,  in_zp);
  Serial.printf("OUTPUT type=%d  scale=%f  zp=%d\n",
                output->type, out_scale, out_zp);
  Serial.println("Modelo TensorFlow Lite carregado com sucesso!");

  // Conecta ao WiFi
  WiFi.begin(ssid, password);
  Serial.print("Conectando ao WiFi");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.printf("WiFi conectado! IP: %s\n", WiFi.localIP().toString().c_str());

  // Configura rotas da API
  server.on("/", handleRoot);
  server.on("/seno", handleSeno);
  server.onNotFound(handleNotFound);
  
  // Executa teste inicial de inferência
  testeInicialInferencia();

  // Inicia servidor
  server.begin();
  Serial.println("Servidor HTTP iniciado!");
  Serial.println("Use: GET /seno?angulo=VALOR");
  Serial.println("Exemplo: http://" + WiFi.localIP().toString() + "/seno?angulo=30");
}

// ═════════════════════════════════════════════════════════════════════════════
// LOOP
// ═════════════════════════════════════════════════════════════════════════════
void loop() {
  // Processa requisições HTTP
  server.handleClient();
  
  // Pequeno delay para não sobrecarregar
  delay(2);
}