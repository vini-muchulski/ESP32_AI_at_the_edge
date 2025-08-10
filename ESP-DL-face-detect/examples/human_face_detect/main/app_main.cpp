#include <string.h>
#include <vector>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include <lwip/netdb.h>

#include "human_face_detect.hpp"

// --- CONFIGURAÇÃO DE REDE ---
#define WIFI_SSID      "REDE WIFI"
#define WIFI_PASS      "PASSWORD"
#define TCP_PORT       3333

static const char *TAG = "FACE_DETECT_WIFI";

// --- Lógica de Detecção (refatorada do original) ---
static void run_face_detection(const uint8_t* image_data, const size_t image_len)
{
    dl::image::jpeg_img_t jpeg_img = {
        .data = (void*)image_data,
        .data_len = image_len
    };

    // Decodifica a imagem JPEG para o formato RGB888
    auto img = sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
    if (!img.data) {
        ESP_LOGE(TAG, "Falha ao decodificar imagem JPEG.");
        return;
    }

    HumanFaceDetect *detect = new HumanFaceDetect();
    if (!detect) {
        ESP_LOGE(TAG, "Falha ao instanciar HumanFaceDetect.");
        heap_caps_free(img.data);
        return;
    }
    ESP_LOGI(TAG, "  ------------------------------- ");
    ESP_LOGI(TAG, "    ");
    ESP_LOGI(TAG, "    ");
    ESP_LOGI(TAG, "  ------------------------------- ");
    ESP_LOGI(TAG, "Iniciando detecção de rostos...");
    ESP_LOGI(TAG, "  ------------------------------- ");
    ESP_LOGI(TAG, "    ");
    ESP_LOGI(TAG, "    ");

    auto &detect_results = detect->run(img);
    ESP_LOGI(TAG, "Encontrados %d rostos.", detect_results.size());

    for (const auto &res : detect_results) {
        ESP_LOGI(TAG, "[score: %.2f, x1: %d, y1: %d, x2: %d, y2: %d]",
                 res.score, res.box[0], res.box[1], res.box[2], res.box[3]);
    }
    
    delete detect;
    heap_caps_free(img.data);
}

// --- Lógica do Servidor TCP ---
static void tcp_server_task(void *pvParameters)
{
    char addr_str[128];
    int addr_family = AF_INET;
    int ip_protocol = 0;
    struct sockaddr_in6 dest_addr;

    struct sockaddr_in *dest_addr_ip4 = (struct sockaddr_in *)&dest_addr;
    dest_addr_ip4->sin_addr.s_addr = htonl(INADDR_ANY);
    dest_addr_ip4->sin_family = AF_INET;
    dest_addr_ip4->sin_port = htons(TCP_PORT);
    ip_protocol = IPPROTO_IP;

    int listen_sock = socket(addr_family, SOCK_STREAM, ip_protocol);
    if (listen_sock < 0) {
        ESP_LOGE(TAG, "Falha ao criar socket: errno %d", errno);
        vTaskDelete(NULL);
        return;
    }

    int err = bind(listen_sock, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
    if (err != 0) {
        ESP_LOGE(TAG, "Falha no bind do socket: errno %d", errno);
        goto CLEAN_UP;
    }

    err = listen(listen_sock, 1);
    if (err != 0) {
        ESP_LOGE(TAG, "Erro no listen do socket: errno %d", errno);
        goto CLEAN_UP;
    }
    ESP_LOGI(TAG, "Socket escutando na porta %d", TCP_PORT);

    while (1) {
        struct sockaddr_storage source_addr;
        socklen_t addr_len = sizeof(source_addr);
        int sock = accept(listen_sock, (struct sockaddr *)&source_addr, &addr_len);
        if (sock < 0) {
            ESP_LOGE(TAG, "Falha ao aceitar conexão: errno %d", errno);
            break;
        }

        inet_ntoa_r(((struct sockaddr_in *)&source_addr)->sin_addr, addr_str, sizeof(addr_str) - 1);
        ESP_LOGI(TAG, "Socket aceito de: %s", addr_str);
        
        std::vector<uint8_t> rx_buffer;
        rx_buffer.reserve(100 * 1024); // Pré-aloca 100KB para evitar realocações
        char temp_buf[128];
        int len;

        do {
            len = recv(sock, temp_buf, sizeof(temp_buf), 0);
            if (len > 0) {
                rx_buffer.insert(rx_buffer.end(), temp_buf, temp_buf + len);
            }
        } while (len > 0);

        if (len < 0) {
            ESP_LOGE(TAG, "Erro na recepção: errno %d", errno);
        } else { // len == 0, conexão fechada pelo cliente
            ESP_LOGI(TAG, "Conexão fechada. Total de bytes recebidos: %d", rx_buffer.size());
            if (!rx_buffer.empty()) {
                run_face_detection(rx_buffer.data(), rx_buffer.size());
            }
        }

        shutdown(sock, 0);
        close(sock);
    }

CLEAN_UP:
    close(listen_sock);
    vTaskDelete(NULL);
}

// --- Lógica de Conexão Wi-Fi ---
static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1

static void event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        ESP_LOGI(TAG, "Falha ao conectar ao AP. Tentando novamente...");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Conectado. IP: " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

void wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL, &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL, &instance_got_ip));

    wifi_config_t wifi_config = {};
    strcpy((char*)wifi_config.sta.ssid, WIFI_SSID);
    strcpy((char*)wifi_config.sta.password, WIFI_PASS);
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
            WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
            pdFALSE,
            pdFALSE,
            portMAX_DELAY);

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Conectado ao AP SSID:%s", WIFI_SSID);
    } else if (bits & WIFI_FAIL_BIT) {
        ESP_LOGW(TAG, "Falha ao conectar ao SSID:%s", WIFI_SSID);
    } else {
        ESP_LOGE(TAG, "EVENTO INESPERADO");
    }
}

extern "C" void app_main(void)
{
    // Inicializa NVS (necessário para o Wi-Fi)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    ESP_LOGI(TAG, "    ");
    ESP_LOGI(TAG, "    ");
    ESP_LOGI(TAG, "  ------------------------------- ");
    ESP_LOGI(TAG, "Inicializando conexão Wi-Fi...");
    ESP_LOGI(TAG, "  ------------------------------- ");
    ESP_LOGI(TAG, "    ");
    ESP_LOGI(TAG, "    ");
    
    
    wifi_init_sta();

    xTaskCreate(tcp_server_task, "tcp_server", 4096, NULL, 5, NULL);
}