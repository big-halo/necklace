#ifndef BLE_UTILS_H
#define BLE_UTILS_H

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <esp_system.h>
#include "esp_camera.h"  // For JPEG conversion
#include "img_converters.h"  // For JPEG conversion
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/event_groups.h"

// If using ESP-IDF version, include low-level BLE APIs
#ifdef ESP_IDF_VERSION_VAL
#include "esp_gatt_common_api.h"
#endif

// Add ESP-IDF specific function declarations
extern "C" {
  // These functions will be used from ESP-IDF
  esp_err_t esp_ble_gatt_set_local_mtu(uint16_t mtu);
}

// Error codes
#define ERR_OK 0
#define ERR_BLE_INIT -1

// BLE Service and Characteristic UUIDs
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// Maximum packet size for BLE data transmission
#define MAX_MTU_SIZE 512  // Maximum theoretical MTU size for ESP32 BLE
#define PACKET_HEADER_SIZE 5  // 4 bytes sequence + 1 byte checksum

// BLE task configuration
#define BLE_TASK_STACK_SIZE 4096
#define BLE_TASK_PRIORITY 10
#define BLE_QUEUE_SIZE 3  // Queue size for transmission requests

// Event bits for signaling
#define BLE_CONGESTION_BIT (1 << 0)
#define BLE_NOTIFY_ENABLED_BIT (1 << 1)

// Maximum time for transmission (ms)
#define BLE_TRANSMISSION_TIMEOUT 5000

// Image format types
enum ImageFormat {
  FORMAT_JPEG,
  FORMAT_RGB565,
  FORMAT_RGB888
};

// Structure for image transmission request
typedef struct {
  uint8_t* data;
  size_t length;
  int width;
  int height;
  ImageFormat format;
  bool free_after_send;  // Whether to free the data buffer after sending
} BLETransmitRequest;

// External declarations of global variables
extern bool deviceConnected;
extern uint32_t packet_sequence;
extern BLEServer *pServer;
extern BLEService *pService;
extern BLECharacteristic *pCharacteristic;
extern BLEAdvertising *pAdvertising;
extern TaskHandle_t bleTaskHandle;
extern QueueHandle_t bleQueue;
extern SemaphoreHandle_t bleMutex;
extern EventGroupHandle_t bleEventGroup;
extern bool isBLECongested;
extern bool isTransmitting;         // Flag to track if transmission is in progress
extern unsigned long transmitStart; // Timestamp when transmission started
extern bool l2cap_error_detected;   // Flag to track L2CAP errors

// Class for BLE server callbacks
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *pServer);
  void onDisconnect(BLEServer *pServer);
};

// Class for BLE characteristic callbacks to handle congestion
class MyCharacteristicCallbacks: public BLECharacteristicCallbacks {
  void onNotify(BLECharacteristic* pCharacteristic);
  void onStatus(BLECharacteristic* pCharacteristic, Status status, uint32_t code);
};

// Function declarations
int setupBLE();
void resetBLEStack();
void install_l2cap_error_handler();
uint8_t calculateChecksum(uint8_t* data, size_t length);
bool compressRGBtoJPEG(uint8_t* rgb_buffer, int width, int height, 
                      uint8_t** jpg_buffer, size_t* jpg_len, int quality = 60);
void convertRGB565toRGB888(uint8_t* rgb565_buffer, uint8_t* rgb888_buffer, int width, int height);

// FreeRTOS task for handling BLE transmissions
void bleTransmissionTask(void* parameter);

// New queue-based transmission functions
bool queueImageForBLE(uint8_t* buffer, size_t length, int width, int height, ImageFormat format, bool free_after_send = true);
bool sendJPEGAsync(uint8_t* jpg_buffer, size_t jpg_len, bool free_after_send = true);
bool sendRGB565Async(uint8_t* rgb565_buffer, int width, int height, bool free_after_send = true);
bool sendRGB888Async(uint8_t* rgb888_buffer, int width, int height, bool free_after_send = true);
bool compressAndSendJPEGAsync(uint8_t* rgb_buffer, int width, int height, int quality = 60);

// Legacy synchronous transmission functions
void sendRGB888(uint8_t* rgb888_buffer, int width, int height);
void sendRGB565(uint8_t* rgb565_buffer, int width, int height);
void sendJPEG(uint8_t* jpg_buffer, size_t jpg_len);

#endif 