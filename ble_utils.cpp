#include <Arduino.h>
#include "ble_utils.h"

// Include ESP-IDF BLE APIs via extern "C"
extern "C" {
  #include "esp_bt_main.h"
  #include "esp_gatt_common_api.h"
  #include "esp_gap_ble_api.h"
  #include "esp_log.h" // Add ESP logging capabilities
}

// Global BLE variables implementation
bool deviceConnected = false;
uint32_t packet_sequence = 0;
BLEServer *pServer = nullptr;
BLEService *pService = nullptr;
BLECharacteristic *pCharacteristic = nullptr;
BLEAdvertising *pAdvertising = nullptr;

// FreeRTOS task and queue handles
TaskHandle_t bleTaskHandle = nullptr;
QueueHandle_t bleQueue = nullptr;
SemaphoreHandle_t bleMutex = nullptr;
EventGroupHandle_t bleEventGroup = nullptr;
bool isBLECongested = false;

// Add the new global variables after existing declarations at the top of the file
bool isTransmitting = false;
unsigned long transmitStart = 0;

// Error detection variables
bool l2cap_error_detected = false;

// Define a custom ESP log error handler to capture L2CAP errors
static esp_err_t ble_l2cap_error_handler(const char *format, va_list args) {
  // Buffer to store the formatted log message
  char log_buffer[256];
  vsnprintf(log_buffer, sizeof(log_buffer), format, args);
  
  // Check if this is the specific L2CAP error we're looking for
  if (strstr(log_buffer, "l2ble_update_att_acl_pkt_num not found p_tcb") != NULL) {
    l2cap_error_detected = true;
    ESP.restart();
    // Will trigger a BLE stack reset in the next cycle
  }
  
  // Let the ESP logger still process this message
  return ESP_OK;
}

// Install custom log handler for BT component
void install_l2cap_error_handler() {
  esp_log_set_vprintf(ble_l2cap_error_handler);
}

// BLE server callbacks implementation
void MyServerCallbacks::onConnect(BLEServer *pServer) {
  deviceConnected = true;
  packet_sequence = 0; // Reset packet sequence when a new client connects
}

void MyServerCallbacks::onDisconnect(BLEServer *pServer) {
  deviceConnected = false;
  
  // Check if we were in the middle of transmitting when disconnect happened
  if (isTransmitting) {
    resetBLEStack();
    isTransmitting = false;
  } else {
    // Normal disconnection, just restart advertising
    if (bleEventGroup != nullptr) {
      xEventGroupSetBits(bleEventGroup, BLE_CONGESTION_BIT);
    }
    
    // Restart advertising
    pServer->startAdvertising();
  }
}

// BLE characteristic callbacks implementation
void MyCharacteristicCallbacks::onNotify(BLECharacteristic* pCharacteristic) {
  // This is called after a notification is sent
}

void MyCharacteristicCallbacks::onStatus(BLECharacteristic* pCharacteristic, Status status, uint32_t code) {
  // Instead of checking specific enum values, we'll track congestion state with the code
  // Code 0 generally means success, anything else might indicate congestion
  
  if (code == 0) {
    // Success case
    isBLECongested = false;
    if (bleEventGroup != nullptr) {
      xEventGroupClearBits(bleEventGroup, BLE_CONGESTION_BIT);
    }
  } else {
    // Any non-zero code might indicate congestion
    isBLECongested = true;
    if (bleEventGroup != nullptr) {
      xEventGroupSetBits(bleEventGroup, BLE_CONGESTION_BIT);
    }
  }
}

// Initialize BLE
int setupBLE() {
  // Install the error handler to capture L2CAP errors
  install_l2cap_error_handler();
  
  BLEDevice::init("Anhthin's Halo");
  
  // Create server
  pServer = BLEDevice::createServer();
  if (!pServer) {
    return ERR_BLE_INIT;
  }
  
  pServer->setCallbacks(new MyServerCallbacks());
  
  // Create service
  pService = pServer->createService(SERVICE_UUID);
  if (!pService) {
    return ERR_BLE_INIT;
  }
  
  // Create characteristic for sending data
  pCharacteristic = pService->createCharacteristic(
      CHARACTERISTIC_UUID,
      BLECharacteristic::PROPERTY_READ | 
      BLECharacteristic::PROPERTY_WRITE |
      BLECharacteristic::PROPERTY_NOTIFY
  );
  
  if (!pCharacteristic) {
    return ERR_BLE_INIT;
  }
  
  // Set up callback to handle congestion events
  pCharacteristic->setCallbacks(new MyCharacteristicCallbacks());
  
  // Add descriptor for client characteristic configuration
  BLEDescriptor *pDescriptor = new BLEDescriptor(BLEUUID((uint16_t)0x2902));
  uint8_t descriptorValue[2] = {0, 0};
  pDescriptor->setValue(descriptorValue, 2);
  pCharacteristic->addDescriptor(pDescriptor);
  
  // Set initial value
  pCharacteristic->setValue("Ready");
  
  // Start service
  pService->start();
  
  // Start advertising
  pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  BLEDevice::startAdvertising();
  
  // Create FreeRTOS queue for BLE transmission requests
  bleQueue = xQueueCreate(BLE_QUEUE_SIZE, sizeof(BLETransmitRequest));
  if (bleQueue == nullptr) {
    return ERR_BLE_INIT;
  }
  
  // Create mutex for BLE operations
  bleMutex = xSemaphoreCreateMutex();
  if (bleMutex == nullptr) {
    return ERR_BLE_INIT;
  }
  
  // Create event group for BLE signaling
  bleEventGroup = xEventGroupCreate();
  if (bleEventGroup == nullptr) {
    return ERR_BLE_INIT;
  }
  
  // Create BLE transmission task
  BaseType_t taskCreated = xTaskCreatePinnedToCore(
    bleTransmissionTask,     // Task function
    "BLE_TX_Task",           // Name
    BLE_TASK_STACK_SIZE,     // Stack size
    NULL,                    // Parameters
    BLE_TASK_PRIORITY,       // Priority
    &bleTaskHandle,          // Task handle
    1                        // Core (1 = separate from Arduino loop)
  );
  
  if (taskCreated != pdPASS) {
    return ERR_BLE_INIT;
  }
  
  delay(1000);
  return ERR_OK;
}

// Calculate checksum for BLE packet validation
uint8_t calculateChecksum(uint8_t* data, size_t length) {
  uint8_t checksum = 0;
  for (size_t i = 0; i < length; i++) {
    checksum ^= data[i];  // XOR all bytes
  }
  return checksum;
}

// Compress RGB888 buffer to JPEG using ESP32's hardware acceleration
bool compressRGBtoJPEG(uint8_t* rgb_buffer, int width, int height, 
                      uint8_t** jpg_buffer, size_t* jpg_len, int quality) {
  
  // For 3 channel RGB format, we use the RGB888 converter
  bool success = fmt2jpg(rgb_buffer, width * height * 3, width, height, 
                        PIXFORMAT_RGB888, quality, jpg_buffer, jpg_len);
  
  if (!success) {
    return false;
  }
  
  return true;
}

// Send the processed image with rectangles over BLE
void sendImageWithRectangles(uint8_t *rgb_buffer, int width, int height) {
  // Only proceed if a device is connected
  if (!deviceConnected) {
    return;
  }
  
  // Convert RGB to JPEG for transmission efficiency
  uint8_t* jpg_buffer = nullptr;
  size_t jpg_len = 0;
  
  if (!compressRGBtoJPEG(rgb_buffer, width, height, &jpg_buffer, &jpg_len)) {
    return;
  }
  
  // Calculate the actual data bytes we can send per packet (accounting for packet header)
  const size_t DATA_SIZE = MAX_MTU_SIZE - PACKET_HEADER_SIZE;
  
  // Create a packet buffer
  uint8_t* packet_buffer = (uint8_t*)malloc(MAX_MTU_SIZE);
  if (!packet_buffer) {
    free(jpg_buffer);
    return;
  }
  
  // Send the JPEG in chunks
  size_t offset = 0;
  int packetCount = 0;
  
  // Record transmission start time
  unsigned long transmitStartTime = millis();
  
  while (offset < jpg_len) {
    packetCount++;
    
    // Calculate how much data to send in this packet
    size_t chunkSize = (jpg_len - offset) > DATA_SIZE ? DATA_SIZE : (jpg_len - offset);
    
    // Packet format:
    // Bytes 0-3: 32-bit packet sequence number (big-endian)
    // Byte 4: Checksum of the data
    // Bytes 5+: Actual JPEG data
    
    // Add sequence number
    packet_buffer[0] = (packet_sequence >> 24) & 0xFF;
    packet_buffer[1] = (packet_sequence >> 16) & 0xFF;
    packet_buffer[2] = (packet_sequence >> 8) & 0xFF;
    packet_buffer[3] = packet_sequence & 0xFF;
    
    // Copy image data after the header
    memcpy(packet_buffer + PACKET_HEADER_SIZE, jpg_buffer + offset, chunkSize);
    
    // Calculate checksum
    packet_buffer[4] = calculateChecksum(jpg_buffer + offset, chunkSize);
    
    // Send the packet
    pCharacteristic->setValue(packet_buffer, PACKET_HEADER_SIZE + chunkSize);
    pCharacteristic->notify();
    
    // Update offset and packet sequence
    offset += chunkSize;
    packet_sequence++;
    
    delay(20);
  }
  
  // Free buffers
  free(packet_buffer);
  free(jpg_buffer);
}

// Convert RGB565 to RGB888
void convertRGB565toRGB888(uint8_t* rgb565_buffer, uint8_t* rgb888_buffer, int width, int height) {
  for (int i = 0; i < width * height; i++) {
    // Get RGB565 pixel (2 bytes)
    uint16_t pixel = (rgb565_buffer[i * 2] << 8) | rgb565_buffer[i * 2 + 1];
    
    // Extract RGB components
    uint8_t r = ((pixel >> 11) & 0x1F) << 3;  // 5 bits to 8 bits
    uint8_t g = ((pixel >> 5) & 0x3F) << 2;   // 6 bits to 8 bits
    uint8_t b = (pixel & 0x1F) << 3;          // 5 bits to 8 bits
    
    // Store RGB888 pixel (3 bytes)
    rgb888_buffer[i * 3] = r;     // Red
    rgb888_buffer[i * 3 + 1] = g; // Green
    rgb888_buffer[i * 3 + 2] = b; // Blue
  }
}

// Send RGB888 buffer directly over BLE
void sendRGB888(uint8_t* rgb888_buffer, int width, int height) {
  // Check if queue exists and clear any pending frames
  if (bleQueue != nullptr) {
    // Use mutex to safely clear the queue
    if (bleMutex != nullptr && xSemaphoreTake(bleMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
      // Check if queue already has items
      UBaseType_t itemsInQueue = uxQueueMessagesWaiting(bleQueue);
      if (itemsInQueue > 0) {
        // Clear the queue by removing all existing items
        BLETransmitRequest dummyRequest;
        while (xQueueReceive(bleQueue, &dummyRequest, 0) == pdTRUE) {
          // Free buffer if needed to avoid memory leaks
          if (dummyRequest.free_after_send) {
            free(dummyRequest.data);
          }
        }
      }
      xSemaphoreGive(bleMutex); // Release the mutex
    }
  }

  sendRGB888Async(rgb888_buffer, width, height, false);
}

// Send JPEG buffer directly over BLE
void sendJPEG(uint8_t* jpg_buffer, size_t jpg_len) {
  // Set the transmission flag and start time
  isTransmitting = true;
  transmitStart = millis();
  
  // Check if queue exists
  if (bleQueue != nullptr) {
    // Use mutex to safely clear the queue
    if (bleMutex != nullptr && xSemaphoreTake(bleMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
      // Check if queue already has items
      UBaseType_t itemsInQueue = uxQueueMessagesWaiting(bleQueue);
      if (itemsInQueue > 0) {
        // Clear the queue by removing all existing items
        BLETransmitRequest dummyRequest;
        while (xQueueReceive(bleQueue, &dummyRequest, 0) == pdTRUE) {
          // Free buffer if needed to avoid memory leaks
          if (dummyRequest.free_after_send) {
            free(dummyRequest.data);
          }
        }
      }
      xSemaphoreGive(bleMutex); // Release the mutex
    }
  }
  
  if (!sendJPEGAsync(jpg_buffer, jpg_len, false)) {
    isTransmitting = false; // Clear flag on immediate failure
  }
}

// FreeRTOS task for handling BLE transmissions
void bleTransmissionTask(void* parameter) {
  BLETransmitRequest request;
  uint16_t conn_id;
  uint8_t* packet_buffer = nullptr;
  const size_t DATA_SIZE = MAX_MTU_SIZE - PACKET_HEADER_SIZE;
  const TickType_t xDelay = pdMS_TO_TICKS(5); // Short delay between congestion checks
  
  // Allocate the packet buffer once for the lifetime of the task
  packet_buffer = (uint8_t*)malloc(MAX_MTU_SIZE);
  if (packet_buffer == nullptr) {
    vTaskDelete(NULL);
    return;
  }
  
  for (;;) {
    // Check for detected L2CAP errors from the log handler
    if (l2cap_error_detected) {
      ESP.restart();
      l2cap_error_detected = false; // Clear the error flag
      isTransmitting = false;       // Ensure transmission flag is cleared
      delay(100);                   // Short delay after reset
      continue;
    }
    
    // Wait for a transmission request with a short timeout so we can check for errors
    if (xQueueReceive(bleQueue, &request, pdMS_TO_TICKS(1000)) == pdTRUE) {
      
      // Check if we have a device connected
      if (!deviceConnected) {
        if (request.free_after_send) {
          free(request.data);
        }
        isTransmitting = false;
        continue;
      }
      
      // Get connection ID for buffer checks
      conn_id = pServer->getConnId();
      
      // First, negotiate MTU size for better throughput
      esp_ble_gatt_set_local_mtu(MAX_MTU_SIZE);
      
      // Send data in chunks
      size_t offset = 0;
      int packetCount = 0;
      unsigned long startTime = millis();
      
      // Take mutex to ensure exclusive access to BLE
      if (xSemaphoreTake(bleMutex, portMAX_DELAY) == pdTRUE) {
        // Reset congestion flag at start of transmission
        isBLECongested = false;
        if (bleEventGroup != nullptr) {
          xEventGroupClearBits(bleEventGroup, BLE_CONGESTION_BIT);
        }
        
        // Process all data in this transmission
        while (offset < request.length) {
          // Check for timeout during transmission (communication error)
          if (millis() - startTime > BLE_TRANSMISSION_TIMEOUT) {
            xSemaphoreGive(bleMutex);
            resetBLEStack();
            isTransmitting = false;
            // Free buffer if needed
            if (request.free_after_send) {
              free(request.data);
            }
            break;
          }
          
          // If congested, wait for congestion to clear
          if (isBLECongested) {
            // If event group exists, wait with timeout for congestion to clear
            if (bleEventGroup != nullptr) {
              EventBits_t bits = xEventGroupWaitBits(
                  bleEventGroup,        // Event group
                  BLE_CONGESTION_BIT,   // Bits to wait for
                  pdTRUE,               // Clear bits on exit
                  pdFALSE,              // Wait for all bits (not used)
                  xDelay                // Timeout
              );
              
              // If still congested, just wait a bit
              if ((bits & BLE_CONGESTION_BIT) != 0) {
                taskYIELD();
                continue;
              }
            } else {
              // If no event group, just use a short delay
              vTaskDelay(pdMS_TO_TICKS(5));
            }
            
            // Congestion cleared or timed out, try again
            isBLECongested = false;
          }
          
          // Check how many packets we can send at once (max throughput approach)
          uint16_t free_buff_num = 0;
          
          // Try to use ESP-IDF function to get exact buffer count
          try {
            free_buff_num = esp_ble_get_cur_sendable_packets_num(conn_id);
          } catch (...) {
            // If the function isn't available, assume we can send 1 packet
            free_buff_num = 1;
          }
          
          // If no buffers available, check congestion or yield
          if (free_buff_num == 0) {
            // Just yield to other tasks for a moment
            taskYIELD();
            continue;
          }
          
          // For safety, limit maximum batch size to avoid potential issues
          if (free_buff_num > 10) {
            free_buff_num = 10;
          }
          
          // Send as many packets as possible with the available buffers
          for (int i = 0; i < free_buff_num && offset < request.length; i++) {
            // Calculate chunk size for this packet
            size_t chunkSize = (request.length - offset) > DATA_SIZE ? DATA_SIZE : (request.length - offset);
            
            // Prepare packet header
            packet_buffer[0] = (packet_sequence >> 24) & 0xFF;
            packet_buffer[1] = (packet_sequence >> 16) & 0xFF;
            packet_buffer[2] = (packet_sequence >> 8) & 0xFF;
            packet_buffer[3] = packet_sequence & 0xFF;
            packet_buffer[4] = calculateChecksum(request.data + offset, chunkSize);
            
            // Copy data to packet buffer
            memcpy(packet_buffer + PACKET_HEADER_SIZE, request.data + offset, chunkSize);
            
            // Send the packet immediately without checking again
            pCharacteristic->setValue(packet_buffer, PACKET_HEADER_SIZE + chunkSize);
            pCharacteristic->notify();
            
            // Check if we got congested after sending
            if (isBLECongested) {
              // Break the loop and wait for congestion to clear
              break;
            }
            
            // Update state
            offset += chunkSize;
            packet_sequence++;
            packetCount++;
            
            // If we've sent all data, break the loop
            if (offset >= request.length) {
              break;
            }
          }
        }
        
        // Release mutex
        xSemaphoreGive(bleMutex);
        
        // Clear transmission flag on completion
        isTransmitting = false;
      }
      
      // Free the data buffer if requested
      if (request.free_after_send) {
        free(request.data);
      }
    }
  }
  
  // Clean up (this should never execute as the task runs forever)
  free(packet_buffer);
  vTaskDelete(NULL);
}

// Generic function to queue an image for BLE transmission
bool queueImageForBLE(uint8_t* buffer, size_t length, int width, int height, ImageFormat format, bool free_after_send) {
  if (!buffer || length == 0) {
    return false;
  }
  
  // Check if queue exists
  if (bleQueue == nullptr) {
    return false;
  }
  
  // Create request structure
  BLETransmitRequest request;
  request.data = buffer;
  request.length = length;
  request.width = width;
  request.height = height;
  request.format = format;
  request.free_after_send = free_after_send;
  
  bool result = false;
  
  // Use mutex to safely add to queue
  if (bleMutex != nullptr && xSemaphoreTake(bleMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
    // Add to queue with timeout
    result = (xQueueSend(bleQueue, &request, 0) == pdTRUE);
    xSemaphoreGive(bleMutex); // Release the mutex
  }
  
  if (!result) {
    return false;
  }
  
  return true;
}

// Queue a JPEG image for async BLE transmission
bool sendJPEGAsync(uint8_t* jpg_buffer, size_t jpg_len, bool free_after_send) {
  return queueImageForBLE(jpg_buffer, jpg_len, 0, 0, FORMAT_JPEG, free_after_send);
}

// Queue an RGB565 image for async BLE transmission
bool sendRGB565Async(uint8_t* rgb565_buffer, int width, int height, bool free_after_send) {
  return queueImageForBLE(rgb565_buffer, width * height * 2, width, height, FORMAT_RGB565, free_after_send);
}

// Queue an RGB888 image for async BLE transmission
bool sendRGB888Async(uint8_t* rgb888_buffer, int width, int height, bool free_after_send) {
  return queueImageForBLE(rgb888_buffer, width * height * 3, width, height, FORMAT_RGB888, free_after_send);
}

// Compress RGB to JPEG and queue for async transmission
bool compressAndSendJPEGAsync(uint8_t* rgb_buffer, int width, int height, int quality) {
  // First check if the queue is not initialized
  if (bleQueue == nullptr) {
    return false;
  }
  
  // Use mutex to safely clear the queue
  if (bleMutex != nullptr && xSemaphoreTake(bleMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
    // Clear any pending frames in the queue
    UBaseType_t itemsInQueue = uxQueueMessagesWaiting(bleQueue);
    if (itemsInQueue > 0) {
      // Clear the queue by removing all existing items
      BLETransmitRequest dummyRequest;
      while (xQueueReceive(bleQueue, &dummyRequest, 0) == pdTRUE) {
        // Free buffer if needed to avoid memory leaks
        if (dummyRequest.free_after_send) {
          free(dummyRequest.data);
        }
      }
    }
    xSemaphoreGive(bleMutex); // Release the mutex
  }
  
  uint8_t* jpg_buffer = nullptr;
  size_t jpg_len = 0;
  
  // Compress to JPEG first
  bool success = compressRGBtoJPEG(rgb_buffer, width, height, &jpg_buffer, &jpg_len, quality);
  
  if (!success || jpg_buffer == nullptr) {
    return false;
  }
  
  // Queue the compressed JPEG (and set free_after_send to true since we own this buffer)
  if (!sendJPEGAsync(jpg_buffer, jpg_len, true)) {
    free(jpg_buffer);
    return false;
  }
  
  return true;
}

// Implement resetBLEStack function
void resetBLEStack() {
  // Stop any ongoing tasks first
  if (bleTaskHandle != NULL) {
    vTaskSuspend(bleTaskHandle);
  }
  
  // Reset flags
  isTransmitting = false;
  isBLECongested = false;
  l2cap_error_detected = false;
  
  // Deinitialize BLE
  BLEDevice::deinit(true);
  delay(200);  // Give some time for the stack to clean up
  
  // Reinitialize BLE
  BLEDevice::init("Anhthin's Halo");
  
  // Create a new server
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());
  
  // Create service
  pService = pServer->createService(SERVICE_UUID);
  
  // Create characteristic
  pCharacteristic = pService->createCharacteristic(
      CHARACTERISTIC_UUID,
      BLECharacteristic::PROPERTY_READ | 
      BLECharacteristic::PROPERTY_WRITE |
      BLECharacteristic::PROPERTY_NOTIFY
  );
  
  // Set up callback
  pCharacteristic->setCallbacks(new MyCharacteristicCallbacks());
  
  // Add descriptor
  BLEDescriptor *pDescriptor = new BLEDescriptor(BLEUUID((uint16_t)0x2902));
  uint8_t descriptorValue[2] = {0, 0};
  pDescriptor->setValue(descriptorValue, 2);
  pCharacteristic->addDescriptor(pDescriptor);
  
  // Set initial value
  pCharacteristic->setValue("YOLO Face Detection Ready");
  
  // Start service
  pService->start();
  
  // Start advertising
  pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  BLEDevice::startAdvertising();
  
  // Reinstall error handler
  install_l2cap_error_handler();
  
  // Resume BLE task if it exists
  if (bleTaskHandle != NULL) {
    vTaskResume(bleTaskHandle);
  }
}

// Send RGB565 buffer directly over BLE
void sendRGB565(uint8_t* rgb565_buffer, int width, int height) {
  // Check if queue exists and clear any pending frames
  if (bleQueue != nullptr) {
    // Use mutex to safely clear the queue
    if (bleMutex != nullptr && xSemaphoreTake(bleMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
      // Check if queue already has items
      UBaseType_t itemsInQueue = uxQueueMessagesWaiting(bleQueue);
      if (itemsInQueue > 0) {
        // Clear the queue by removing all existing items
        BLETransmitRequest dummyRequest;
        while (xQueueReceive(bleQueue, &dummyRequest, 0) == pdTRUE) {
          // Free buffer if needed to avoid memory leaks
          if (dummyRequest.free_after_send) {
            free(dummyRequest.data);
          }
        }
      }
      xSemaphoreGive(bleMutex); // Release the mutex
    }
  }

  sendRGB565Async(rgb565_buffer, width, height, false);
} 