#include <Arduino.h>
#include "model_utils.h"
#define CAMERA_MODEL_XIAO_ESP32S3
#include "camera_pins.h"
#include "esp_camera.h"  // Include ESP32 camera library
#include "img_converters.h"  // Include the image converter library

int setupCamera()
{
  // Add retry mechanism 
  const int MAX_RETRIES = 3;
  int retry_count = 0;
  esp_err_t err = ESP_FAIL;
  
  while (retry_count < MAX_RETRIES && err != ESP_OK) {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 10000000;
    config.pixel_format = PIXFORMAT_JPEG;  // Start with JPEG format
    config.frame_size = FRAMESIZE_SVGA;
    config.fb_count = 2;
    config.jpeg_quality = 12;

    err = esp_camera_init(&config);
    
    if (err == ESP_OK) {
      break;  // Exit the loop if successful
    } else {
      retry_count++;
      
      // Wait a bit before retrying
      delay(500);
    }
  }
  
  if (err != ESP_OK) {
    return ERR_CAMERA_INIT;
  }

  // Get sensor info to identify camera model
  sensor_t *s = esp_camera_sensor_get();
  if (s)
  {
    // Check if this is specifically an OV2640
    bool is_ov2640 = (s->id.PID == 0x2640);
    
    // Apply settings only if it's an OV2640
    if (is_ov2640) {
      // Basic image quality settings
      s->set_brightness(s, 1);               // Increase brightness slightly
      s->set_contrast(s, 1);                 // Increase contrast slightly
      s->set_saturation(s, 1);               // Increase saturation
      s->set_gainceiling(s, GAINCEILING_4X); // Increase gain ceiling

      // Optimize for face detection
      s->set_quality(s, 20);   // Higher quality JPEG
      s->set_sharpness(s, 3); // Increase sharpness
      s->set_denoise(s, 1);   // Enable denoise to improve face clarity

      // Faster shutter speed for better motion capture
      s->set_exposure_ctrl(s, 1); // Enable auto exposure control
      s->set_aec_value(s, 300);   // Faster shutter speed
      s->set_gain_ctrl(s, 1);     // Enable auto gain control
      s->set_awb_gain(s, 1);      // Enable Auto White Balance gain
    }
  }

  return ERR_OK;
}

