#ifndef MODEL_UTILS_H
#define MODEL_UTILS_H

#include "esp_camera.h"
#include "esp_heap_caps.h"

// Error code definitions
#define ERR_OK 0
#define ERR_CAMERA_INIT 5

// Function declarations
int setupCamera();

#endif // MODEL_UTILS_H 