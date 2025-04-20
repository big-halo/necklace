#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#define CAMERA_MODEL_XIAO_ESP32S3
#include "camera_pins.h"
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "tensorflow/lite/micro/debug_log.h"
#include "esp_sleep.h"
#include "driver/rtc_io.h"

// Include our utility headers
#include "model_utils.h"
#include "ble_utils.h" // This provides 'deviceConnected' variable used for BLE connection state

// Define states for the state machine
typedef enum
{
  STATE_START,
  STATE_DEEP_SLEEP,
  STATE_SETUP_ADVERTISE,
  STATE_CONNECTED,
} SystemState;

// Current state of the system
SystemState currentState = STATE_START;

// Sleep settings
#define uS_TO_S_FACTOR 1000000ULL                   // Conversion factor for micro seconds to seconds
#define TIME_TO_SLEEP 5                             // Time ESP32 will go to sleep (in seconds)
RTC_DATA_ATTR int bootCount = 0;                    // Stores number of boots in RTC memory
RTC_DATA_ATTR SystemState savedState = STATE_START; // Remember state across deep sleep

// Button pin: change to match your board's pin
#define BUTTON_PIN D1

// DEBUG flag to bypass face validation checks
#define DEBUG_BYPASS_VALIDATION false

// Add debug mode flag to bypass deep sleep at start
#define DEBUG_BYPASS_DEEPSLEEP false

// Add debug mode flag to bypass deep sleep on disconnect and return to advertising
#define DEBUG_ADVERTISING_ON_DISCONNECT true

// Counter for saved images
int image_counter = 0;

// Flag to track if system is ready
bool systemInitialized = false;

// Flag to track if camera is initialized
bool cameraInitialized = false;

// Add these at the top with your other global variables
#define DEBOUNCE_MS 50
uint8_t lastButtonState = HIGH; // Because of pull-up, idle is HIGH
uint32_t lastDebounceTime = 0;

// Add this to your global variables section
volatile bool buttonPressed = false;

// Button interrupt handler function
void IRAM_ATTR buttonISR()
{
  static unsigned long lastInterruptTime = 0;
  unsigned long interruptTime = millis();

  // If interrupts come faster than 200ms, assume it's a bounce and ignore
  if (interruptTime - lastInterruptTime > 200)
  {
    buttonPressed = true;
  }
  lastInterruptTime = interruptTime;
}

// Function to enter deep sleep
void enterDeepSleep()
{
  digitalWrite(LED_BUILTIN, HIGH); // Turn off LED before sleep

  // Update saved state before sleep
  savedState = STATE_DEEP_SLEEP;

  // Stop BLE if it was initialized
  if (currentState >= STATE_SETUP_ADVERTISE)
  {
    BLEDevice::deinit(true);
  }

  // Put camera in software standby and deinitialize if it was initialized
  if (cameraInitialized)
  {
    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor)
    {
      sensor->set_reg(sensor, 0x3008, 0xFF, 0x42);
      delay(100);
    }

    // Deinitialize camera
    esp_camera_deinit();
    cameraInitialized = false;
  }

  // Configure GPIO for external wakeup (using D1 button)
  esp_sleep_enable_ext0_wakeup((gpio_num_t)BUTTON_PIN, LOW);

  // Enable pull-up on the button pin during sleep (crucial for proper wake-up)
  rtc_gpio_pullup_en((gpio_num_t)BUTTON_PIN);
  rtc_gpio_pulldown_dis((gpio_num_t)BUTTON_PIN);

  Serial.flush();

  // Change state before actually entering sleep
  currentState = STATE_DEEP_SLEEP;

  esp_deep_sleep_start();
}

// Function to reset and reinitialize the camera
bool resetCamera()
{
  // First deinitialize the camera if it was initialized
  if (cameraInitialized)
  {
    // Put camera in software standby first
    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor)
    {
      sensor->set_reg(sensor, 0x3008, 0xFF, 0x42);
      delay(100);
    }
    
    // Deinitialize camera
    esp_camera_deinit();
    cameraInitialized = false;
  }
  
  // Small delay to make sure camera hardware has time to reset
  delay(200);
  
  // Try to initialize the camera again
  int camera_err = setupCamera();
  
  if (camera_err == ERR_OK)
  {
    cameraInitialized = true;
    return true;
  }
  else
  {
    return false;
  }
}

// State handler functions
void handleStartState()
{
  if (DEBUG_BYPASS_DEEPSLEEP)
  {
    // Initialize button pin with internal pull-up and attach interrupt
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), buttonISR, FALLING);

    // Initialize BLE
    setupBLE();

    // Start advertising
    BLEDevice::startAdvertising();

    currentState = STATE_SETUP_ADVERTISE;
  }
  else
  {
    digitalWrite(LED_BUILTIN, HIGH); // Turn off LED before sleep
    enterDeepSleep();
  }
}

void handleWakeFromSleep()
{
  digitalWrite(LED_BUILTIN, LOW); // Turn on LED when waking up

  // Initialize button pin with internal pull-up and attach interrupt
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), buttonISR, FALLING);

  // Initialize BLE first so clients can connect while other components are initializing
  setupBLE();

  // Start advertising
  BLEDevice::startAdvertising();

  currentState = STATE_SETUP_ADVERTISE;
}

void handleSetupAdvertise()
{
  // Flash LED every 500ms while advertising
  static unsigned long lastLedToggle = 0;
  if (millis() - lastLedToggle >= 500)
  {
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); // Toggle LED
    lastLedToggle = millis();
  }

  // If client connected, move to CONNECTED state
  if (deviceConnected)
  {
    digitalWrite(LED_BUILTIN, LOW); // Keep LED on when connected

    // Initialize camera only if not already initialized
    if (!cameraInitialized)
    {
      int camera_err = setupCamera();

      // Check for errors
      if (camera_err != ERR_OK)
      {
        enterDeepSleep(); // Go to sleep if initialization failed
        return;
      }

      cameraInitialized = true;
    }

    systemInitialized = true;
    currentState = STATE_CONNECTED;
  }
  else
  {
    // Brief delay to prevent tight loop
    delay(100);
  }
}

void handleConnectedState()
{
  // Check if client disconnected
  if (!deviceConnected)
  {
    if (DEBUG_ADVERTISING_ON_DISCONNECT)
    {
      currentState = STATE_SETUP_ADVERTISE;
      return;
    }
    else
    {
      enterDeepSleep();
      return;
    }
  }

  // Check for BLE L2CAP errors detected via logging
  if (l2cap_error_detected) {
    ESP.restart();
    resetBLEStack();
    delay(100); // Short delay after reset
    return;
  }

  static uint32_t last_frame = 0;
  static uint32_t last_activity = millis(); // Track last activity
  static int consecutive_failures = 0;      // Track consecutive camera failures
  static const int MAX_FAILURES = 3;        // Max consecutive failures before reset

  if (millis() - last_frame < 100)
    return; // Limit to ~10fps
  last_frame = millis();
  last_activity = millis(); // Update activity timestamp

  digitalWrite(LED_BUILTIN, LOW); // Keep LED on while connected

  // Timing: Camera capture start
  uint32_t start_camera = micros();

  // Capture frame from camera
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb)
  {
    consecutive_failures++;
    
    // Check if we've had too many consecutive failures
    if (consecutive_failures >= MAX_FAILURES)
    {
      if (resetCamera())
      {
        consecutive_failures = 0; // Reset failure counter after successful reset
      }
      else
      {
        // Add a delay to prevent tight reset loop
        delay(1000);
        
        // After multiple failed reset attempts, restart ESP32
        if (consecutive_failures >= MAX_FAILURES * 3) {
          delay(1000);
          ESP.restart();
        }
      }
    }
    return;
  }

  // Reset consecutive failures counter on successful capture
  consecutive_failures = 0;

  // Timing: Camera capture end
  uint32_t camera_time = micros() - start_camera;

  // Timing: BLE send start
  uint32_t start_ble_send = micros();

  // Send JPEG directly over BLE without any processing
  sendJPEG(fb->buf, fb->len);

  // Timing: BLE send end
  uint32_t ble_send_time = micros() - start_ble_send;

  // Clean up
  esp_camera_fb_return(fb);

  // Increment counter
  image_counter++;

  // After completing a processing cycle, check for inactivity timeout
  if (millis() - last_activity > 60000)
  { // 1 minute inactivity timeout
    enterDeepSleep();
  }
}

void setup()
{
  Serial.begin(115200);

  // Initialize LED_BUILTIN as output
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW); // Turn on LED (LOW is on for ESP32)

  bootCount++;

  // Check wake-up reason to determine state
  esp_sleep_wakeup_cause_t wakeup_reason = esp_sleep_get_wakeup_cause();

  // Set initial state based on boot or wake-up
  if (wakeup_reason == ESP_SLEEP_WAKEUP_EXT0)
  {
    // Woke up from button press
    currentState = STATE_DEEP_SLEEP; // We're coming from deep sleep
  }
  else
  {
    // First boot or reset
    currentState = STATE_START;
  }
}

void loop()
{
  // Check for button press in all states
  if (buttonPressed)
  {
    // Wait until button is released
    while (digitalRead(BUTTON_PIN) == LOW)
    {
      delay(10);
    }
    delay(50);

    buttonPressed = false;
    enterDeepSleep();
    return;
  }

  // State machine
  switch (currentState)
  {
  case STATE_START:
    handleStartState();
    break;

  case STATE_DEEP_SLEEP:
    handleWakeFromSleep();
    break;

  case STATE_SETUP_ADVERTISE:
    handleSetupAdvertise();
    break;

  case STATE_CONNECTED:
    handleConnectedState();
    break;

  default:
    currentState = STATE_START;
    break;
  }
}