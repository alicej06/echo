/**
 * myoware_ble_streamer.ino
 *
 * EMG-ASL Layer — BLE EMG Streamer Firmware
 * Target:  Adafruit Feather nRF52840 Express
 * Sensors: 8× MyoWare 2.0 Muscle Sensor (AT-04-001) on analog pins A0–A7
 *
 * Behavior:
 *   - Samples all 8 ADC channels at exactly 200 Hz via a hardware timer ISR.
 *   - Packs each sample as 8× int16 big-endian into a 16-byte buffer.
 *   - Sends the buffer as a BLE notification on the EMG characteristic.
 *   - Advertises as "MAIA-EMG-Band" until a central connects.
 *   - LED: blinking = advertising, solid = connected.
 *   - Reduces TX power when connected to conserve battery.
 *   - Streams Serial debug output at 115200 baud.
 *
 * BLE UUIDs (must match src/utils/constants.py and mobile BLEManager.ts):
 *   Service:        12345678-1234-5678-1234-56789abcdef0
 *   Characteristic: 12345678-1234-5678-1234-56789abcdef1
 *
 * Wiring:
 *   Feather 3V3 → MyoWare +V  (all 8 sensors, star)
 *   Feather GND → MyoWare GND (all 8 sensors, star)
 *   Feather A0  → MyoWare #0 SIG
 *   Feather A1  → MyoWare #1 SIG
 *   Feather A2  → MyoWare #2 SIG
 *   Feather A3  → MyoWare #3 SIG
 *   Feather A4  → MyoWare #4 SIG
 *   Feather A5  → MyoWare #5 SIG
 *   Feather A6  → MyoWare #6 SIG
 *   Feather A7  → MyoWare #7 SIG
 *
 * Board: Adafruit Feather nRF52840 Express (via Adafruit nRF52 board package)
 * IDE: Arduino 2.x
 * Libraries: Adafruit Bluefruit nRF52 (included in board package)
 *
 * Revision history:
 *   2026-02-01  v1.0.0  Initial release — MAIA Biotech Spring 2026
 */

#include <bluefruit.h>

// =============================================================================
// Configuration constants
// (Must match src/utils/constants.py and mobile/react-native/src/bluetooth/)
// =============================================================================

#define N_CHANNELS         8        // Number of EMG channels (A0..A7)
#define SAMPLE_RATE_HZ     200      // Samples per second per channel
#define DEVICE_NAME        "MAIA-EMG-Band"
#define SERIAL_BAUD        115200
#define ADC_RESOLUTION     12       // bits — 0..4095 range

// BLE Service and Characteristic UUIDs
// 128-bit UUIDs stored little-endian in the array (reversed from human-readable)
static const uint8_t EMG_SERVICE_UUID[] = {
  0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12,
  0x78, 0x56, 0x34, 0x12, 0x78, 0x56, 0x34, 0x12
};

static const uint8_t EMG_CHAR_UUID[] = {
  0xf1, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12,
  0x78, 0x56, 0x34, 0x12, 0x78, 0x56, 0x34, 0x12
};

// Analog input pins in channel order (A0–A7 on nRF52840 Feather)
static const uint8_t ADC_PINS[N_CHANNELS] = { A0, A1, A2, A3, A4, A5, A6, A7 };

// LED pin (built-in blue LED on Feather nRF52840)
#define LED_PIN            LED_BLUE

// =============================================================================
// BLE objects
// =============================================================================

BLEService        emgService(BLEUuid(EMG_SERVICE_UUID));
BLECharacteristic emgChar(BLEUuid(EMG_CHAR_UUID));

// =============================================================================
// Globals
// =============================================================================

// Double-buffered sample storage: ISR writes to one buffer, BLE task reads from other
volatile uint16_t sampleBuf[2][N_CHANNELS];   // raw ADC values (0..4095)
volatile uint8_t  writeIdx    = 0;            // index of buffer ISR is writing to
volatile bool     sampleReady = false;         // set by ISR, cleared by main loop

// Connection state
volatile bool     bleConnected = false;

// Timer handle for 200 Hz sampling ISR
SoftwareTimer samplingTimer;

// LED blink state
uint32_t lastLedToggle = 0;

// =============================================================================
// Timer ISR — runs at SAMPLE_RATE_HZ
// Reads all 8 ADC channels and sets sampleReady flag.
// =============================================================================

void IRAM_ATTR samplingISR() {
  uint8_t buf = writeIdx;           // capture write index atomically
  for (uint8_t ch = 0; ch < N_CHANNELS; ch++) {
    sampleBuf[buf][ch] = (uint16_t)analogRead(ADC_PINS[ch]);
  }
  writeIdx  ^= 1;                   // swap buffers (0→1 or 1→0)
  sampleReady = true;
}

// =============================================================================
// BLE connection / disconnection callbacks
// =============================================================================

void connectCallback(uint16_t connHandle) {
  bleConnected = true;
  digitalWrite(LED_PIN, HIGH);      // solid LED = connected

  // Reduce TX power when connected to conserve battery
  // nRF52840 supported values: -40, -20, -16, -12, -8, -4, 0, +3, +4 dBm
  Bluefruit.setTxPower(-8);

  BLEConnection* conn = Bluefruit.Connection(connHandle);
  conn->requestPHY(CONN_PHY_LE_2M);            // Request 2 Mbps PHY if supported
  conn->requestConnectionParameter(6, 8, 0, 500); // interval 6–8×1.25ms, latency 0, timeout 5s

  Serial.print("[BLE] Central connected. Handle: ");
  Serial.println(connHandle);
}

void disconnectCallback(uint16_t connHandle, uint8_t reason) {
  (void)connHandle;
  bleConnected = false;

  // Restore advertising TX power
  Bluefruit.setTxPower(0);

  Serial.print("[BLE] Disconnected. Reason: 0x");
  Serial.println(reason, HEX);
  Serial.println("[BLE] Restarting advertising...");

  startAdvertising();
}

// =============================================================================
// BLE advertising setup
// =============================================================================

void startAdvertising() {
  Bluefruit.Advertising.clearData();
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(emgService);
  Bluefruit.Advertising.setInterval(160, 160);  // 100 ms in 0.625 ms units
  Bluefruit.Advertising.setFastTimeout(30);      // 30 seconds fast mode
  Bluefruit.Advertising.start(0);               // 0 = advertise indefinitely

  Bluefruit.ScanResponse.addName();             // Device name in scan response

  Serial.println("[BLE] Advertising started.");
}

// =============================================================================
// Setup
// =============================================================================

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 3000) { }        // wait up to 3 s for Serial Monitor

  Serial.println("==============================================");
  Serial.println(" MAIA-EMG-Band starting...");
  Serial.println(" EMG-ASL Layer — MAIA Biotech Spring 2026");
  Serial.println("==============================================");
  Serial.print  (" Device name:    "); Serial.println(DEVICE_NAME);
  Serial.print  (" Channels:       "); Serial.println(N_CHANNELS);
  Serial.print  (" Sample rate:    "); Serial.print(SAMPLE_RATE_HZ); Serial.println(" Hz");
  Serial.print  (" ADC resolution: "); Serial.print(ADC_RESOLUTION); Serial.println(" bit");
  Serial.println("==============================================");

  // ---- LED setup
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // ---- ADC setup
  analogReadResolution(ADC_RESOLUTION);    // 12-bit: 0..4095
  analogReference(AR_INTERNAL_3_0);       // 3.0 V reference (matches MyoWare 3V3 supply)

  // Warm-up reads to stabilize ADC (nRF52840 ADC needs settling time)
  for (uint8_t ch = 0; ch < N_CHANNELS; ch++) {
    analogRead(ADC_PINS[ch]);
    analogRead(ADC_PINS[ch]);
  }
  Serial.println("[ADC] Initialized. 12-bit resolution, 3.0V reference.");

  // ---- BLE setup
  Bluefruit.begin();
  Bluefruit.setName(DEVICE_NAME);
  Bluefruit.setTxPower(0);              // 0 dBm while advertising
  Bluefruit.Periph.setConnectCallback(connectCallback);
  Bluefruit.Periph.setDisconnectCallback(disconnectCallback);

  // Configure EMG BLE service and characteristic
  emgService.begin();

  emgChar.setProperties(CHR_PROPS_NOTIFY);
  emgChar.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
  emgChar.setFixedLen(N_CHANNELS * 2);  // 8 channels × 2 bytes = 16 bytes
  emgChar.setUserDescriptor("EMG 8-Ch Raw");
  emgChar.begin();

  Serial.println("[BLE] Service and characteristic initialized.");
  Serial.print  ("[BLE] Service UUID:        12345678-1234-5678-1234-56789abcdef0\n");
  Serial.print  ("[BLE] Characteristic UUID: 12345678-1234-5678-1234-56789abcdef1\n");

  // ---- Start advertising
  startAdvertising();

  // ---- Start hardware sampling timer at 200 Hz
  samplingTimer.begin(1000000UL / SAMPLE_RATE_HZ, samplingISR);  // period in microseconds
  samplingTimer.start();

  Serial.print("[Timer] Sampling ISR started at ");
  Serial.print(SAMPLE_RATE_HZ);
  Serial.println(" Hz.");
  Serial.println("[READY] Waiting for BLE central...");
}

// =============================================================================
// Main loop — send BLE notifications when ISR has a new sample ready
// =============================================================================

void loop() {
  // --- Handle LED blink when advertising (not connected)
  if (!bleConnected) {
    uint32_t now = millis();
    if (now - lastLedToggle >= 500) {    // blink at ~1 Hz
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      lastLedToggle = now;
    }
  }

  // --- Send BLE notification when a new sample is ready
  if (sampleReady) {
    sampleReady = false;

    // Read from the buffer that the ISR just finished writing to (inverted index)
    uint8_t readIdx = writeIdx ^ 1;

    // Pack 8 uint16 ADC values as signed int16 big-endian into a 16-byte packet.
    // The MyoWare 2.0 output is always positive (0..3V3), so mapping:
    //   raw uint16 (0..4095) → int16: subtract 2048 to center around zero.
    // This allows the server to detect both baseline and peak deflections.
    uint8_t packet[N_CHANNELS * 2];
    for (uint8_t ch = 0; ch < N_CHANNELS; ch++) {
      int16_t val = (int16_t)sampleBuf[readIdx][ch] - 2048; // center at 0
      // Big-endian packing (network byte order)
      packet[ch * 2]     = (uint8_t)((val >> 8) & 0xFF);   // high byte
      packet[ch * 2 + 1] = (uint8_t)( val       & 0xFF);   // low byte
    }

    // Send notification only if a central is subscribed
    if (bleConnected && emgChar.notifyEnabled()) {
      if (!emgChar.notify(packet, sizeof(packet))) {
        // Notification failed (e.g., BLE queue full) — drop this sample silently.
        // At 200 Hz we can afford occasional drops without impacting feature extraction.
      }
    }

    // Serial debug output (throttled to every 50th sample = 4 Hz display rate)
    static uint32_t debugCounter = 0;
    if (++debugCounter >= 50) {
      debugCounter = 0;
      Serial.print("[EMG] Ch0–7: ");
      for (uint8_t ch = 0; ch < N_CHANNELS; ch++) {
        Serial.print(sampleBuf[readIdx][ch]);
        if (ch < N_CHANNELS - 1) Serial.print(", ");
      }
      Serial.print("  BLE: ");
      Serial.println(bleConnected ? "connected" : "advertising");
    }
  }

  // Yield to the nRF52 SoftDevice scheduler — required for BLE stack operation
  yield();
}
