# Thalmic MYO Armband: Hardware Reference

The MYO Armband is the primary EMG hardware for the MAIA EMG-ASL Layer.
It provides 8-channel sEMG at 200 Hz without any custom wiring, firmware,
or electrode preparation.

---

## Specifications

| Spec | Value |
|------|-------|
| Channels | 8 sEMG electrodes, arranged circumferentially |
| Sample rate | 200 Hz |
| ADC resolution | ~8-bit effective (int8 output, −128 to 127) |
| Amplitude range | ±1.25 mV (after int8 → mV conversion) |
| Connectivity | Bluetooth Low Energy 4.0 |
| Power | Rechargeable internal battery (micro-USB) |
| Battery life | ~6-8 hours of continuous streaming |
| Weight | ~93 g |
| Dimensions | Adjustable band, fits forearm circumference 175-270 mm |

---

## Connection Paths

### Path A: Laptop (Data Collection)

```
MYO Armband
    │ Bluetooth (proprietary dongle protocol)
    ▼
MYO USB Bluetooth Dongle
    │ USB
    ▼
MyoConnect (macOS / Windows host app)
    │ IPC / SDK
    ▼
myo-python (CFFI bindings to Myo C++ SDK)
    │ Python API
    ▼
scripts/record_session.py
    │ CSV
    ▼
data/raw/  →  training pipeline
```

**Required software:**
- MyoConnect (Thalmic Labs, available via archive download)
- `pip install myo-python>=0.2.1 cffi>=1.16`
- `export MYO_SDK_PATH="/Applications/Myo Connect.app/Contents/Frameworks"` (macOS)

### Path B: iPhone (Live Inference)

```
MYO Armband
    │ Bluetooth Low Energy (direct GATT protocol)
    │ No dongle or MyoConnect required
    ▼
React Native App (BLEManager.ts)
    │ raw int8 bytes
    ▼
EMGStreamProcessor.ts  →  ONNX inference  →  TTS
```

**Required:** iOS Bluetooth permission granted on first launch.

---

## MYO BLE GATT Protocol

The MYO uses a proprietary BLE GATT profile reverse-engineered from the
Thalmic SDK. Source: https://github.com/thalmiclabs/myo-bluetooth

### Services and Characteristics

| Resource | UUID |
|----------|------|
| **Control Service** | `d5060001-a904-deb9-4748-2c7f4a124842` |
| Command Characteristic | `d5060401-a904-deb9-4748-2c7f4a124842` |
| **EMG Service** | `d5060005-a904-deb9-4748-2c7f4a124842` |
| EMG Char 0 | `d5060105-a904-deb9-4748-2c7f4a124842` |
| EMG Char 1 | `d5060205-a904-deb9-4748-2c7f4a124842` |
| EMG Char 2 | `d5060305-a904-deb9-4748-2c7f4a124842` |
| EMG Char 3 | `d5060405-a904-deb9-4748-2c7f4a124842` |

### Enabling EMG Streaming

Write the following 5-byte command to the Command Characteristic before
subscribing to the EMG notify characteristics:

```
[0x01, 0x03, 0x02, 0x01, 0x01]
  │     │     │     │     └── classifier mode: on
  │     │     │     └──────── IMU mode: on
  │     │     └────────────── EMG mode: 0x02 = raw filtered (200 Hz, int8)
  │     └──────────────────── payload size: 3 bytes
  └────────────────────────── command type: 0x01 = set mode
```

Without this command, no EMG notifications will be delivered.

### Notification Format

Each of the 4 EMG characteristics delivers notifications at ~50 Hz.
Each notification is **8 bytes**: 2 samples × 4 channels (int8):

```
Byte:  [0]       [1]       [2]       [3]       [4]       [5]       [6]       [7]
       s0_ch0    s0_ch1    s0_ch2    s0_ch3    s1_ch0    s1_ch1    s1_ch2    s1_ch3
Type:  int8      int8      int8      int8      int8      int8      int8      int8
Range: −128 to 127
```

The 4 characteristics carry different channel groups:

| Characteristic | Channels | Samples |
|----------------|----------|---------|
| EMG Char 0 (`d5060105`) | Ch 0–3 | Samples 0–1 |
| EMG Char 1 (`d5060205`) | Ch 4–7 | Samples 0–1 |
| EMG Char 2 (`d5060305`) | Ch 0–3 | Samples 2–3 |
| EMG Char 3 (`d5060405`) | Ch 4–7 | Samples 2–3 |

Subscribe to all 4 simultaneously. Together they reconstruct all 8 channels
at 200 Hz.

### int8 to Millivolts

```python
mv = value / 127.0 * 1250.0   # result in µV (±1250 µV = ±1.25 mV)
```

---

## Electrode Placement

### Position

Place the armband on the **dominant forearm**, **2-3 cm below the elbow
crease**, tightened until snug. The MYO logo (electrode 1 / Ch0) should
face the **palmar (inner wrist) side**.

```
 RIGHT FOREARM: cross-section view (elbow toward viewer)

                  DORSAL (top)
                    [ Ch6 ]
         [ Ch5 ]  Ext. Digitorum  [ Ch7 ]
     10 o'clock   Communis        2 o'clock

RADIAL                                    ULNAR
(thumb)  [ Ch4 ]           [ Ch0 / logo ] (pinky)
          9 o'clock          3 o'clock

         [ Ch3 ]  Flex. Digitorum  [ Ch1 ]
      8 o'clock   Superficialis    4 o'clock

                    [ Ch2 ]
                  PALMAR (bottom)
```

### Tightening

- Snug — electrode pods must contact skin with no gaps
- Not too tight — should not cause discomfort or impede blood flow
- A loose armband is the #1 cause of flat or noisy channels

### Skin Prep (Long Sessions)

1. Wipe with 70% isopropyl alcohol wipe
2. Lightly abrade with gauze (2-3 passes)
3. Let dry completely before fitting armband
4. Avoid applying lotion or sunscreen before the session

---

## myo_ecn Toolbox Integration

The `myo_ecn` toolbox (https://github.com/smetanadvorak/myo_ecn) provides
`EmgBuffer`, `Collector`, and `MultichannelPlot` utilities built on top of
myo-python. These are useful for interactive data exploration:

```python
from myo_ecn import Collector

collector = Collector()
collector.run(duration=10)  # collect 10 seconds of EMG
data = collector.emg_data   # numpy array, shape (N, 8)
```

For production data recording, use `scripts/record_session.py` which provides
label prompting, annotation, and CSV export in the project's session format.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Hub failed to start` | MyoConnect not running | Launch MyoConnect; confirm dongle is plugged in |
| `myo.init()` fails | Wrong `MYO_SDK_PATH` | Point to directory containing `myo.framework` (macOS) |
| No `on_emg` callbacks | `stream_emg(True)` not called | Call `event.device.stream_emg(True)` in `on_connected` |
| All channels flat | Armband too loose | Tighten until snug |
| One channel flat during clench | Electrode pod not on muscle | Rotate armband ±1 cm, re-tighten |
| High noise at rest | Dry skin | Wipe forearm with water; allow to dry slightly; re-seat |
| BLE scan timeout (iPhone) | Armband asleep | Double-tap MYO logo to wake |
| "Myo not found" on iPhone | iOS Bluetooth off | Enable Bluetooth and grant app permission |
| Battery drains fast | Old battery | Charge via micro-USB (1–2 hours for full charge) |

---

## Firmware Version

The firmware version is displayed in MyoConnect's Devices panel. Supported
versions: **1.1.x** and **1.6.x**. Older versions may not support raw EMG
streaming — update via MyoConnect → Help → Update Firmware.

---

## References

- MYO BLE protocol (community reverse-engineering): https://github.com/thalmiclabs/myo-bluetooth
- myo-python library: https://github.com/NiklasRosenstein/myo-python
- myo_ecn toolbox: https://github.com/smetanadvorak/myo_ecn
- awesome-emg-data (datasets collected with MYO): https://github.com/x-labs-xyz/awesome-emg-data
