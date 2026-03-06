# Hardware — EMG-ASL Layer

This directory contains hardware documentation for the EMG wristband used by
the EMG-ASL Layer.

---

## Supported Devices

| Device | Status | Directory | Notes |
|--------|--------|-----------|-------|
| **Thalmic MYO Armband** | **Primary / Supported** | `myo_armband/` | 8-ch sEMG, 200 Hz, int8. USB dongle + MyoConnect for laptop; direct BLE for mobile. |
| OpenBCI Cyton (8-ch) | Supported (via LSL bridge) | `openbci/` | Higher ADC resolution; less portable |
| OpenBCI Ganglion (4-ch) | Partial support | `openbci/` | 4-channel mode; reduced accuracy |

---

## Quick Setup Links

- **Full hardware setup guide**: [`../docs/hardware-setup.md`](../docs/hardware-setup.md)
- **MYO day-one checklist**: [`../HARDWARE_ARRIVES.md`](../HARDWARE_ARRIVES.md)
- **MYO BLE protocol reference**: [`myo_armband/README.md`](myo_armband/README.md)
- **myo-python docs**: https://github.com/NiklasRosenstein/myo-python
- **MYO BLE reverse engineering**: https://github.com/thalmiclabs/myo-bluetooth
- **myo_ecn toolbox**: https://github.com/smetanadvorak/myo_ecn

---

## MYO BLE UUIDs

These UUIDs are used by the mobile app and are defined in:
- `src/utils/constants.py` (Python server)
- `mobile/react-native/src/bluetooth/BLEManager.ts` (React Native app)

| Resource | UUID |
|----------|------|
| Control Service | `d5060001-a904-deb9-4748-2c7f4a124842` |
| Command Characteristic | `d5060401-a904-deb9-4748-2c7f4a124842` |
| EMG Service | `d5060005-a904-deb9-4748-2c7f4a124842` |
| EMG Char 0 (ch 0–3, samples 0–1) | `d5060105-a904-deb9-4748-2c7f4a124842` |
| EMG Char 1 (ch 4–7, samples 0–1) | `d5060205-a904-deb9-4748-2c7f4a124842` |
| EMG Char 2 (ch 0–3, samples 2–3) | `d5060305-a904-deb9-4748-2c7f4a124842` |
| EMG Char 3 (ch 4–7, samples 2–3) | `d5060405-a904-deb9-4748-2c7f4a124842` |

---

## MYO EMG Data Format

Each of the 4 EMG notify characteristics delivers:
- **2 samples** × **4 channels** (int8, signed) per notification
- ~50 Hz per characteristic → **8 channels at 200 Hz** combined

```
Notification bytes (8 bytes total):
  [s0_ch0, s0_ch1, s0_ch2, s0_ch3, s1_ch0, s1_ch1, s1_ch2, s1_ch3]
  Each value: int8, range −128 to 127

To convert to millivolts:
  mv = value / 127.0 * 1250.0   →  range ≈ ±1250 µV = ±1.25 mV
```

The four characteristics are subscribed simultaneously. Together they
reconstruct all 8 channels at 200 Hz.

---

## SET_EMG_MODE Command

Before subscribing to EMG characteristics, write this command to the
Command Characteristic (`d5060401-...`) to enable raw EMG streaming:

```
Bytes: [0x01, 0x03, 0x02, 0x01, 0x01]
       cmd=set_mode, size=3, emg=raw_filtered, imu=on, classifier=on
```

Without this command, the EMG characteristics will not emit notifications.

---

## Connection Paths

**Laptop / data collection path** (primary for recording):
```
MYO Armband → MYO USB Dongle → MyoConnect → myo-python → record_session.py
```

**Mobile app path** (primary for live inference):
```
MYO Armband → Direct BLE → React Native BLEManager → EMGStreamProcessor → ONNX
```

---

## Directory Structure

```
hardware/
├── README.md                     This file
└── myo_armband/
    └── README.md                 MYO setup guide, BLE protocol, troubleshooting
```

Legacy (archived):
```
hardware/
└── myoware_ble/                  (Archived) nRF52840 + MyoWare 2.0 custom firmware
    └── myoware_ble_streamer.ino
```
