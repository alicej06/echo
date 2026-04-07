# Hardware Setup Guide — MYO Armband

This guide covers setup, SDK installation, electrode placement, and
troubleshooting for the **Thalmic MYO Armband** used by the EMG-ASL Layer.

---

## Table of Contents

1. [What You Need](#1-what-you-need)
2. [MyoConnect + SDK Setup](#2-myoconnect--sdk-setup)
3. [myo-python Setup](#3-myo-python-setup)
4. [First Connection Test](#4-first-connection-test)
5. [Electrode Placement](#5-electrode-placement)
6. [Direct BLE Mode (Mobile App)](#6-direct-ble-mode-mobile-app)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. What You Need

| # | Component | Notes |
|---|-----------|-------|
| 1 | **Thalmic MYO Armband** | 8-channel sEMG, 200 Hz, int8 output |
| 2 | **MYO USB Bluetooth Dongle** | Required for laptop/desktop connection via MyoConnect |
| 3 | **MyoConnect software** | Thalmic Labs host application for macOS/Windows |
| 4 | **Mac or Windows laptop** | For server-side data collection (myo-python) |
| 5 | **iPhone (iOS 14+)** | For the React Native app (direct BLE, no dongle needed) |
| 6 | **USB-A port or USB-A adapter** | For the Bluetooth dongle |

**No custom wiring, soldering, or firmware flashing is required.**
The MYO Armband is a complete, self-contained 8-channel sEMG device.

---

## 2. MyoConnect + SDK Setup

MyoConnect is the host application that manages the MYO's Bluetooth dongle
connection. The myo-python SDK communicates through it.

### 2.1 Install MyoConnect

1. Obtain `MyoConnect.app` (macOS) or `MyoConnect_Installer.exe` (Windows)
   from the Thalmic Labs archive.
2. Install and launch MyoConnect. A menu bar icon (macOS) or system tray icon
   (Windows) will appear.
3. **Plug in the MYO USB Bluetooth dongle**.
4. Wake the MYO Armband by **double-tapping the MYO logo** or holding it
   for 2 seconds. The LED pulses white while pairing.
5. MyoConnect discovers and pairs automatically. The LED turns solid green
   when paired and connected.

### 2.2 Set SDK Environment Variable

The myo-python library needs to find the Myo C++ SDK dynamic library.
MyoConnect bundles it inside the application package.

**macOS:**
```bash
export MYO_SDK_PATH="/Applications/Myo Connect.app/Contents/Frameworks"
```

**Windows (PowerShell):**
```powershell
$env:MYO_SDK_PATH = "C:\Program Files (x86)\Myo Connect\sdk"
```

Add this to your shell profile (`~/.zshrc`, `~/.bash_profile`, or `.env`)
to avoid re-entering it each session.

### 2.3 Verify MyoConnect is Working

In MyoConnect's Devices panel, the armband should appear with:
- Battery level indicator
- Signal strength bar
- Firmware version (1.1.x or 1.6.x)

If the armband appears but shows "No Signal," re-seat the dongle in a
different USB port, preferably USB 2.0 (some USB 3.0 ports cause 2.4 GHz
interference).

---

## 3. myo-python Setup

myo-python is a CFFI Python wrapper around the Myo C++ SDK. It communicates
with the MYO through MyoConnect — MyoConnect must be running and the armband
must be connected before any myo-python script can use it.

### 3.1 Install

```bash
pip install myo-python>=0.2.1 cffi>=1.16
```

### 3.2 Verify Installation

```bash
python - <<'EOF'
import myo
myo.init()
hub = myo.Hub()
print("myo-python OK — hub created successfully")
hub.stop()
EOF
```

Expected output: `myo-python OK — hub created successfully`

If you see `RuntimeError: Hub failed to start`, check that:
1. MyoConnect is running (menu bar icon visible).
2. The MYO USB dongle is plugged in.
3. `MYO_SDK_PATH` points to the directory containing `myo.framework` (macOS)
   or `myo32.dll` / `myo64.dll` (Windows).

### 3.3 Test EMG Streaming

```bash
python - <<'EOF'
import myo, time

class Listener(myo.DeviceListener):
    def on_connected(self, event):
        print(f"[MYO] Connected: {event.device_name}")
        event.device.stream_emg(True)
    def on_emg(self, event):
        print(f"[EMG] {list(event.emg)}")
        return False  # return False stops the hub after first sample

myo.init()
hub = myo.Hub()
hub.run(5000, Listener())  # 5 second timeout
print("Done.")
EOF
```

Expected output (one line):
```
[MYO] Connected: Myo
[EMG] [-12, 3, -7, 21, -4, 8, 0, -15]
Done.
```

The 8 values are signed int8 (range −128 to 127), representing raw sEMG
amplitude from each of the 8 electrodes.

### 3.4 Convert int8 to millivolts

The MYO outputs signed int8 values. To convert to approximate millivolts:

```python
mv = [v / 127.0 * 1250.0 for v in emg_data]
```

This maps the int8 range [−128, 127] to approximately [−1250, 1250] µV
(±1.25 mV), which is the typical sEMG amplitude range.

---

## 4. First Connection Test

### 4.1 Run the Smoke Test

```bash
cd /path/to/emg-asl-layer
source venv/bin/activate

python scripts/smoke_test.py
```

Expected output: `8/8 tests passed`

The smoke test does not require the MYO hardware — it validates the
signal processing and inference pipeline with synthetic data.

### 4.2 Live EMG Stream Verification (Hardware Required)

With MyoConnect running and the armband connected:

```bash
python scripts/record_session.py \
  --participant TEST \
  --output /tmp/ \
  --labels A \
  --reps 1 \
  --duration 5
```

This records 5 seconds of channel 'A' and prints sample statistics.
You should see non-zero variance on most channels.

### 4.3 Signal Quality Check

Good signal indicators:
- **At rest (relaxed forearm):** All 8 channels near zero (< ±10 counts)
- **Light fist clench:** Palmar channels (approx. Ch1–Ch4) spike to ±40–80
- **Strong fist clench:** Palmar channels spike to ±100 or more
- **Noise floor:** < 5 counts RMS at rest after notch filter

If channels are flat near zero even during clench:
- Armband may be too loose — tighten until snug
- Skin may be dry — wipe electrode contacts with slightly damp cloth

---

## 5. Electrode Placement

The MYO Armband is a complete pre-assembled unit — no individual electrode
preparation is needed. The 8 electrodes are built into the armband and make
contact with the skin when worn.

### 5.1 Target Position

Place the MYO on the **dominant forearm**, **2–3 cm distal to the elbow
crease** (toward the wrist), targeting the widest circumferential point of
the upper forearm where the flexor and extensor muscle bellies are broadest.

```
 ELBOW                                           WRIST
   |---- 2-3 cm ----|
                    ┌────────────────────────┐
                    │     MYO Armband        │   ← place here
                    │  (8 electrodes, 360°)  │
                    └────────────────────────┘
```

### 5.2 MYO Logo Orientation

The MYO logo indicates electrode 1 (Ch0 in software). Orient the armband so
the logo faces toward the **inside of the wrist** (palmar side) for
consistency across sessions.

### 5.3 Muscle Groups Targeted

```
 RIGHT FOREARM — cross-section, looking from elbow toward wrist
 Outer ring = forearm surface

                      12 o'clock (DORSAL)
                          [ Ch6 ]
               [ Ch5 ]  Ext. Digitorum  [ Ch7 ]
            10 o'clock   Communis        2 o'clock
        Ext. Carpi                         Ext. Carpi
        Radialis Br.                       Radialis Lg.

 RADIAL                                           ULNAR
 (thumb side)   [ Ch4 ]           [ Ch0 ]    (pinky side)
                 9 o'clock          3 o'clock
            Flexor Carpi          Flexor Carpi
             Radialis              Ulnaris

               [ Ch3 ]  Flexor Digitorum  [ Ch1 ]
           8 o'clock    Superficialis      4 o'clock

                          [ Ch2 ]
                      6 o'clock (PALMAR)
```

### 5.4 Tightening

Tighten the armband until it is snug — all 8 metal electrode pods should
maintain firm contact with the skin. A loose armband is the single most
common cause of noisy or flat channels.

- Too loose: flat or low-amplitude signal, high motion artifact
- Too tight: uncomfortable and may impede blood flow; loosen one click
- Correct: slight indentation visible after removal; all channels active

### 5.5 Left-Handed Participants

Mirror the diagram (radial and ulnar sides swap). Record `dominant_hand: left`
in session metadata. No other changes are needed.

### 5.6 Skin Preparation

For long sessions (> 30 minutes):
1. Wipe the forearm with an isopropyl alcohol wipe (70%). Allow to dry.
2. Gently abrade with gauze (one or two passes, not harsh).
3. Allow skin to air dry before fitting the armband.

For standard sessions:
- Clean, dry skin is sufficient.
- Avoid lotions or oils — they increase electrode impedance.

---

## 6. Direct BLE Mode (Mobile App)

The iOS React Native app connects **directly to the MYO** via the
reverse-engineered MYO BLE GATT protocol — no MyoConnect or USB dongle
required on the phone.

### MYO BLE Protocol Summary

| Resource | UUID |
|----------|------|
| Control Service | `d5060001-a904-deb9-4748-2c7f4a124842` |
| Command Characteristic | `d5060401-a904-deb9-4748-2c7f4a124842` |
| EMG Service | `d5060005-a904-deb9-4748-2c7f4a124842` |
| EMG Char 0 | `d5060105-a904-deb9-4748-2c7f4a124842` |
| EMG Char 1 | `d5060205-a904-deb9-4748-2c7f4a124842` |
| EMG Char 2 | `d5060305-a904-deb9-4748-2c7f4a124842` |
| EMG Char 3 | `d5060405-a904-deb9-4748-2c7f4a124842` |

**SET_EMG_MODE command** (written to Command Characteristic before subscribing):
```
0x01 0x03 0x02 0x01 0x01
```
- `0x01` = command type (set mode)
- `0x03` = payload size
- `0x02` = EMG mode: raw filtered (200 Hz, int8)
- `0x01` = IMU mode: on
- `0x01` = classifier mode: on

**EMG data format per characteristic:**
Each notification delivers 2 samples × 4 channels (int8) at ~50 Hz.
All 4 characteristics together reconstruct 8 channels at 200 Hz.

The app handles this automatically. See
[`src/bluetooth/BLEManager.ts`](../mobile/react-native/src/bluetooth/BLEManager.ts)
for the full implementation.

### iOS BLE Permissions

On first launch the app requests Bluetooth permission. Tap "Allow."

If permission was previously denied:
iOS Settings → Privacy & Security → Bluetooth → MAIA → toggle ON.

### Connecting from the App

1. Wake the MYO armband (double-tap logo; LED pulses white).
2. Open the MAIA app, tap **Connect to MYO**.
3. The app scans for a device whose name starts with "Myo" and connects.
4. The status pill turns green: "MYO Connected."
5. EMG streaming begins automatically.

Note: If MyoConnect is running on the same phone or laptop and has claimed the
dongle connection, the phone's direct BLE connection will still work
independently — the two paths do not conflict.

---

## 7. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `Hub failed to start` | MyoConnect not running | Launch MyoConnect, confirm dongle is plugged in |
| `myo.init()` crash | Wrong `MYO_SDK_PATH` | Set `MYO_SDK_PATH` to the directory containing `myo.framework` (macOS) |
| No EMG data | `stream_emg(True)` not called | Call `event.device.stream_emg(True)` in `on_connected` |
| Flat signal on all channels | Armband too loose | Tighten until snug; all electrode pods must contact skin |
| Noisy signal at rest (> ±20 counts) | Skin dry or dirty | Clean with alcohol wipe; re-seat armband |
| One channel consistently near zero | Electrode pod not contacting skin | Loosen, rotate slightly, re-tighten |
| BLE scan timeout on iPhone | Armband asleep | Double-tap MYO logo to wake (LED pulses white) |
| App can't find "Myo" | iOS Bluetooth disabled | Enable Bluetooth in Control Center; grant permission in Settings |
| High 60 Hz noise | Powerline interference near dry skin | Clean skin, re-seat armband; notch filter (60 Hz) is applied automatically by the server |
| Armband disconnects mid-session | Low battery | Charge via micro-USB before session; battery indicator in MyoConnect |
| Firmware version warnings | Old firmware | Use MyoConnect's firmware update function (Help → Update Firmware) |

### Extended: Flat Channel During Clench

1. Remove and re-seat the armband — rotate it ±1 cm and re-tighten.
2. Check whether the flat channel corresponds to the extensor side (dorsal);
   these have lower amplitude than the flexor side. Try a stronger extension
   gesture (pull wrist back) to confirm the electrode is working.
3. If the channel is flat for all gestures on multiple tries, the electrode
   pod may be damaged — contact the MAIA team.

### Extended: No EMG After Connection

Ensure `stream_emg(True)` is called explicitly — the MYO does not stream EMG
by default. The call must happen inside the `on_connected` callback:

```python
def on_connected(self, event):
    event.device.stream_emg(True)
```

Without this call, `on_emg` will never fire even though the device is paired.
