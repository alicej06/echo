# EMG-ASL Mobile App

React Native / Expo companion app for the EMG-ASL Layer wristband. Connects to the EMG wristband over BLE, streams raw EMG data to the inference server, and speaks recognized ASL signs aloud using on-device TTS.

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Node.js | 20 LTS | https://nodejs.org/en/download |
| npm | 10+ | Bundled with Node 20 |
| Expo CLI | Latest | `npm install -g expo-cli` (or use `npx expo`) |
| Expo Go app | Latest | iOS App Store / Google Play Store |
| Physical iOS or Android device | — | **Required** — BLE does not work in simulators |

**macOS only** (for native iOS builds via EAS):
- Xcode 15+
- CocoaPods: `sudo gem install cocoapods`

---

## Setup

**Step 1 — Install Node dependencies**

```bash
cd mobile/react-native
npm install
```

**Step 2 — Configure server URL**

The app needs to know where your inference server is running. Copy the example env file and edit it:

```bash
cp .env.example .env
```

Open `.env` and set the IP address of the machine running the Python inference server. Use the machine's **LAN IP** (not `localhost` — your phone cannot reach `localhost` on your laptop):

```env
EXPO_PUBLIC_SERVER_URL=ws://192.168.1.42:8765
EXPO_PUBLIC_REST_URL=http://192.168.1.42:8000
EXPO_PUBLIC_BLE_DEVICE_NAME=Myo
```

To find your laptop's LAN IP:
- macOS: `ipconfig getifaddr en0`
- Linux: `hostname -I | awk '{print $1}'`
- Windows: `ipconfig` → look for IPv4 Address under your WiFi adapter

Your phone and laptop **must be on the same WiFi network**.

**Step 3 — Start the Expo dev server**

```bash
npx expo start
```

Scan the QR code with the **Expo Go** app on your phone. The app will load and bundle on your device.

> **BLE Note**: BLE functionality is unavailable in iOS Simulator and Android Emulator. You must test on a physical device. Also, iOS requires Bluetooth permission to be granted the first time the app launches — tap "Allow" when prompted.

---

## .env.example

```env
# Inference server WebSocket URL (BLE EMG streaming)
EXPO_PUBLIC_SERVER_URL=ws://192.168.1.x:8765

# Inference server REST API URL (calibration, health)
EXPO_PUBLIC_REST_URL=http://192.168.1.x:8000

# BLE device name to scan for (must match firmware DEVICE_NAME)
EXPO_PUBLIC_BLE_DEVICE_NAME=Myo
```

Replace `192.168.1.x` with your server machine's actual LAN IP address.

---

## Screen Descriptions

### Home Screen

The main screen during active sign recognition.

```
┌─────────────────────────────────┐
│  Myo  [●BLE] [●WS]   │  ← ConnectionStatusBar
│─────────────────────────────────│
│  ┌─────────────────────────┐   │
│  │  EMG Signal (8 channels) │   │  ← SignalMonitor (live waveforms)
│  │  ~~~~~~~~~~~~~~~~~~~~   │   │
│  │  ────────────────────   │   │
│  │  ~~~~~~~~~~            │   │
│  └─────────────────────────┘   │
│─────────────────────────────────│
│  "HELLO"              0.94 ▓▓▓ │  ← ASLPredictionDisplay (last prediction)
│─────────────────────────────────│
│  Transcript:                    │  ← Running transcript view
│  14:32:01  HELLO                │
│  14:32:03  MY                   │
│  14:32:05  NAME                 │
│                     [Clear]     │
└─────────────────────────────────┘
```

### Calibration Screen

Guides the user through collecting labeled EMG samples for per-user fine-tuning.

```
┌─────────────────────────────────┐
│  Calibration                    │
│─────────────────────────────────│
│                                 │
│     Hold sign:                  │
│                                 │
│       ┌───────────────────┐     │
│       │        A          │     │  ← Large letter/word prompt
│       └───────────────────┘     │
│                                 │
│  Progress: 12 / 50 samples      │
│  ████████░░░░░░░░░░░░  24%     │  ← CalibrationPrompt
│                                 │
│  [ Skip this sign ]             │
│  [ Done (5 signs left) ]        │
└─────────────────────────────────┘
```

### Settings Screen

Configure server connection, BLE pairing, TTS behavior, and inference parameters.

```
┌─────────────────────────────────┐
│  Settings                       │
│─────────────────────────────────│
│  Server                         │
│  WebSocket URL: [_____________] │
│  REST URL:      [_____________] │
│─────────────────────────────────│
│  BLE Device                     │
│  Name: [Myo        ]  │
│  [ Scan for devices ]           │
│─────────────────────────────────│
│  Text-to-Speech                 │
│  Rate:  ─────●────────  1.0x    │
│  Pitch: ─────●────────  1.0x    │
│─────────────────────────────────│
│  Inference                      │
│  Confidence: ────●───────  0.75 │
│  Debounce:   ────●───────  300ms│
│─────────────────────────────────│
│  User Profile                   │
│  [ Run Calibration ]            │
│  [ Delete Profile ]             │
└─────────────────────────────────┘
```

---

## Building for Production (EAS Build)

The app uses [Expo Application Services (EAS)](https://expo.dev/eas) for production builds.

**Step 1 — Install EAS CLI**

```bash
npm install -g eas-cli
eas login
```

**Step 2 — Configure EAS (one-time)**

```bash
eas build:configure
```

This creates `eas.json` in the project root.

**Step 3 — Build for iOS**

```bash
eas build --platform ios --profile production
```

**Step 4 — Build for Android**

```bash
eas build --platform android --profile production
```

Builds are queued on Expo's build servers (or a self-hosted EAS server). Download the `.ipa` / `.apk` when complete.

**Note on BLE in production builds**: `react-native-ble-plx` requires the following permissions in your app config (`app.json`):
- iOS: `NSBluetoothAlwaysUsageDescription`
- Android: `BLUETOOTH`, `BLUETOOTH_ADMIN`, `BLUETOOTH_CONNECT`, `BLUETOOTH_SCAN`

These are already configured in the project's `app.json`.

---

## Project Structure

```
mobile/react-native/
├── README.md               # This file
├── package.json
├── tsconfig.json
├── app.json                # Expo app config
├── .env.example            # Environment variable template
├── App.tsx                 # Root component (expo-router entry)
├── app/                    # expo-router screens (file-based routing)
│   ├── (tabs)/
│   │   ├── index.tsx       # Home / recognition screen
│   │   ├── calibration.tsx # Per-user calibration flow
│   │   └── settings.tsx    # App settings
│   └── _layout.tsx         # Root layout with tab navigator
└── src/
    ├── bluetooth/
    │   └── BLEManager.ts   # react-native-ble-plx wrapper
    ├── screens/
    │   ├── HomeScreen.tsx
    │   ├── CalibrationScreen.tsx
    │   └── SettingsScreen.tsx
    ├── components/
    │   ├── ASLPredictionDisplay.tsx
    │   ├── CalibrationPrompt.tsx
    │   └── ConnectionStatusBar.tsx
    ├── hooks/
    │   ├── useWebSocket.ts  # WebSocket connection + reconnect logic
    │   └── useSpeech.ts     # expo-speech TTS queue
    ├── inference/           # Client-side inference helpers (if used)
    └── tts/                 # TTS configuration helpers
```
