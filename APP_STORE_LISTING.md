# MAIA ASL — App Store Listing

## App Name (max 30 chars)

```
MAIA ASL
```

## Subtitle (max 30 chars)

```
EMG-Powered Sign Language
```

## Category

- Primary: **Medical**
- Secondary: **Utilities**

## Age Rating

**4+**

## Keywords (max 100 chars)

```
asl,sign language,emg,accessibility,deaf,gesture,wristband,sEMG,communication,myo
```

(82 chars)

---

## Description (max 4000 chars)

```
MAIA ASL translates American Sign Language directly from your muscles.

Strap on the Myo EMG wristband and form any letter from A to Z. Eight surface electromyography sensors read the electrical signals firing through your forearm. A machine learning model built specifically for ASL recognizes your gesture in under 200 milliseconds and speaks it aloud.

No camera. No computer vision. No line-of-sight requirement. Just muscle signals and math.

HOW IT WORKS
• Connect the Myo armband via Bluetooth in one tap
• Form any ASL letter — the app reads your forearm EMG in real time
• A confidence score shows exactly how certain the model is
• Auto-speak reads each recognized letter through the speaker
• Recognition history tracks your recent letters

CALIBRATION
Every hand is different. The one-time calibration flow adapts the model to your specific forearm geometry, improving accuracy for your personal signing style.

DEMO MODE
No hardware yet? Demo mode cycles through the full ASL alphabet so you can explore the interface before your Myo band arrives.

TECHNICAL
• 8-channel sEMG at 200Hz via Myo armband
• LSTM neural network classifier, trained on real ASL data
• Under 200ms end-to-end recognition latency
• On-device inference capable
• Railway WebSocket server for enhanced accuracy

PRIVACY
MAIA ASL does not collect, transmit, or store any personal data. EMG signals are processed locally on your device. No account required. See our full privacy policy at https://calebsnewton.github.io/maia-emg-asl/privacy
```

---

## What's New (v1.0.0)

```
First release. Connect your Myo EMG wristband and start recognizing ASL letters in real time.
```

---

## Support URL

```
https://github.com/calebnewtonusc/maia-emg-asl
```

## Privacy Policy URL

```
https://calebsnewton.github.io/maia-emg-asl/privacy
```

(Enable GitHub Pages on the repo — source: /docs folder — to activate this URL)

---

## App Store Screenshots Needed

Apple requires screenshots for these device sizes:

| Device                   | Size           | Required               |
| ------------------------ | -------------- | ---------------------- |
| iPhone 6.7" (15 Pro Max) | 1290 x 2796 px | Yes                    |
| iPhone 6.5" (14 Plus)    | 1284 x 2778 px | Yes                    |
| iPhone 5.5" (8 Plus)     | 1242 x 2208 px | Recommended            |
| iPad Pro 12.9"           | 2048 x 2732 px | Only if iPad supported |

**Screens to capture:**

1. ASL Live — showing a recognized letter (e.g. "A") with high confidence
2. ASL Live — Myo connected, streaming state
3. Calibration screen — step in progress
4. Demo screen — alphabet cycling
5. Settings screen

**Use Stora** (`stora_generate_screenshots`) to auto-generate all sizes after restart.

---

## App Store Connect Checklist

- [ ] Version: 1.0.0 / Build: 1
- [ ] Bundle ID: com.maiatech.emgasl
- [ ] Category: Medical
- [ ] Age Rating: 4+
- [ ] Privacy Policy URL live
- [ ] Support URL live (GitHub repo)
- [ ] Screenshots uploaded (all required sizes)
- [ ] App Review notes: "Requires Myo EMG wristband hardware. Use Demo tab to test without hardware."
- [ ] Export compliance: ITSAppUsesNonExemptEncryption = false (already set)
- [ ] Submit for review
