# EMG-ASL Data Collection Protocol v1.0

**Protocol Version**: 1.0
**Date**: February 2026
**Project**: EMG-ASL Layer -- Real-Time ASL Sign Recognition via Forearm sEMG
**Institution**: MAIA Biotech / [University Name -- insert before IRB submission]

> **IRB Notice**: This document is intended as an internal operational guide and as a
> template for IRB submission. Before collecting data from human participants beyond
> your own pilot sessions, obtain Institutional Review Board (IRB) approval from your
> institution. Adapt the consent section to your IRB's approved form. Do not begin
> multi-participant data collection without a valid IRB approval number.

---

## Table of Contents

1. [Study Overview](#1-study-overview)
2. [Inclusion and Exclusion Criteria](#2-inclusion-and-exclusion-criteria)
3. [Consent and Privacy](#3-consent-and-privacy)
4. [Equipment Setup](#4-equipment-setup)
5. [Recording Protocol](#5-recording-protocol)
6. [Data Quality Checks](#6-data-quality-checks)
7. [File Naming and Storage](#7-file-naming-and-storage)
8. [Quick Reference Card](#8-quick-reference-card)

---

## 1. Study Overview

### Purpose

This study collects surface electromyography (sEMG) signals from the forearm muscles
of participants while they perform American Sign Language (ASL) alphabet handshapes.
The recorded data are used to train and evaluate machine learning classifiers that
recognize ASL signs in real time from muscle activity alone.

### Equipment

**Primary rig (currently in use):**
- **Thalmic MYO Armband** — 8-channel sEMG wristband, 200 Hz, int8 output.
  Pre-assembled; no custom wiring or firmware required.
- **MYO USB Bluetooth Dongle** + **MyoConnect** software — manages the BLE
  connection from the laptop.
- **myo-python** SDK — Python wrapper used by `scripts/record_session.py` to
  stream raw EMG from the armband to the data collection pipeline.
- Optional: iPhone running the MAIA React Native app (direct BLE, no dongle)
  for fully untethered live inference.

### Data Collected

- 8-channel forearm sEMG sampled at 200 Hz.
- Timestamped sign labels (A-Z), repetition index, hold duration, and quality flags.
- Session metadata: participant code, hardware version, firmware version, baseline
  noise levels, and experimenter notes.

No personally identifiable information (PII) is collected or stored in data files.

---

## 2. Inclusion and Exclusion Criteria

### Inclusion Criteria

Participants must meet **all** of the following:

- Age 18 years or older.
- Right-handed (dominant right hand) OR left-handed with an explicit note recorded in
  session metadata. Left-handed participants use the left forearm; all placement
  diagrams are mirrored accordingly.
- No known neuromuscular conditions affecting the dominant forearm or hand (e.g.,
  essential tremor, peripheral neuropathy, carpal tunnel syndrome, Parkinson's
  disease).
- Able to form each of the 26 ASL alphabet handshapes, or willing to learn from
  reference images provided by the researcher.
- Willing to participate for approximately 15-20 minutes per session.

### Exclusion Criteria

Participants are excluded if they meet **any** of the following:

- Active skin condition, open wound, rash, or dermatitis on the dominant forearm that
  would prevent secure electrode placement.
- Presence of an active implanted electronic device: pacemaker, cochlear implant,
  spinal cord stimulator, or deep brain stimulator. EMG electrode use is
  contraindicated in these cases.
- Pregnancy (precautionary exclusion; EMG is generally considered safe, but
  participant comfort and signal quality may be affected).
- Known allergy to electrode adhesive or isopropyl alcohol.

### ASL Experience -- Optional Stratification

For studies targeting a representative dataset, record self-reported ASL experience
in session metadata:

| Group       | Definition                             |
|-------------|----------------------------------------|
| Naive       | No prior ASL knowledge                 |
| Intermediate| 1-3 years of ASL study or casual use   |
| Fluent      | Native or near-native ASL signer       |

Naive participants are guided through each sign using the printed reference chart
(see Section 5).

---

## 3. Consent and Privacy

### IRB Waiver of Consent Eligibility

This study is designed to qualify for an IRB waiver of written informed consent under
45 CFR 46.116(c) because:

1. The research involves no greater than minimal risk to participants.
2. The data collected are anonymous by design -- no PII is recorded.
3. The waiver will not adversely affect the rights and welfare of subjects.
4. The research could not practicably be carried out without the waiver (anonymous
   collection at scale).

Submit this protocol to your institution's IRB to confirm waiver eligibility. If the
IRB requires written consent, use the consent template in `docs/consent-form.md`
(create from the description in Section 2 of the full IRB protocol).

### Data Anonymization

- Participant IDs are sequential codes: P001, P002, P003, and so on. No PII is
  collected or stored in any data file.
- If a linkage file (mapping participant IDs to real names) is maintained for
  follow-up contact, it must be stored in a separate, encrypted location that is NOT
  committed to any git repository.
- Data files use participant ID only as the identifier.

### Storage and Sharing

- All session data is stored locally on the experimenter's laptop and/or an encrypted
  external drive.
- Data is never uploaded to any cloud service without explicit opt-in by the
  participant and IRB approval for cloud storage.
- Trained model weights derived from this data may be published. Raw sEMG signals are
  not embedded in model weights.

### Data Retention

Raw session CSVs and metadata files are retained for 3 years after study completion,
then securely deleted in accordance with institutional data retention policy.

---

## 4. Equipment Setup

### 4.1 Parts Used in This Protocol

| Component                                   | Qty | Notes |
|---------------------------------------------|-----|-------|
| **Thalmic MYO Armband**                     | 1   | Pre-assembled 8-ch sEMG, 200 Hz, int8 |
| **MYO USB Bluetooth Dongle**                | 1   | Required for laptop data collection (MyoConnect) |
| Laptop running MyoConnect + myo-python      | 1   | macOS or Windows; see `docs/hardware-setup.md` |
| Isopropyl alcohol wipes (70%)               | box | For skin prep before session |
| Gauze pads or skin prep tape                | box | Light abrasion to lower electrode impedance |

The MYO Armband is a self-contained unit — no individual electrodes, cables,
or custom wiring are required. Full setup instructions are in `docs/hardware-setup.md`.

### 4.2 Armband Placement Diagram

Place the 8 sensors circumferentially around the dominant forearm, centered
approximately 2-3 cm distal (toward wrist) from the elbow crease. The exact
placement targets the forearm flexor and extensor muscle bellies.

```
  RIGHT FOREARM -- cross-section view (looking from elbow toward wrist)
  Imagine the forearm as a clock face.

                  DORSAL (back of hand up)
                        12 o'clock
                        [ Ch6 ]
             [ Ch5 ]              [ Ch7 ]
          10 o'clock               2 o'clock

  RADIAL                                     ULNAR
  (thumb side)                           (pinky side)
          [ Ch4 ]               [ Ch0 ]
           9 o'clock             3 o'clock

             [ Ch3 ]              [ Ch1 ]
          8 o'clock               4 o'clock
                        [ Ch2 ]
                        6 o'clock
                  PALMAR (palm facing up)

  Ground (reference) electrode: bony tip of elbow (olecranon process).

  Channel-to-muscle mapping (approximate):
    Ch0, Ch1 -- Flexor Carpi Radialis (palmar-radial)
    Ch2, Ch3 -- Flexor Digitorum Superficialis (palmar)
    Ch4, Ch5 -- Extensor Carpi Radialis (dorsal-radial)
    Ch6, Ch7 -- Extensor Digitorum Communis (dorsal)
```

Channel-to-muscle mapping is approximate; MYO logo (Ch0) faces the palmar side.

### 4.3 Skin Preparation and Armband Placement

1. Ask participant to expose the dominant forearm and rest it palm-up on the table.
2. Identify the placement ring: 2–3 cm below the elbow crease, at the widest
   circumferential point of the upper forearm.
3. For long sessions (> 30 minutes), prepare the skin:
   a. Wipe the forearm ring with an isopropyl alcohol wipe (70%).
   b. Allow to dry for at least 30 seconds.
   c. Gently abrade with gauze (2–3 passes) to lower contact impedance.
   d. Allow skin to air dry completely before fitting the armband.
4. Place the MYO Armband at the target ring, with the MYO logo facing the palmar
   (inner wrist) side, and tighten until snug — all 8 electrode pods must
   maintain firm contact with the skin.
5. Wake the armband (double-tap the MYO logo; LED pulses white) and confirm
   it connects in MyoConnect (LED turns solid green).
6. Confirm signal quality via the Signal Monitor before recording begins.

---

## 5. Recording Protocol

### 5.1 Pre-Recording Warm-Up

Before any labeled data is recorded, allow the participant 60 seconds of free
movement:

- Ask the participant to open and close the hand, rotate the wrist, and flex and
  extend the fingers at their own pace.
- This warms up the forearm muscles and allows the electrode-skin interface to
  stabilize.
- The experimenter monitors the Signal Monitor screen during warm-up and confirms
  that all 8 channels show responsive signals during movement and a quiet baseline
  at rest.

### 5.2 Sign Recording -- 26 ASL Letters

**Target**: 5 repetitions minimum per letter (10 repetitions recommended for training
data quality).

**Cue procedure per trial:**

```
[Screen or audio cue]: "Ready -- [LETTER]"     <- 1 second warning; participant
                                                   looks at reference image
[Screen or audio cue]: "HOLD NOW"              <- participant forms and holds sign
[Silence for 3 seconds]                        <- sEMG recorded during hold
[Screen or audio cue]: "RELAX"                 <- participant returns to neutral
[Rest for 1 second]                            <- EMG returns to baseline
```

- Hold duration: 3 seconds per sign.
- Rest between signs: 1 second.
- Reference: researcher shows a printed or on-screen ASL alphabet reference image
  before each sign cue. The participant mirrors the handshape shown.
- Randomize letter order per participant to prevent ordering effects. Use
  `numpy.random.seed(participant_id_integer)` for reproducibility.

### 5.3 Session Duration

| Phase            | Time       |
|------------------|------------|
| Equipment setup  | 5 min      |
| Warm-up          | 1 min      |
| 26 letters x 5 reps (3 s hold + 1 s rest) = 26 x 5 x 4 s | ~8-9 min |
| 26 letters x 10 reps                       | ~17-18 min |
| Break (recommended at 10 min mark)         | 2-3 min    |
| Validate + re-record failed letters        | 2-5 min    |
| **Total (5 rep session)**                  | **~15 min**|
| **Total (10 rep session)**                 | **~25 min**|

### 5.4 Break Schedule

- Offer a 2-minute break after the first 13 letters (midpoint of alphabet).
- Allow additional breaks on participant request at any time.
- If a participant reports forearm fatigue or discomfort, end the session and retain
  all data collected to that point. Mark the session as partial in metadata.

---

## 6. Data Quality Checks

### 6.1 Automated Validation

Run the validation script immediately after each session:

```bash
python scripts/validate_session.py data/raw/P001_20260301_143022.csv
```

The script checks:
- Window count per label: each letter must have at least 3 valid (non-rejected)
  windows extracted from the hold period.
- Clipping rate: the fraction of samples per channel that saturate the ADC
  (raw value >= 4090 or <= 5).
- Baseline noise: pre-sign RMS per channel.

**Pass/fail thresholds:**

| Check                         | Pass             | Action if Fail                     |
|-------------------------------|------------------|------------------------------------|
| Valid windows per letter      | >= 3             | Re-record that letter              |
| Clipping rate per session     | < 5%             | Re-seat all electrodes; re-record  |
| Baseline RMS at rest          | < 15 mV per ch   | Re-seat noisy channel electrode    |
| Channels with valid signal    | >= 6 of 8        | Re-seat bad channel; re-record     |

### 6.2 Re-Recording Procedure

If any letter fails the minimum window count:

1. Note which letters failed (the script prints a list).
2. Without removing the armband, re-record only the failed letters.
3. Use the `--labels` flag to target specific letters:

```bash
python scripts/record_session.py \
  --participant P001 \
  --output data/raw/ \
  --labels F J X \
  --reps 5
```

4. Re-run `validate_session.py` on the new partial session file.
5. Merge partial file into the main session file if your pipeline supports it, or
   treat it as a supplementary file (name: `P001_20260301_143022_rerecord.csv`).

### 6.3 Electrode Repositioning

If the clipping rate exceeds 5% across all channels:

- Remove the armband entirely.
- Clean skin again with alcohol wipes.
- Re-apply fresh electrode pads (do not reuse pads after removal).
- Re-run the calibration verification procedure in `docs/hardware-setup.md`
  Section 1.5 before recording again.

---

## 7. File Naming and Storage

### Directory Structure

```
data/
  raw/
    P001_20260301_143022.csv     <- auto-generated by record_session.py
    P001_20260301_143022_metadata.json
    P002_20260301_154511.csv
    P002_20260301_154511_metadata.json
  processed/                     <- generated by preprocessing pipeline
  external/                      <- public datasets (NinaProDB, GRABMyo, etc.)
```

File naming convention: `{participant_id}_{YYYYMMDD}_{HHMMSS}.csv`
The timestamp is local time at session start (or UTC if the machine is set to UTC).
Record the timezone in session metadata.

### CSV Column Schema

```
timestamp_ms   int64    Unix timestamp in milliseconds at sample receipt
sample_index   int64    Sample number within session (monotonically increasing)
ch0            float32  Channel 0 sEMG in millivolts (before filtering)
ch1            float32  Channel 1 sEMG in mV
ch2            float32  Channel 2 sEMG in mV
ch3            float32  Channel 3 sEMG in mV
ch4            float32  Channel 4 sEMG in mV
ch5            float32  Channel 5 sEMG in mV
ch6            float32  Channel 6 sEMG in mV
ch7            float32  Channel 7 sEMG in mV
label          str      ASL letter ("A" through "Z") or word token
rep_index      int8     Repetition index within this label (0-indexed)
trial_id       str      Unique ID: "{session_id}_{label}_{rep_index}"
rejected       bool     True if trial was flagged as low quality
reject_reason  str      Reason code; empty string if not rejected
```

Reject reason codes: `HIGH_BASELINE`, `SATURATION`, `WRONG_SIGN`, `CHANNEL_FAULT`,
`PARTICIPANT_REQUEST`.

Rejected rows are retained in the CSV -- they are not deleted. This allows future
re-analysis with different rejection thresholds.

### gitignore

All contents of `data/raw/` and `data/processed/` are gitignored. Never commit raw
sEMG data or session metadata to the repository.

---

## 8. Quick Reference Card

Print and post at the recording station.

```
=========================================================
  EMG-ASL DATA COLLECTION -- QUICK REFERENCE v1.0
=========================================================

BEFORE EACH SESSION
  [ ] MYO Armband battery charged (check MyoConnect battery indicator)
  [ ] MyoConnect running, MYO armband shows green LED in Devices panel
  [ ] Inference server running: ./start-server.sh
  [ ] Signal Monitor screen open on laptop or phone

ARMBAND PLACEMENT
  1. Expose dominant forearm
  2. (Long sessions) Clean forearm ring with alcohol wipe -- wait 30 s to dry
  3. (Long sessions) Abrade lightly with gauze
  4. Place MYO Armband 2-3 cm below elbow crease, logo facing palmar side
  5. Tighten until snug -- all electrode pods contact skin
  6. Double-tap MYO logo to wake (LED pulses white → solid green when connected)

SIGNAL CHECK (before recording)
  - At rest: all channels RMS < 15 mV    <- pass
  - Fist clench: >= 4 channels RMS > 100 mV   <- pass
  - If fail: re-seat the electrode on the noisy channel

WARM-UP
  Ask participant to open/close hand freely for 60 seconds.

RECORDING
  python scripts/record_session.py \
    --participant P00N \
    --output data/raw/ \
    --labels A B C D E F G H I J K L M N O P Q R S T U V W X Y Z \
    --reps 5

  - Cue: "Ready -- [LETTER]" then "HOLD NOW" (3 s) then "RELAX" (1 s)
  - Show reference image for each letter
  - Offer break at midpoint (after letter M)

VALIDATE IMMEDIATELY AFTER
  python scripts/validate_session.py data/raw/P00N_*.csv

  If any letter shows < 3 valid windows: re-record just those letters
    python scripts/record_session.py --participant P00N \
      --output data/raw/ --labels [FAILED LETTERS] --reps 5

  If clipping rate > 5%: re-seat all electrodes, re-record full session

FILE SAVED AT
  data/raw/P00N_YYYYMMDD_HHMMSS.csv
=========================================================
```
