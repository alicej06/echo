# Vision Teacher Guide

How to use MediaPipe-based auto-labeling to produce a labeled EMG dataset without
manual annotation.

---

## 1. Why Vision + EMG

Collecting labeled EMG training data is the hardest part of building an ASL
classifier. The naive approach -- have a person hold each hand sign while you
press a key to mark the label -- is slow, inconsistent, and doesn't scale past
a handful of letters.

The **vision teacher** approach offloads labeling to a webcam. You record your
forearm EMG and a video of your hand at the same time. After the session,
MediaPipe inspects every frame of the video, predicts which ASL letter the hand
is forming, and attaches those predictions to the matching EMG samples. The
result is a labeled session CSV that `train_real.py` can consume directly.

This means:
- No keyboard annotations during recording.
- Labels are continuous and dense (one per video frame, roughly 30 Hz).
- The session feels natural because the signer just signs at their own pace.
- Once the pose classifier is trained, the pipeline is fully automated.

---

## 2. The Cross-Modal Pipeline

```
Webcam/Video --> MediaPipe Hands --> Hand Landmarks (63-dim)
                                            |
                                    ASL Pose Classifier
                                    (SimpleASLPoseClassifier)
                                            |
                                Timestamped Labels (A-Z, conf)
                                            |
                                sync_labels_to_emg()
                                            |
                Raw EMG CSV -------------> Labeled Session CSV
                                            |
                                    train_real.py --> ONNX model
```

Each stage explained:

| Stage | File / function | Notes |
|---|---|---|
| Video capture | `cv2.VideoCapture` | Webcam (live) or MP4 (offline) |
| Hand detection | `HandLandmarkExtractor.extract()` | MediaPipe Hands, 21 landmarks |
| Normalization | Inside `extract()` | Origin at wrist, scale to [-1, 1] |
| Classification | `SimpleASLPoseClassifier.predict_proba()` | sklearn SVC, 26-class |
| Sync | `sync_labels_to_emg()` | Nearest-neighbor, 50 ms tolerance |
| Training | `train_real.py` | Uses labeled CSV as input |

---

## 3. Current Limitations

**Static letters only (A-Z).** A single video frame captures a frozen hand
shape. Letters like A, B, C, D map cleanly to static poses. Dynamic words such
as HELLO or THANK_YOU involve motion trajectories that cannot be extracted from
a single frame. Classifying those requires a temporal vision model (e.g. an
LSTM over a sequence of landmark frames or an optical-flow CNN). That is out of
scope for the current `SimpleASLPoseClassifier`.

**Untrained baseline.** The classifier ships with no weights. Until you run the
training procedure described in Section 4, every prediction returns `UNKNOWN`
with confidence 0.0. The landmark extraction pipeline is real and functional;
the bottleneck is the absence of training data, not missing code.

**Single-hand, frontal view.** MediaPipe Hands performs best when the hand is
clearly visible, well-lit, and not occluded. Accuracy drops at sharp angles or
under low-light conditions.

**Clock drift.** The video and EMG streams run on separate clocks. The 50 ms
sync tolerance absorbs small drift, but for sessions longer than a few minutes
on hardware without a shared hardware clock you may need to insert sync pulses
(e.g. briefly pressing a BLE button that is timestamped in both streams) and
do affine clock alignment before calling `sync_labels_to_emg`.

---

## 4. Training the Pose Classifier

You do not need an EMG band to train the pose classifier. All you need is a
webcam and some patience.

### 4.1 Collect training frames

Run the live webcam stream and redirect output to a file, or write a short
script that:

1. Opens the webcam with `HandLandmarkExtractor`.
2. Shows a prompt on screen: "Hold sign A for 3 seconds".
3. Extracts landmarks from every frame during the hold period.
4. Saves `(landmark_vector, label)` pairs to a CSV or numpy file.

Example pseudocode:

```python
import numpy as np
from src.data.vision_teacher import HandLandmarkExtractor
import cv2, time

extractor = HandLandmarkExtractor()
cap = cv2.VideoCapture(0)

X, y = [], []
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    input(f"Hold sign {letter} and press Enter ...")
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if not ok:
            continue
        lm = extractor.extract(frame)
        if lm is not None:
            X.append(lm)
            y.append(letter)

X = np.array(X)   # shape (N, 63)
y = np.array(y)   # shape (N,)
np.save("data/pose_X.npy", X)
np.save("data/pose_y.npy", y)
```

Aim for at least 50 repetitions per letter across varied hand positions to
improve generalization.

### 4.2 Train and save

```python
from src.data.vision_teacher import SimpleASLPoseClassifier
import numpy as np

X = np.load("data/pose_X.npy")
y = np.load("data/pose_y.npy")

clf = SimpleASLPoseClassifier()
clf.fit(X, y)
clf.save("models/pose_classifier.joblib")
print("Saved.")
```

### 4.3 Use the trained classifier

Pass `--classifier models/pose_classifier.joblib` to `auto_label_session.py`,
or load it directly:

```python
from src.data.vision_teacher import VisionTeacher, load_trained_classifier

teacher = VisionTeacher()
teacher._classifier = load_trained_classifier("models/pose_classifier.joblib")
label_df = teacher.label_video_file("data/raw/P001_session.mp4")
```

### 4.4 Evaluate

Check per-class accuracy with a held-out validation set before trusting
auto-labeled EMG data. A classifier with less than ~85% letter accuracy will
introduce enough label noise to hurt downstream EMG model performance.

---

## 5. Advanced: Cross-Modal Embedding (Future Work)

The current pipeline relies on an explicit pose classifier trained on
hand-landmark vectors. A more powerful long-term direction is **cross-modal
embedding alignment**, sometimes called a CLIP-style approach.

The idea is to train a shared embedding space where:
- A vision encoder (e.g. a ViT fine-tuned on hand images) produces embeddings
  for video frames showing each ASL letter.
- An EMG encoder (e.g. the existing CNN-LSTM) produces embeddings for EMG
  windows.
- A contrastive loss (InfoNCE) pulls together embeddings of the same letter and
  pushes apart different letters.

Once both encoders are aligned, you can classify an EMG window by computing its
embedding and finding the nearest vision embedding in the shared space. This
enables **zero-shot generalization**: a new sign that appears only in video data
(no EMG recordings needed) can be classified because its vision embedding acts
as a proxy for what the EMG embedding should look like.

Practical steps when hardware arrives:
1. Train the vision encoder on a large ASL video dataset (e.g. MS-ASL, WLASL).
2. Record paired EMG + video for a small set of letters to anchor the EMG
   encoder into the shared space.
3. Use InfoNCE contrastive loss with vision embeddings as the positive targets.
4. At inference time the EMG encoder alone is used -- no camera needed on the
   end device.

This path is the reason `SimpleASLPoseClassifier` is designed as a replaceable
component: the `VisionTeacher` class is agnostic to what generates the
per-frame label, making it straightforward to swap in a learned vision encoder
later.
