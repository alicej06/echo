/**
 * ONNXInference.ts
 * On-device ASL classification using the ONNX Runtime for React Native.
 *
 * Model contract:
 *   Input  : "input"  -- float32 tensor of shape [1, 320]  (40 samples x 8 channels, row-major)
 *   Output : "output" -- float32 tensor of shape [1, 36]   (raw logits, one per ASL class)
 *
 * The raw int16 values arriving from the BLE armband are pre-centred by the
 * firmware (device subtracts 2048 before transmission), so dividing by 2048
 * maps them to the [-1, 1] range that the model was trained on.
 *
 * Usage:
 *   await onDeviceInference.loadModel();
 *   const result = await onDeviceInference.predict(windowData); // windowData: Int16Array(320)
 *   if (result) console.log(result.label, result.confidence);
 */

import { InferenceSession, Tensor } from 'onnxruntime-react-native';

// ---------------------------------------------------------------------------
// ASL label vocabulary -- 26 letters followed by 10 common-word tokens.
// Index i in this array corresponds to output logit i from the model.
// ---------------------------------------------------------------------------

const ASL_LABELS: readonly string[] = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
  'U', 'V', 'W', 'X', 'Y', 'Z',
  'HELLO', 'THANK_YOU', 'PLEASE', 'SORRY',
  'YES', 'NO', 'HELP', 'MORE', 'STOP', 'I_LOVE_YOU',
];

const NUM_CLASSES = ASL_LABELS.length; // 36

// ---------------------------------------------------------------------------
// Inference constants
// ---------------------------------------------------------------------------

/** Normalisation divisor: firmware centres at 0, max absolute value is 2048. */
const INT16_NORM = 2048;

/** Number of float32 values expected per inference call (40 samples x 8 ch). */
const FEATURE_LENGTH = 320;

/**
 * Predictions whose softmax probability falls below this threshold are
 * suppressed and cause predict() to return null.
 */
const CONFIDENCE_THRESHOLD = 0.75;

/** Minimum milliseconds between consecutive accepted predictions. */
const DEBOUNCE_MS = 300;

// ---------------------------------------------------------------------------
// Path to the bundled ONNX model
// ---------------------------------------------------------------------------

/**
 * React Native's Metro bundler resolves asset paths at build time.
 * The model file must be placed at:
 *   mobile/react-native/assets/model/asl_emg_classifier.onnx
 *
 * On iOS the asset is copied into the app bundle; on Android it ends up
 * in the APK's assets/ directory.  onnxruntime-react-native's
 * InferenceSession.create() accepts the require() result directly.
 */
// eslint-disable-next-line @typescript-eslint/no-var-requires
const MODEL_ASSET = require('../../assets/model/asl_emg_classifier.onnx');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Numerically stable softmax over a plain Float32Array.
 * Subtracts the maximum value before exponentiation to prevent overflow.
 */
function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }

  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i] - max);
    sum += exps[i];
  }

  for (let i = 0; i < exps.length; i++) {
    exps[i] /= sum;
  }

  return exps;
}

// ---------------------------------------------------------------------------
// ONNXInferenceService
// ---------------------------------------------------------------------------

/**
 * Thin wrapper around an ONNX Runtime InferenceSession that handles:
 *   - model loading from the app bundle
 *   - int16 -> float32 normalisation
 *   - softmax post-processing
 *   - confidence thresholding
 *   - 300 ms prediction debounce
 */
class ONNXInferenceService {
  private session: InferenceSession | null = null;

  // Debounce state
  private lastLabel: string | null = null;
  private lastPredictionTime = 0;

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Load the ONNX model from the app bundle into memory.
   * Must be awaited before any call to predict().
   * Safe to call multiple times -- subsequent calls are no-ops if the session
   * is already open.
   */
  async loadModel(): Promise<void> {
    if (this.session !== null) {
      return; // Already loaded
    }

    try {
      this.session = await InferenceSession.create(MODEL_ASSET, {
        executionProviders: ['cpu'],
      });
      console.log('[ONNXInference] Model loaded successfully.');
    } catch (err) {
      console.error('[ONNXInference] Failed to load model:', err);
      throw err;
    }
  }

  /**
   * Returns true when the ONNX session is ready for inference.
   */
  isModelLoaded(): boolean {
    return this.session !== null;
  }

  /**
   * Classify a single EMG window.
   *
   * @param windowData - Flat Int16Array of length 320 containing 40 samples
   *   across 8 channels in row-major order:
   *   [ch0_t0, ch1_t0, ..., ch7_t0, ch0_t1, ch1_t1, ..., ch7_t39].
   *   Values are pre-centred by the firmware (range roughly -2048 to +2048).
   *
   * @returns An object {label, confidence} if the top class clears
   *   CONFIDENCE_THRESHOLD (0.75) and the 300 ms debounce has elapsed;
   *   otherwise null.
   */
  async predict(windowData: Int16Array): Promise<{ label: string; confidence: number } | null> {
    if (!this.session) {
      console.warn('[ONNXInference] predict() called before loadModel().');
      return null;
    }

    if (windowData.length !== FEATURE_LENGTH) {
      console.warn(
        `[ONNXInference] Expected ${FEATURE_LENGTH} samples, got ${windowData.length}.`,
      );
      return null;
    }

    // --- Normalise int16 -> float32 in [-1, 1] ---
    const floatData = new Float32Array(FEATURE_LENGTH);
    for (let i = 0; i < FEATURE_LENGTH; i++) {
      floatData[i] = windowData[i] / INT16_NORM;
    }

    // --- Build input tensor: shape [1, 320] ---
    const inputTensor = new Tensor('float32', floatData, [1, FEATURE_LENGTH]);

    // --- Run inference ---
    let outputMap: Record<string, Tensor>;
    try {
      outputMap = await this.session.run({ input: inputTensor });
    } catch (err) {
      console.error('[ONNXInference] Inference error:', err);
      return null;
    }

    // --- Extract logits from output tensor (shape [1, 36]) ---
    const outputTensor = outputMap['output'];
    if (!outputTensor) {
      console.error('[ONNXInference] Model output key "output" not found.');
      return null;
    }

    const logits = outputTensor.data as Float32Array;

    if (logits.length !== NUM_CLASSES) {
      console.error(
        `[ONNXInference] Expected ${NUM_CLASSES} logits, got ${logits.length}.`,
      );
      return null;
    }

    // --- Softmax -> probabilities ---
    const probs = softmax(logits);

    // --- Find argmax ---
    let topIdx = 0;
    let topProb = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > topProb) {
        topProb = probs[i];
        topIdx = i;
      }
    }

    // --- Confidence threshold ---
    if (topProb < CONFIDENCE_THRESHOLD) {
      return null;
    }

    const topLabel = ASL_LABELS[topIdx];

    // --- Debounce: suppress repeated same-label predictions within 300 ms ---
    const now = Date.now();
    if (topLabel === this.lastLabel && now - this.lastPredictionTime < DEBOUNCE_MS) {
      return null;
    }

    this.lastLabel = topLabel;
    this.lastPredictionTime = now;

    return { label: topLabel, confidence: topProb };
  }

  /**
   * Release the ONNX session and free native memory.
   * Call this when the screen or component using inference is unmounted.
   */
  dispose(): void {
    if (this.session) {
      // InferenceSession does not expose an async close() in all RN builds;
      // release the reference and let the native side GC the session.
      this.session = null;
      this.lastLabel = null;
      this.lastPredictionTime = 0;
      console.log('[ONNXInference] Session disposed.');
    }
  }
}

// ---------------------------------------------------------------------------
// Singleton export
// ---------------------------------------------------------------------------

/** Application-wide singleton. Import this instead of constructing directly. */
export const onDeviceInference = new ONNXInferenceService();
