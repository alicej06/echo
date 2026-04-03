/**
 * Sliding window buffer for real-time Myo Band EMG data.
 *
 * Myo notification format: 16 bytes = 2 samples × 8 channels × int8
 * Window: 40 samples × 8 channels = 320 floats, 50% overlap (20-sample hop)
 * Effective sample rate: 200 Hz (4 EMG characteristics × 50 Hz × 2 samples)
 */

const N_CHANNELS = 8;
const WINDOW_SAMPLES = 40;
const HOP_SAMPLES = 20; // 50% overlap
const INT8_SCALE = 1 / 128;

export type WindowCallback = (window: Float32Array) => void;

export class EMGWindowBuffer {
  private buffer: Float32Array;
  private writePtr: number = 0;
  private samplesUntilEmit: number = HOP_SAMPLES;
  private onWindow: WindowCallback;

  constructor(onWindow: WindowCallback) {
    this.buffer = new Float32Array(WINDOW_SAMPLES * N_CHANNELS);
    this.onWindow = onWindow;
  }

  /**
   * Feed a 16-byte Myo EMG notification (2 samples × 8 channels × int8).
   * Called for each BLE notification from any of the 4 EMG characteristics.
   */
  feedMyo(data: Uint8Array): void {
    if (data.length < 16) return;
    for (let s = 0; s < 2; s++) {
      const sample = new Float32Array(N_CHANNELS);
      for (let ch = 0; ch < N_CHANNELS; ch++) {
        // Myo sends unsigned bytes — reinterpret as signed int8
        const raw = data[s * 8 + ch];
        sample[ch] = (raw > 127 ? raw - 256 : raw) * INT8_SCALE;
      }
      this.feedSample(sample);
    }
  }

  private feedSample(sample: Float32Array): void {
    const offset = (this.writePtr % WINDOW_SAMPLES) * N_CHANNELS;
    this.buffer.set(sample, offset);
    this.writePtr++;

    if (this.writePtr < WINDOW_SAMPLES) return;

    this.samplesUntilEmit--;
    if (this.samplesUntilEmit <= 0) {
      this.samplesUntilEmit = HOP_SAMPLES;
      this.emitWindow();
    }
  }

  private emitWindow(): void {
    const window = new Float32Array(WINDOW_SAMPLES * N_CHANNELS);
    const start = this.writePtr % WINDOW_SAMPLES;
    for (let i = 0; i < WINDOW_SAMPLES; i++) {
      const src = ((start + i) % WINDOW_SAMPLES) * N_CHANNELS;
      window.set(this.buffer.subarray(src, src + N_CHANNELS), i * N_CHANNELS);
    }
    this.onWindow(window);
  }

  reset(): void {
    this.buffer.fill(0);
    this.writePtr = 0;
    this.samplesUntilEmit = HOP_SAMPLES;
  }
}
