/**
 * EMGStream.ts
 * Processes raw BLE bytes into windowed Float32Array frames suitable for
 * the inference model.
 *
 * Assumptions (adjust to match your armband firmware):
 *   - 8 EMG channels
 *   - 200 Hz sample rate
 *   - Each sample is a signed 16-bit integer (little-endian)
 *   - Each BLE packet contains one or more complete samples
 *     (packet format: [ch0_lo, ch0_hi, ch1_lo, ch1_hi, ... ch7_lo, ch7_hi])
 *
 * Window parameters:
 *   - Window length : 200 ms  →  40 samples at 200 Hz
 *   - Overlap       : 50 %    →  advance by 20 samples per emission
 */

import { EventEmitter } from "events";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const SAMPLE_RATE_HZ = 200;
export const NUM_CHANNELS = 8;
export const WINDOW_DURATION_MS = 200;
export const OVERLAP_FRACTION = 0.5;

/** Number of samples in one window. */
export const WINDOW_SAMPLES = Math.round(
  (SAMPLE_RATE_HZ * WINDOW_DURATION_MS) / 1000,
); // 40

/** Number of samples to advance before emitting the next window. */
export const HOP_SAMPLES = Math.round(WINDOW_SAMPLES * (1 - OVERLAP_FRACTION)); // 20

/** Bytes per sample across all channels (2 bytes per int16 × 8 channels). */
const BYTES_PER_SAMPLE = NUM_CHANNELS * 2;

/** Ring buffer capacity in samples (holds 2 full windows for safety). */
const RING_CAPACITY = WINDOW_SAMPLES * 2;

// ---------------------------------------------------------------------------
// Event map
// ---------------------------------------------------------------------------

export interface EMGStreamEvents {
  window: (data: Float32Array) => void;
  error: (err: Error) => void;
}

// ---------------------------------------------------------------------------
// Class
// ---------------------------------------------------------------------------

export class EMGStreamProcessor extends EventEmitter {
  /**
   * Ring buffer stores int16 values as floats (normalised to [-1, 1]).
   * Layout: [sample0_ch0, sample0_ch1, ..., sample0_ch7, sample1_ch0, ...]
   */
  private ring: Float32Array;

  /** Write head — index of the next slot to write (in samples, not floats). */
  private writeHead = 0;

  /** Total samples written since last reset (for determining fill level). */
  private totalWritten = 0;

  /** Leftover bytes from a partial sample at the end of the last packet. */
  private partial: Uint8Array = new Uint8Array(0);

  constructor() {
    super();
    this.ring = new Float32Array(RING_CAPACITY * NUM_CHANNELS);
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Feed raw BLE bytes into the processor.  Internally parses int16 samples,
   * fills the ring buffer, and emits 'window' events when a full window is
   * ready.
   */
  ingestBytes(raw: Uint8Array): void {
    // Prepend any leftover bytes from the previous call
    let data: Uint8Array;
    if (this.partial.byteLength > 0) {
      data = new Uint8Array(this.partial.byteLength + raw.byteLength);
      data.set(this.partial, 0);
      data.set(raw, this.partial.byteLength);
      this.partial = new Uint8Array(0);
    } else {
      data = raw;
    }

    const completeSamples = Math.floor(data.byteLength / BYTES_PER_SAMPLE);
    const leftoverBytes = data.byteLength % BYTES_PER_SAMPLE;

    // Store leftover bytes for next call
    if (leftoverBytes > 0) {
      this.partial = data.slice(data.byteLength - leftoverBytes);
    }

    if (completeSamples === 0) return;

    const view = new DataView(
      data.buffer,
      data.byteOffset,
      completeSamples * BYTES_PER_SAMPLE,
    );

    for (let s = 0; s < completeSamples; s++) {
      const ringBase = (this.writeHead % RING_CAPACITY) * NUM_CHANNELS;

      for (let ch = 0; ch < NUM_CHANNELS; ch++) {
        const byteOffset = s * BYTES_PER_SAMPLE + ch * 2;
        const int16Val = view.getInt16(byteOffset, /* littleEndian= */ true);
        // Normalise to [-1, 1] — int16 range is ±32768
        this.ring[ringBase + ch] = int16Val / 32768.0;
      }

      this.writeHead++;
      this.totalWritten++;

      // Check whether we have accumulated enough new samples to emit a window
      if (
        this.totalWritten >= WINDOW_SAMPLES &&
        (this.totalWritten - WINDOW_SAMPLES) % HOP_SAMPLES === 0
      ) {
        const window = this.buildWindow();
        if (window !== null) {
          this.emit("window", window);
        }
      }
    }
  }

  /**
   * Returns the most recent complete window as a Float32Array of length
   * WINDOW_SAMPLES × NUM_CHANNELS, or null if fewer than WINDOW_SAMPLES
   * have been received since the last reset.
   */
  getWindow(): Float32Array | null {
    if (this.totalWritten < WINDOW_SAMPLES) return null;
    return this.buildWindow();
  }

  /** Clear all buffered data and reset state. */
  reset(): void {
    this.ring.fill(0);
    this.writeHead = 0;
    this.totalWritten = 0;
    this.partial = new Uint8Array(0);
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  /**
   * Copies the most recent WINDOW_SAMPLES samples out of the ring buffer into
   * a freshly allocated Float32Array.  Handles wrapping correctly.
   */
  private buildWindow(): Float32Array | null {
    if (this.totalWritten < WINDOW_SAMPLES) return null;

    const out = new Float32Array(WINDOW_SAMPLES * NUM_CHANNELS);

    // The newest sample is at writeHead - 1 (mod RING_CAPACITY).
    // We want samples [writeHead - WINDOW_SAMPLES .. writeHead - 1].
    const startSample =
      (((this.writeHead - WINDOW_SAMPLES) % RING_CAPACITY) + RING_CAPACITY) %
      RING_CAPACITY;

    for (let i = 0; i < WINDOW_SAMPLES; i++) {
      const srcSample = (startSample + i) % RING_CAPACITY;
      const srcBase = srcSample * NUM_CHANNELS;
      const dstBase = i * NUM_CHANNELS;

      for (let ch = 0; ch < NUM_CHANNELS; ch++) {
        out[dstBase + ch] = this.ring[srcBase + ch];
      }
    }

    return out;
  }

  // -------------------------------------------------------------------------
  // Typed emit/on overloads (TypeScript ergonomics)
  // -------------------------------------------------------------------------

  emit(event: "window", data: Float32Array): boolean;
  emit(event: "error", err: Error): boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  emit(event: string, ...args: any[]): boolean {
    return super.emit(event, ...args);
  }

  on(event: "window", listener: (data: Float32Array) => void): this;
  on(event: "error", listener: (err: Error) => void): this;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  on(event: string, listener: (...args: any[]) => void): this {
    return super.on(event, listener);
  }

  off(event: "window", listener: (data: Float32Array) => void): this;
  off(event: "error", listener: (err: Error) => void): this;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  off(event: string, listener: (...args: any[]) => void): this {
    return super.off(event, listener);
  }
}
