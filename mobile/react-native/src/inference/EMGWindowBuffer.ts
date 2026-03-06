/**
 * EMGWindowBuffer.ts
 * Accumulates raw BLE bytes from the EMG armband and emits complete,
 * overlapping analysis windows ready for ONNX inference.
 *
 * BLE packet format (16 bytes per notification):
 *   8 channels x 2 bytes each = 16 bytes
 *   Each pair is a big-endian (MSB-first) signed int16.
 *   The firmware subtracts 2048 before transmission, so values are centred at 0.
 *
 * Window layout produced (flat row-major Int16Array of length windowSize * nChannels):
 *   [ch0_t0, ch1_t0, ..., ch7_t0,   <- sample 0
 *    ch0_t1, ch1_t1, ..., ch7_t1,   <- sample 1
 *    ...
 *    ch0_t39, ch1_t39, ..., ch7_t39] <- sample 39
 *
 * Sliding-window strategy:
 *   - stride = windowSize - overlapSamples  (default: 40 - 20 = 20 samples)
 *   - A new window is emitted every `stride` incoming samples.
 *   - Each window reuses `overlapSamples` samples from the previous window,
 *     giving 50% overlap by default.
 */

// ---------------------------------------------------------------------------
// EMGWindowBuffer
// ---------------------------------------------------------------------------

export class EMGWindowBuffer {
  private readonly windowSize: number;
  private readonly nChannels: number;
  private readonly overlapSamples: number;
  private readonly stride: number;

  /**
   * Ring buffer holding at most `windowSize` samples.
   * Layout: [ch0_tN, ch1_tN, ..., ch(C-1)_tN, ch0_t(N+1), ...]
   * sampleCount tracks how many valid samples are currently stored.
   */
  private readonly buffer: Int16Array;

  /** Number of valid samples currently in the buffer (0..windowSize). */
  private sampleCount = 0;

  /**
   * Counts new samples since the last window was emitted.
   * When this reaches `stride` a new window is produced and it resets.
   */
  private samplesSinceLastWindow = 0;

  // -------------------------------------------------------------------------
  // Constructor
  // -------------------------------------------------------------------------

  /**
   * @param windowSize     - Number of samples per analysis window (default 40).
   * @param nChannels      - Number of EMG channels per sample (default 8).
   * @param overlapSamples - Samples shared between consecutive windows (default 20).
   *   Must be strictly less than windowSize.
   */
  constructor(
    windowSize: number = 40,
    nChannels: number = 8,
    overlapSamples: number = 20,
  ) {
    if (overlapSamples >= windowSize) {
      throw new RangeError(
        `overlapSamples (${overlapSamples}) must be less than windowSize (${windowSize}).`,
      );
    }

    this.windowSize = windowSize;
    this.nChannels = nChannels;
    this.overlapSamples = overlapSamples;
    this.stride = windowSize - overlapSamples; // 20

    // Pre-allocate a flat buffer: windowSize rows x nChannels columns.
    this.buffer = new Int16Array(windowSize * nChannels);
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Ingest one raw BLE notification packet and return any newly completed
   * analysis windows.
   *
   * @param bytes - A Uint8Array of exactly 16 bytes (8 channels x 2 bytes,
   *   big-endian int16 per channel).  If the packet length is not 16 the
   *   method logs a warning and returns an empty array.
   *
   * @returns An array (usually empty, occasionally one element) of complete
   *   Int16Array windows, each of length windowSize * nChannels (320).
   *   The array is in row-major order: all channels for sample 0 first,
   *   then all channels for sample 1, etc.
   */
  ingestBytes(bytes: Uint8Array): Int16Array[] {
    const expectedBytes = this.nChannels * 2; // 16

    if (bytes.length !== expectedBytes) {
      console.warn(
        `[EMGWindowBuffer] Expected ${expectedBytes} bytes, got ${bytes.length}. Packet skipped.`,
      );
      return [];
    }

    // --- Parse one sample (nChannels int16 big-endian values) ---
    const sample = this.parseSample(bytes);

    // --- Append the sample to the rolling buffer ---
    this.appendSample(sample);

    // --- Emit a window when a full stride of new samples has accumulated ---
    const completedWindows: Int16Array[] = [];

    if (this.sampleCount === this.windowSize) {
      this.samplesSinceLastWindow += 1;

      if (this.samplesSinceLastWindow >= this.stride) {
        completedWindows.push(this.copyCurrentWindow());
        this.samplesSinceLastWindow = 0;
      }
    } else {
      // Still filling up the initial windowSize samples -- count incoming
      // samples but do not emit yet.
      this.samplesSinceLastWindow = 0;
    }

    return completedWindows;
  }

  /**
   * Clear all buffered samples and reset stride counter.
   * Call this on BLE disconnect or when starting a new recording session.
   */
  reset(): void {
    this.buffer.fill(0);
    this.sampleCount = 0;
    this.samplesSinceLastWindow = 0;
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  /**
   * Parse nChannels big-endian int16 values from a raw byte packet.
   * MSB comes first in each pair (byte[0] is high byte, byte[1] is low byte).
   */
  private parseSample(bytes: Uint8Array): Int16Array {
    const sample = new Int16Array(this.nChannels);

    for (let ch = 0; ch < this.nChannels; ch++) {
      const byteIdx = ch * 2;
      // Reconstruct signed int16: combine high and low bytes, then sign-extend.
      const unsigned = (bytes[byteIdx] << 8) | bytes[byteIdx + 1];
      // Reinterpret as signed 16-bit by checking the sign bit.
      sample[ch] = unsigned >= 0x8000 ? unsigned - 0x10000 : unsigned;
    }

    return sample;
  }

  /**
   * Append a single sample to the circular rolling buffer.
   *
   * The buffer is treated as a sliding window:
   *   - While sampleCount < windowSize: write new samples at position sampleCount.
   *   - Once full: shift the contents left by one sample position and write
   *     the new sample at the last slot.  This is O(windowSize * nChannels)
   *     per sample but windowSize is only 40, keeping it cheap on mobile.
   */
  private appendSample(sample: Int16Array): void {
    if (this.sampleCount < this.windowSize) {
      // Fill phase: write directly at the end of valid data.
      const offset = this.sampleCount * this.nChannels;
      for (let ch = 0; ch < this.nChannels; ch++) {
        this.buffer[offset + ch] = sample[ch];
      }
      this.sampleCount += 1;
    } else {
      // Slide phase: drop the oldest sample by shifting everything left by
      // one sample (nChannels values), then write new sample at the end.
      this.buffer.copyWithin(0, this.nChannels); // shift left by nChannels elements
      const lastOffset = (this.windowSize - 1) * this.nChannels;
      for (let ch = 0; ch < this.nChannels; ch++) {
        this.buffer[lastOffset + ch] = sample[ch];
      }
      // sampleCount stays at windowSize
    }
  }

  /**
   * Return a snapshot of the current window as a new Int16Array.
   * The returned array is a copy so that callers can hold onto it safely
   * while the buffer continues to receive data.
   */
  private copyCurrentWindow(): Int16Array {
    return this.buffer.slice(0, this.windowSize * this.nChannels);
  }
}
