/**
 * SpeechEngine.ts
 * Queue-aware text-to-speech engine backed by expo-speech.
 *
 * Features:
 *   - Single queue — if the engine is already speaking the next item waits.
 *   - Configurable speech rate (0.5–2.0) and language/locale.
 *   - Safe stop that also drains the queue.
 */

import * as Speech from 'expo-speech';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_RATE = 1.0;
const DEFAULT_LANGUAGE = 'en-US';
const MIN_RATE = 0.5;
const MAX_RATE = 2.0;

// ---------------------------------------------------------------------------
// Class
// ---------------------------------------------------------------------------

export class ASLSpeechEngine {
  private queue: string[] = [];
  private speaking = false;
  private rate: number = DEFAULT_RATE;
  private language: string = DEFAULT_LANGUAGE;

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Speak `text` immediately if idle, or enqueue it if currently speaking.
   * Repeated consecutive identical words are ignored to avoid spam.
   */
  speak(text: string): void {
    const trimmed = text.trim();
    if (!trimmed) return;

    // Skip if the same word is already at the tail of the queue or playing
    const lastQueued = this.queue[this.queue.length - 1];
    if (this.speaking && lastQueued === trimmed) return;
    if (!this.speaking && trimmed === this._currentlySpoken) return;

    this.queue.push(trimmed);

    if (!this.speaking) {
      this.advance();
    }
  }

  /** Update the speech rate.  Clamped to [MIN_RATE, MAX_RATE]. */
  setRate(rate: number): void {
    this.rate = Math.min(MAX_RATE, Math.max(MIN_RATE, rate));
  }

  /** Set the BCP-47 language/locale tag (e.g. 'en-US', 'en-GB'). */
  setLanguage(lang: string): void {
    this.language = lang;
  }

  /** Immediately stop speech and clear the queue. */
  stop(): void {
    this.queue = [];
    this.speaking = false;
    this._currentlySpoken = null;

    Speech.stop();
  }

  /** Returns true when the engine is currently producing audio. */
  isSpeaking(): boolean {
    return this.speaking;
  }

  /** Returns the full pending queue (does not include the currently speaking item). */
  getQueue(): readonly string[] {
    return this.queue;
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  private _currentlySpoken: string | null = null;

  private advance(): void {
    const next = this.queue.shift();

    if (next === undefined) {
      this.speaking = false;
      this._currentlySpoken = null;
      return;
    }

    this.speaking = true;
    this._currentlySpoken = next;

    Speech.speak(next, {
      language: this.language,
      rate: this.rate,
      onDone: () => {
        this.speaking = false;
        this._currentlySpoken = null;
        this.advance();
      },
      onError: (err) => {
        console.error('[SpeechEngine] TTS error:', err);
        this.speaking = false;
        this._currentlySpoken = null;
        this.advance(); // continue draining queue even after error
      },
      onStopped: () => {
        // Called when stop() is invoked externally; queue already cleared.
        this.speaking = false;
        this._currentlySpoken = null;
      },
    });
  }
}

// Singleton instance
export const speechEngine = new ASLSpeechEngine();
