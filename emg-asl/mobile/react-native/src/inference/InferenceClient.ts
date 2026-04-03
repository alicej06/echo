/**
 * InferenceClient.ts
 * Sends windowed EMG data to the Python inference server over WebSocket
 * (socket.io) and delivers ASL predictions back to the caller.
 *
 * Protocol:
 *   emit  → 'emg_window'  : { data: number[], shape: [number, number] }
 *   on    ← 'prediction'  : { label: string, confidence: number }
 *   on    ← 'error'       : { message: string }
 */

import { io, Socket } from 'socket.io-client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PredictionResult {
  label: string;
  confidence: number;
}

export type PredictionCallback = (label: string, confidence: number) => void;

interface ServerToClientEvents {
  prediction: (result: PredictionResult) => void;
  error: (payload: { message: string }) => void;
  connect_error: (err: Error) => void;
}

interface ClientToServerEvents {
  emg_window: (payload: {
    data: number[];
    shape: [number, number];
    timestamp: number;
  }) => void;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Minimum milliseconds between accepted predictions for the same label. */
const DEBOUNCE_MS = 300;

/** How many times socket.io should attempt reconnection before giving up. */
const MAX_RECONNECT_ATTEMPTS = 10;

// ---------------------------------------------------------------------------
// Class
// ---------------------------------------------------------------------------

export class ASLInferenceClient {
  private socket: Socket<ServerToClientEvents, ClientToServerEvents> | null =
    null;
  private predictionCallbacks: Set<PredictionCallback> = new Set();
  private connected = false;

  // Debounce state
  private lastLabel: string | null = null;
  private lastPredictionTime = 0;

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Establish a WebSocket connection to the inference server.
   * Resolves when the connection is confirmed (the 'connect' event fires).
   */
  connect(serverUrl: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.socket?.connected) {
        resolve();
        return;
      }

      // Tear down any stale socket before creating a new one
      this.destroySocket();

      const socket: Socket<ServerToClientEvents, ClientToServerEvents> = io(
        serverUrl,
        {
          transports: ['websocket'],
          reconnection: true,
          reconnectionAttempts: MAX_RECONNECT_ATTEMPTS,
          reconnectionDelay: 1_000,
          reconnectionDelayMax: 10_000,
          timeout: 10_000,
        },
      );

      this.socket = socket;

      const onConnect = () => {
        this.connected = true;
        console.log('[InferenceClient] Connected to', serverUrl);
        socket.off('connect_error', onConnectError);
        resolve();
      };

      const onConnectError = (err: Error) => {
        console.error('[InferenceClient] Connection error:', err.message);
        // Only reject on the first attempt; after that socket.io retries
        socket.off('connect', onConnect);
        reject(err);
      };

      socket.once('connect', onConnect);
      socket.once('connect_error', onConnectError);

      // Persistent event handlers
      socket.on('disconnect', (reason) => {
        this.connected = false;
        console.warn('[InferenceClient] Disconnected:', reason);
      });

      socket.on('prediction', (result: PredictionResult) => {
        this.handlePrediction(result.label, result.confidence);
      });

      socket.on('error', (payload: { message: string }) => {
        console.error('[InferenceClient] Server error:', payload.message);
      });
    });
  }

  /**
   * Send an EMG window to the server for classification.
   * The window should be a Float32Array of shape [WINDOW_SAMPLES × NUM_CHANNELS].
   */
  sendWindow(window: Float32Array): void {
    if (!this.socket?.connected) {
      console.warn('[InferenceClient] sendWindow called while disconnected.');
      return;
    }

    this.socket.emit('emg_window', {
      data: Array.from(window),
      // Inferred shape; server should validate
      shape: [40, 8],
      timestamp: Date.now(),
    });
  }

  /**
   * Register a callback that will be called whenever a new (non-debounced)
   * prediction arrives from the server.
   * Returns an unsubscribe function.
   */
  onPrediction(cb: PredictionCallback): () => void {
    this.predictionCallbacks.add(cb);
    return () => this.predictionCallbacks.delete(cb);
  }

  /** Gracefully close the WebSocket connection. */
  disconnect(): void {
    this.destroySocket();
    this.connected = false;
  }

  /** Returns true when the WebSocket is open and ready. */
  isConnected(): boolean {
    return this.connected && (this.socket?.connected ?? false);
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  private handlePrediction(label: string, confidence: number): void {
    const now = Date.now();
    const sameLabel = label === this.lastLabel;
    const withinDebounce = now - this.lastPredictionTime < DEBOUNCE_MS;

    if (sameLabel && withinDebounce) return; // debounced

    this.lastLabel = label;
    this.lastPredictionTime = now;

    this.predictionCallbacks.forEach((cb) => {
      try {
        cb(label, confidence);
      } catch (err) {
        console.error('[InferenceClient] Prediction callback threw:', err);
      }
    });
  }

  private destroySocket(): void {
    if (!this.socket) return;
    this.socket.removeAllListeners();
    this.socket.disconnect();
    this.socket = null;
  }
}

// Singleton instance
export const inferenceClient = new ASLInferenceClient();
