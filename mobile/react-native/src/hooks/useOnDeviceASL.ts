/**
 * useOnDeviceASL.ts
 * React hook that wires the full on-device pipeline together:
 *   BLE -> EMGWindowBuffer -> ONNX inference -> React state
 *
 * Optionally falls back to a WebSocket server when the ONNX model cannot
 * be loaded (e.g. missing asset during development).
 *
 * Usage:
 *   const { connectionStatus, lastLabel, lastConfidence, connect, disconnect } =
 *     useOnDeviceASL({ autoConnect: false, fallbackToServer: true });
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Device } from "react-native-ble-plx";

import {
  bleManager,
  ConnectionStatus,
  BLE_DEVICE_NAME,
} from "../bluetooth/BLEManager";
import { EMGWindowBuffer } from "../inference/EMGWindowBuffer";
import { onDeviceInference } from "../inference/ONNXInference";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface UseOnDeviceASLOptions {
  /** BLE advertised name to scan for. Default: 'Myo' (MYO Armband prefix). */
  deviceName?: string;
  /** Called whenever a prediction clears the confidence threshold. */
  onPrediction?: (label: string, confidence: number) => void;
  /** Called whenever the BLE connection status changes. */
  onConnectionChange?: (status: ConnectionStatus) => void;
  /** Automatically call connect() on mount. Default: false. */
  autoConnect?: boolean;
  /**
   * If true and the ONNX model fails to load, opens a WebSocket to
   * serverUrl and forwards raw EMG windows to the server for inference.
   * Default: false.
   */
  fallbackToServer?: boolean;
  /** WebSocket endpoint used when fallbackToServer is true. Default: 'ws://localhost:8000/stream'. */
  serverUrl?: string;
}

export interface PredictionEntry {
  label: string;
  confidence: number;
  timestamp: number;
}

export interface UseOnDeviceASLResult {
  // Connection
  connectionStatus: ConnectionStatus;
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
  isConnected: boolean;

  // Predictions
  lastLabel: string | null;
  lastConfidence: number;
  predictionHistory: PredictionEntry[];
  clearHistory: () => void;

  // Model
  modelLoaded: boolean;
  modelLoadError: string | null;

  // Stats
  windowsProcessed: number;
  predictionsAccepted: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_HISTORY = 50;
const DEFAULT_DEVICE_NAME = BLE_DEVICE_NAME; // 'Myo'
const DEFAULT_SERVER_URL = "ws://localhost:8000/stream";

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useOnDeviceASL(
  options: UseOnDeviceASLOptions = {},
): UseOnDeviceASLResult {
  const {
    deviceName = DEFAULT_DEVICE_NAME,
    onPrediction,
    onConnectionChange,
    autoConnect = false,
    fallbackToServer = false,
    serverUrl = DEFAULT_SERVER_URL,
  } = options;

  // -------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------

  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("disconnected");
  const [lastLabel, setLastLabel] = useState<string | null>(null);
  const [lastConfidence, setLastConfidence] = useState(0);
  const [predictionHistory, setPredictionHistory] = useState<PredictionEntry[]>(
    [],
  );
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelLoadError, setModelLoadError] = useState<string | null>(null);
  const [windowsProcessed, setWindowsProcessed] = useState(0);
  const [predictionsAccepted, setPredictionsAccepted] = useState(0);

  // -------------------------------------------------------------------------
  // Refs (avoid stale closures; survive re-renders without triggering effects)
  // -------------------------------------------------------------------------

  const mountedRef = useRef(true);
  const deviceRef = useRef<Device | null>(null);
  // Pending windows collected from EMGWindowBuffer onWindow callbacks
  const pendingWindowsRef = useRef<Float32Array[]>([]);
  // EMGWindowBuffer is initialized in useEffect once the ref infrastructure is set up
  const windowBufferRef = useRef<EMGWindowBuffer | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wsRef = useRef<any>(null);
  const modelLoadErrorRef = useRef<string | null>(null);

  // Keep callback refs stable so inner callbacks always call the latest version.
  const onPredictionRef = useRef(onPrediction);
  onPredictionRef.current = onPrediction;

  const onConnectionChangeRef = useRef(onConnectionChange);
  onConnectionChangeRef.current = onConnectionChange;

  // Initialize EMGWindowBuffer with onWindow callback on first render
  if (windowBufferRef.current === null) {
    windowBufferRef.current = new EMGWindowBuffer((window) => {
      pendingWindowsRef.current.push(window);
    });
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  const updateStatus = useCallback((status: ConnectionStatus) => {
    if (!mountedRef.current) return;
    setConnectionStatus(status);
    onConnectionChangeRef.current?.(status);
  }, []);

  /**
   * Attempt a WebSocket fallback connection to the inference server.
   * When a completed EMG window arrives, it is serialised as a binary
   * message (raw Int16Array buffer) and sent over the socket.
   * The server is expected to reply with a JSON message:
   *   { "label": "A", "confidence": 0.92 }
   */
  const openFallbackSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    console.log(`[useOnDeviceASL] Opening fallback WebSocket: ${serverUrl}`);
    const ws = new WebSocket(serverUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      console.log("[useOnDeviceASL] Fallback WebSocket connected.");
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ws.onmessage = (event: any) => {
      if (!mountedRef.current) return;
      try {
        const parsed = JSON.parse(event.data as string) as {
          label?: string;
          confidence?: number;
        };

        if (
          typeof parsed.label === "string" &&
          typeof parsed.confidence === "number"
        ) {
          const { label, confidence } = parsed;
          setLastLabel(label);
          setLastConfidence(confidence);
          setPredictionHistory((prev) => {
            const entry: PredictionEntry = {
              label,
              confidence,
              timestamp: Date.now(),
            };
            return [entry, ...prev].slice(0, MAX_HISTORY);
          });
          setPredictionsAccepted((n) => n + 1);
          onPredictionRef.current?.(label, confidence);
        }
      } catch {
        // Non-JSON or malformed message -- ignore.
      }
    };

    ws.onerror = (err) => {
      console.warn("[useOnDeviceASL] Fallback WebSocket error:", err);
    };

    ws.onclose = () => {
      console.log("[useOnDeviceASL] Fallback WebSocket closed.");
    };

    wsRef.current = ws;
  }, [serverUrl]);

  const closeFallbackSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  /**
   * Core BLE data handler. Called once per BLE notification (one EMG sample
   * packet = 16 bytes). Feeds bytes into the sliding window buffer; for each
   * complete window either runs on-device ONNX inference or forwards to the
   * fallback WebSocket server.
   */
  const handleBytes = useCallback(
    async (bytes: Uint8Array) => {
      // Feed bytes into the window buffer; onWindow callback populates pendingWindowsRef
      windowBufferRef.current?.feedMyo(bytes);

      const windows = pendingWindowsRef.current.splice(0);

      if (windows.length === 0) return;

      setWindowsProcessed((n) => n + windows.length);

      for (const window of windows) {
        // On-device path
        if (modelLoadErrorRef.current === null) {
          let result: { label: string; confidence: number } | null = null;
          try {
            // EMGWindowBuffer yields Float32Array; predict() expects Int16Array.
            // Scale back to int16 range for the inference contract.
            const int16Window = Int16Array.from(window, (v) =>
              Math.round(v * 2048),
            );
            result = await onDeviceInference.predict(int16Window);
          } catch (err) {
            console.warn("[useOnDeviceASL] ONNX predict error:", err);
          }

          if (result !== null && mountedRef.current) {
            const { label, confidence } = result;
            setLastLabel(label);
            setLastConfidence(confidence);
            setPredictionHistory((prev) => {
              const entry: PredictionEntry = {
                label,
                confidence,
                timestamp: Date.now(),
              };
              return [entry, ...prev].slice(0, MAX_HISTORY);
            });
            setPredictionsAccepted((n) => n + 1);
            onPredictionRef.current?.(label, confidence);
          }
          continue;
        }

        // Fallback WebSocket path
        if (
          fallbackToServer &&
          wsRef.current !== null &&
          wsRef.current.readyState === WebSocket.OPEN
        ) {
          wsRef.current.send(window.buffer);
        }
      }
    },
    [fallbackToServer],
  );

  // -------------------------------------------------------------------------
  // Load ONNX model on mount
  // -------------------------------------------------------------------------

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        await onDeviceInference.loadModel();
        if (cancelled) return;
        modelLoadErrorRef.current = null;
        setModelLoaded(true);
        setModelLoadError(null);
        console.log("[useOnDeviceASL] ONNX model ready.");
      } catch (err) {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        modelLoadErrorRef.current = msg;
        setModelLoadError(msg);
        console.warn("[useOnDeviceASL] Model load failed:", msg);

        if (fallbackToServer) {
          openFallbackSocket();
        }
      }
    })();

    return () => {
      cancelled = true;
    };
    // openFallbackSocket and fallbackToServer are stable across renders
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // -------------------------------------------------------------------------
  // BLE status listener
  // -------------------------------------------------------------------------

  useEffect(() => {
    const unsub = bleManager.addStatusListener((status) => {
      updateStatus(status);
    });
    return unsub;
  }, [updateStatus]);

  // -------------------------------------------------------------------------
  // Auto-connect on mount
  // -------------------------------------------------------------------------

  // connect and disconnect are defined below; use a ref to call the latest
  // version inside the mount-only effect.
  const connectRef = useRef<() => Promise<void>>(async () => {
    /* placeholder */
  });

  useEffect(() => {
    if (autoConnect) {
      connectRef.current();
    }
    // Run only on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // -------------------------------------------------------------------------
  // Cleanup on unmount
  // -------------------------------------------------------------------------

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      bleManager.disconnect().catch(() => {
        /* best-effort */
      });
      windowBufferRef.current?.reset();
      closeFallbackSocket();
      onDeviceInference.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // -------------------------------------------------------------------------
  // Public: connect
  // -------------------------------------------------------------------------

  const connect = useCallback(async () => {
    const status = bleManager.getStatus();
    if (
      status === "connected" ||
      status === "connecting" ||
      status === "scanning"
    ) {
      return;
    }

    try {
      const device = await bleManager.scanAndConnect(deviceName);
      deviceRef.current = device;
      bleManager.subscribeToEMG(device, handleBytes);
    } catch (err) {
      console.error("[useOnDeviceASL] connect() error:", err);
      // BLEManager already set status to 'error'.
    }
  }, [deviceName, handleBytes]);

  // Expose the latest connect to the auto-connect ref.
  connectRef.current = connect;

  // -------------------------------------------------------------------------
  // Public: disconnect
  // -------------------------------------------------------------------------

  const disconnect = useCallback(async () => {
    await bleManager.disconnect();
    windowBufferRef.current?.reset();
    deviceRef.current = null;
  }, []);

  // -------------------------------------------------------------------------
  // Public: clearHistory
  // -------------------------------------------------------------------------

  const clearHistory = useCallback(() => {
    setPredictionHistory([]);
  }, []);

  // -------------------------------------------------------------------------
  // Derived state
  // -------------------------------------------------------------------------

  const isConnected = connectionStatus === "connected";

  // -------------------------------------------------------------------------
  // Return
  // -------------------------------------------------------------------------

  return {
    connectionStatus,
    connect,
    disconnect,
    isConnected,

    lastLabel,
    lastConfidence,
    predictionHistory,
    clearHistory,

    modelLoaded,
    modelLoadError,

    windowsProcessed,
    predictionsAccepted,
  };
}
