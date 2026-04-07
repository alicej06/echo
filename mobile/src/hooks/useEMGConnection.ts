/**
 * useEMGConnection.ts
 * React hook that wires together BLEManager, EMGStreamProcessor, and
 * InferenceClient into a single, lifecycle-aware connection object.
 *
 * Usage:
 *   const { connectionStatus, lastLabel, lastConfidence, connect, disconnect } =
 *     useEMGConnection();
 */

import { useCallback, useEffect, useRef, useState } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { Device } from "react-native-ble-plx";

import { bleManager, ConnectionStatus } from "../bluetooth/BLEManager";
import { EMGStreamProcessor } from "../bluetooth/EMGStream";
import { inferenceClient } from "../inference/InferenceClient";
import { speechEngine } from "../tts/SpeechEngine";

// ---------------------------------------------------------------------------
// Storage keys
// ---------------------------------------------------------------------------

const KEY_SERVER_URL = "settings:serverUrl";
const KEY_DEVICE_NAME = "settings:deviceName";
const DEFAULT_SERVER_URL = "ws://localhost:8765";
const DEFAULT_DEVICE_NAME = "EMG-Band";

// ---------------------------------------------------------------------------
// Hook return type
// ---------------------------------------------------------------------------

export interface EMGConnectionState {
  connectionStatus: ConnectionStatus;
  lastLabel: string | null;
  lastConfidence: number;
  history: string[];
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
  startCalibration: () => void;
  stopCalibration: () => void;
  isCalibrating: boolean;
  serverUrl: string;
  deviceName: string;
  setServerUrl: (url: string) => Promise<void>;
  setDeviceName: (name: string) => Promise<void>;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useEMGConnection(): EMGConnectionState {
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("disconnected");
  const [lastLabel, setLastLabel] = useState<string | null>(null);
  const [lastConfidence, setLastConfidence] = useState(0);
  const [history, setHistory] = useState<string[]>([]);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [serverUrl, setServerUrlState] = useState(DEFAULT_SERVER_URL);
  const [deviceName, setDeviceNameState] = useState(DEFAULT_DEVICE_NAME);

  // Refs to avoid stale closures inside callbacks
  const processorRef = useRef<EMGStreamProcessor | null>(null);
  const deviceRef = useRef<Device | null>(null);
  const emgSubRef = useRef<boolean>(false);
  const predUnsubRef = useRef<(() => void) | null>(null);
  const statusUnsubRef = useRef<(() => void) | null>(null);
  const isCalibratingRef = useRef(false);
  const mountedRef = useRef(true);

  // -------------------------------------------------------------------------
  // Load persisted settings on mount
  // -------------------------------------------------------------------------
  useEffect(() => {
    (async () => {
      const [storedUrl, storedName] = await Promise.all([
        AsyncStorage.getItem(KEY_SERVER_URL),
        AsyncStorage.getItem(KEY_DEVICE_NAME),
      ]);
      if (!mountedRef.current) return;
      if (storedUrl) setServerUrlState(storedUrl);
      if (storedName) setDeviceNameState(storedName);
    })();
  }, []);

  // -------------------------------------------------------------------------
  // Attach BLE status listener
  // -------------------------------------------------------------------------
  useEffect(() => {
    const unsub = bleManager.addStatusListener((status) => {
      if (mountedRef.current) setConnectionStatus(status);
    });
    statusUnsubRef.current = unsub;
    return unsub;
  }, []);

  // -------------------------------------------------------------------------
  // Cleanup on unmount
  // -------------------------------------------------------------------------
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      _disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  const _disconnect = useCallback(async () => {
    predUnsubRef.current?.();
    predUnsubRef.current = null;

    emgSubRef.current = false;

    processorRef.current?.reset();
    processorRef.current = null;

    deviceRef.current = null;

    inferenceClient.disconnect();
    await bleManager.disconnect();

    speechEngine.stop();
  }, []);

  // -------------------------------------------------------------------------
  // Public: connect
  // -------------------------------------------------------------------------

  const connect = useCallback(async () => {
    if (connectionStatus === "connected" || connectionStatus === "connecting") {
      return;
    }

    try {
      // 1. Resolve current settings
      const [url, name] = await Promise.all([
        AsyncStorage.getItem(KEY_SERVER_URL),
        AsyncStorage.getItem(KEY_DEVICE_NAME),
      ]);
      const resolvedUrl = url ?? DEFAULT_SERVER_URL;
      const resolvedName = name ?? DEFAULT_DEVICE_NAME;

      // 2. Connect inference server
      await inferenceClient.connect(resolvedUrl);

      // 3. Register prediction callback
      predUnsubRef.current?.();
      predUnsubRef.current = inferenceClient.onPrediction(
        (label, confidence) => {
          if (!mountedRef.current || isCalibratingRef.current) return;

          setLastLabel(label);
          setLastConfidence(confidence);
          setHistory((prev) => [label, ...prev].slice(0, 5));

          speechEngine.speak(label);
        },
      );

      // 4. Set up EMG stream processor
      const processor = new EMGStreamProcessor();
      processorRef.current = processor;

      processor.on("window", (window) => {
        if (inferenceClient.isConnected()) {
          inferenceClient.sendWindow(window);
        }
      });

      // 5. Scan and connect BLE
      const device = await bleManager.scanAndConnect(resolvedName);
      deviceRef.current = device;

      // 6. Subscribe to EMG notifications
      await bleManager.subscribeToEMG(device, (bytes) => {
        processorRef.current?.ingestBytes(bytes);
      });
      emgSubRef.current = true;
    } catch (err) {
      console.error("[useEMGConnection] connect error:", err);
      // Status will have been set to 'error' by BLEManager
    }
  }, [connectionStatus]);

  // -------------------------------------------------------------------------
  // Public: disconnect
  // -------------------------------------------------------------------------

  const disconnect = useCallback(async () => {
    await _disconnect();
  }, [_disconnect]);

  // -------------------------------------------------------------------------
  // Public: calibration helpers
  // -------------------------------------------------------------------------

  const startCalibration = useCallback(() => {
    isCalibratingRef.current = true;
    setIsCalibrating(true);
  }, []);

  const stopCalibration = useCallback(() => {
    isCalibratingRef.current = false;
    setIsCalibrating(false);
  }, []);

  // -------------------------------------------------------------------------
  // Public: settings setters
  // -------------------------------------------------------------------------

  const setServerUrl = useCallback(async (url: string) => {
    await AsyncStorage.setItem(KEY_SERVER_URL, url);
    setServerUrlState(url);
  }, []);

  const setDeviceName = useCallback(async (name: string) => {
    await AsyncStorage.setItem(KEY_DEVICE_NAME, name);
    setDeviceNameState(name);
  }, []);

  // -------------------------------------------------------------------------
  // Return
  // -------------------------------------------------------------------------

  return {
    connectionStatus,
    lastLabel,
    lastConfidence,
    history,
    connect,
    disconnect,
    startCalibration,
    stopCalibration,
    isCalibrating,
    serverUrl,
    deviceName,
    setServerUrl,
    setDeviceName,
  };
}
