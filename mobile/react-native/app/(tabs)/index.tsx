import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  Animated,
  Pressable,
  ScrollView,
  ActivityIndicator,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { getServerConfig } from "../../src/config/serverConfig";
import { EMGWindowBuffer } from "../../src/inference/EMGWindowBuffer";
import { MyoBLEManager, MyoStatus } from "../../src/ble/MyoBLEManager";

const WS_RECONNECT_MS = 3000;
const N_CHANNELS = 8;
const WINDOW_SAMPLES = 40;

interface Prediction {
  class: string | null;
  confidence: number;
  latency_ms: number;
}

function makeMockFrame(): Uint8Array {
  // Simulate a Myo notification: 2 samples × 8 channels × int8
  const bytes = new Uint8Array(16);
  for (let i = 0; i < 16; i++) bytes[i] = Math.floor(Math.random() * 256);
  return bytes;
}

function windowToWSFrame(window: Float32Array): ArrayBuffer {
  // Convert float window (40×8) → int16 big-endian for Railway WS API
  const buf = new ArrayBuffer(WINDOW_SAMPLES * N_CHANNELS * 2);
  const view = new DataView(buf);
  for (let i = 0; i < window.length; i++) {
    const val = Math.max(
      -32768,
      Math.min(32767, Math.round(window[i] * 32768)),
    );
    view.setInt16(i * 2, val, false); // big-endian
  }
  return buf;
}

const MYO_STATUS_LABEL: Record<MyoStatus, string> = {
  idle: "Myo disconnected",
  scanning: "Scanning for Myo...",
  connecting: "Connecting to Myo...",
  connected: "Myo connected",
  streaming: "Myo streaming",
  disconnected: "Myo disconnected",
  error: "Myo error",
};

export default function ASLLiveScreen() {
  const [wsConnected, setWsConnected] = useState(false);
  const [myoStatus, setMyoStatus] = useState<MyoStatus>("idle");
  const [useMyo, setUseMyo] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wsRef = useRef<any>(null);
  const streamInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const myoRef = useRef<MyoBLEManager | null>(null);
  const bufferRef = useRef<EMGWindowBuffer | null>(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;

  const triggerPulse = useCallback(() => {
    Animated.sequence([
      Animated.timing(pulseAnim, {
        toValue: 1.15,
        duration: 100,
        useNativeDriver: true,
      }),
      Animated.timing(pulseAnim, {
        toValue: 1,
        duration: 200,
        useNativeDriver: true,
      }),
    ]).start();
  }, [pulseAnim]);

  // WebSocket connection to Railway inference server
  const connect = useCallback(() => {
    const cfg = getServerConfig();
    setError(null);
    try {
      const ws = new WebSocket(cfg.wsUrl);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        setWsConnected(true);
        setError(null);
      };
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data as string) as Prediction;
          setPrediction(data);
          setLatency(data.latency_ms);
          if (data.class) {
            triggerPulse();
            setHistory((prev) => [data.class!, ...prev].slice(0, 20));
          }
        } catch {
          /* ignore parse errors */
        }
      };
      ws.onerror = () => setError("WebSocket error");
      ws.onclose = () => {
        setWsConnected(false);
        setStreaming(false);
        setTimeout(connect, WS_RECONNECT_MS);
      };
      wsRef.current = ws;
    } catch {
      setError(`Cannot connect: ${cfg.wsUrl}`);
    }
  }, [triggerPulse]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      if (streamInterval.current) clearInterval(streamInterval.current);
    };
  }, [connect]);

  // EMG window buffer: emits 40×8 windows → send to Railway WS
  useEffect(() => {
    bufferRef.current = new EMGWindowBuffer((window) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(windowToWSFrame(window));
      }
    });
  }, []);

  const connectMyo = useCallback(async () => {
    if (myoRef.current) return;
    const myo = new MyoBLEManager(
      (data) => bufferRef.current?.feedMyo(data),
      (status) => {
        setMyoStatus(status);
        if (status === "streaming") {
          setUseMyo(true);
          setStreaming(true);
        }
        if (status === "disconnected" || status === "error") {
          setUseMyo(false);
          setStreaming(false);
        }
      },
    );
    myoRef.current = myo;
    try {
      await myo.connect();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      myoRef.current = null;
    }
  }, []);

  const disconnectMyo = useCallback(async () => {
    await myoRef.current?.disconnect();
    myoRef.current = null;
    setUseMyo(false);
    setStreaming(false);
  }, []);

  // Mock streaming (simulator fallback)
  const toggleMockStream = useCallback(() => {
    if (streaming) {
      if (streamInterval.current) clearInterval(streamInterval.current);
      streamInterval.current = null;
      setStreaming(false);
    } else {
      streamInterval.current = setInterval(() => {
        // Feed mock Myo notifications into the buffer at ~200Hz
        bufferRef.current?.feedMyo(makeMockFrame());
        bufferRef.current?.feedMyo(makeMockFrame());
      }, 10);
      setStreaming(true);
    }
  }, [streaming]);

  useEffect(
    () => () => {
      myoRef.current?.destroy();
    },
    [],
  );

  const clearHistory = () => setHistory([]);
  const conf = prediction?.confidence ?? 0;
  const confColor =
    conf > 0.85 ? "#4CAF50" : conf > 0.6 ? "#FFC107" : "#f44336";
  const myoConnected = myoStatus === "connected" || myoStatus === "streaming";

  return (
    <View style={styles.container}>
      {/* Status bar */}
      <View style={styles.statusBar}>
        <View
          style={[
            styles.dot,
            { backgroundColor: wsConnected ? "#4CAF50" : "#f44336" },
          ]}
        />
        <Text style={styles.statusText}>
          {wsConnected ? "Server" : "Server offline"}
        </Text>
        <View
          style={[
            styles.dot,
            {
              backgroundColor: myoConnected ? "#2196F3" : "#555",
              marginLeft: 12,
            },
          ]}
        />
        <Text style={styles.statusText}>{MYO_STATUS_LABEL[myoStatus]}</Text>
        {latency != null && (
          <Text style={styles.latencyText}>{latency.toFixed(1)}ms</Text>
        )}
      </View>

      {error && <Text style={styles.errorText}>{error}</Text>}

      {/* Big prediction */}
      <View style={styles.predictionContainer}>
        <Animated.Text
          style={[styles.predLetter, { transform: [{ scale: pulseAnim }] }]}
        >
          {prediction?.class ?? "?"}
        </Animated.Text>
        <View style={styles.confRow}>
          <View
            style={[
              styles.confBar,
              { width: `${conf * 100}%`, backgroundColor: confColor },
            ]}
          />
        </View>
        <Text style={[styles.confText, { color: confColor }]}>
          {(conf * 100).toFixed(1)}% confidence
        </Text>
      </View>

      {/* Myo connect button */}
      <Pressable
        style={[styles.myoBtn, myoConnected && styles.myoBtnActive]}
        onPress={myoConnected ? disconnectMyo : connectMyo}
      >
        {myoStatus === "scanning" || myoStatus === "connecting" ? (
          <ActivityIndicator size="small" color="#fff" />
        ) : (
          <Ionicons
            name={myoConnected ? "bluetooth" : "bluetooth-outline"}
            size={20}
            color="#fff"
          />
        )}
        <Text style={styles.myoBtnText}>
          {myoConnected ? "Disconnect Myo" : "Connect Myo Band"}
        </Text>
      </Pressable>

      {/* Mock stream button (shown when Myo not connected) */}
      {!myoConnected && (
        <Pressable
          style={[styles.streamBtn, streaming && styles.streamBtnActive]}
          onPress={toggleMockStream}
          disabled={!wsConnected}
        >
          <Ionicons
            name={streaming ? "stop-circle" : "radio-button-on"}
            size={22}
            color="#fff"
          />
          <Text style={styles.streamBtnText}>
            {streaming ? "Stop Mock Stream" : "Start Mock Stream"}
          </Text>
        </Pressable>
      )}

      {/* History */}
      <View style={styles.historyHeader}>
        <Text style={styles.historyTitle}>Recognition history</Text>
        <Pressable onPress={clearHistory}>
          <Text style={styles.clearBtn}>Clear</Text>
        </Pressable>
      </View>
      <ScrollView
        style={styles.historyScroll}
        horizontal
        showsHorizontalScrollIndicator={false}
      >
        {history.map((letter, i) => (
          <View key={i} style={[styles.historyChip, { opacity: 1 - i * 0.04 }]}>
            <Text style={styles.historyLetter}>{letter}</Text>
          </View>
        ))}
        {history.length === 0 && (
          <Text style={styles.historyEmpty}>
            {myoConnected
              ? "Form an ASL letter..."
              : wsConnected
                ? "Connect Myo or start mock stream"
                : "Waiting for server..."}
          </Text>
        )}
      </ScrollView>

      <Text style={styles.footnote}>
        {useMyo ? "Live Myo EMG" : "Mock EMG"} → Railway server → prediction
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#121212", padding: 16 },
  statusBar: { flexDirection: "row", alignItems: "center", marginBottom: 8 },
  dot: { width: 8, height: 8, borderRadius: 4, marginRight: 6 },
  statusText: { color: "#aaa", fontSize: 12, flex: 1 },
  latencyText: { color: "#666", fontSize: 12 },
  errorText: { color: "#f44336", fontSize: 13, marginBottom: 8 },
  predictionContainer: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    maxHeight: 280,
  },
  predLetter: {
    fontSize: 160,
    fontWeight: "800",
    color: "#fff",
    textShadowColor: "#4CAF5066",
    textShadowRadius: 40,
    lineHeight: 180,
  },
  confRow: {
    width: "80%",
    height: 6,
    backgroundColor: "#333",
    borderRadius: 3,
    overflow: "hidden",
    marginTop: 8,
  },
  confBar: { height: "100%", borderRadius: 3 },
  confText: { fontSize: 14, marginTop: 6, fontWeight: "600" },
  myoBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#1a2b4a",
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    gap: 10,
  },
  myoBtnActive: { backgroundColor: "#0d3b6e" },
  myoBtnText: { color: "#fff", fontWeight: "600", fontSize: 15 },
  streamBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#1e3a2f",
    borderRadius: 12,
    padding: 14,
    marginBottom: 16,
    gap: 10,
  },
  streamBtnActive: { backgroundColor: "#3a1e1e" },
  streamBtnText: { color: "#fff", fontWeight: "600", fontSize: 14 },
  historyHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 8,
  },
  historyTitle: { color: "#aaa", fontSize: 13, fontWeight: "600" },
  clearBtn: { color: "#4CAF50", fontSize: 13 },
  historyScroll: { maxHeight: 56, marginBottom: 8 },
  historyChip: {
    width: 40,
    height: 40,
    borderRadius: 8,
    backgroundColor: "#2a2a2a",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 8,
  },
  historyLetter: { color: "#fff", fontSize: 20, fontWeight: "700" },
  historyEmpty: { color: "#555", fontSize: 13, paddingTop: 10 },
  footnote: { color: "#444", fontSize: 11, textAlign: "center", marginTop: 4 },
});
