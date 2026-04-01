import React, { useState, useRef, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  Animated,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { getServerConfig } from "../../src/config/serverConfig";

const ASL_CLASSES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

// Cycle through alphabet automatically for demo purposes
const DEMO_SEQUENCE = [...ASL_CLASSES, ..."MAIA".split("")];

export default function DemoScreen() {
  const [running, setRunning] = useState(false);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [serverStatus, setServerStatus] = useState<"unknown" | "ok" | "error">(
    "unknown",
  );
  const [predictions, setPredictions] = useState<
    Array<{ letter: string; conf: number }>
  >([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const scaleAnim = useRef(new Animated.Value(1)).current;

  // Check server health on mount
  useEffect(() => {
    const cfg = getServerConfig();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    fetch(`${cfg.serverUrl}/health`, { signal: controller.signal })
      .then((r) => setServerStatus(r.ok ? "ok" : "error"))
      .catch(() => setServerStatus("error"))
      .finally(() => clearTimeout(timer));
  }, []);

  const animate = () => {
    Animated.sequence([
      Animated.timing(scaleAnim, {
        toValue: 1.2,
        duration: 150,
        useNativeDriver: true,
      }),
      Animated.timing(scaleAnim, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }),
    ]).start();
  };

  const startDemo = () => {
    setRunning(true);
    setCurrentIdx(0);
    intervalRef.current = setInterval(() => {
      setCurrentIdx((i) => {
        const next = (i + 1) % DEMO_SEQUENCE.length;
        animate();
        // Simulate a prediction
        setPredictions((prev) =>
          [
            {
              letter: DEMO_SEQUENCE[next],
              conf: 0.75 + Math.random() * 0.24,
            },
            ...prev,
          ].slice(0, 10),
        );
        return next;
      });
    }, 800);
  };

  const stopDemo = () => {
    setRunning(false);
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  useEffect(
    () => () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    },
    [],
  );

  const letter = DEMO_SEQUENCE[currentIdx];

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.serverRow}>
        <View
          style={[
            styles.dot,
            {
              backgroundColor:
                serverStatus === "ok"
                  ? "#4CAF50"
                  : serverStatus === "error"
                    ? "#f44336"
                    : "#888",
            },
          ]}
        />
        <Text style={styles.serverText}>
          {serverStatus === "ok"
            ? "Railway server online"
            : serverStatus === "error"
              ? "Server offline - check Settings"
              : "Checking server..."}
        </Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Sign Demo Playback</Text>
        <Text style={styles.cardSubtitle}>
          Simulates the UI experience of real-time ASL recognition. Connect the
          MYO Armband to see live predictions.
        </Text>

        <Animated.Text
          style={[styles.bigLetter, { transform: [{ scale: scaleAnim }] }]}
        >
          {letter}
        </Animated.Text>

        <Pressable
          style={[styles.demoBtn, running && styles.demoBtnStop]}
          onPress={running ? stopDemo : startDemo}
        >
          <Ionicons name={running ? "stop" : "play"} size={20} color="#fff" />
          <Text style={styles.demoBtnText}>
            {running ? "Stop Demo" : "Start Demo"}
          </Text>
        </Pressable>
      </View>

      {predictions.length > 0 && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Simulated Predictions</Text>
          {predictions.map((p, i) => (
            <View key={i} style={styles.predRow}>
              <Text style={[styles.predLetter, { opacity: 1 - i * 0.08 }]}>
                {p.letter}
              </Text>
              <View style={styles.barBg}>
                <View style={[styles.barFill, { width: `${p.conf * 100}%` }]} />
              </View>
              <Text style={[styles.predConf, { opacity: 1 - i * 0.08 }]}>
                {(p.conf * 100).toFixed(0)}%
              </Text>
            </View>
          ))}
        </View>
      )}

      <View style={styles.card}>
        <Text style={styles.cardTitle}>How it works</Text>
        {[
          ["MYO Armband", "8-channel sEMG - BLE - iPhone"],
          ["Signal Processing", "200Hz - bandpass - 40-sample windows"],
          ["On-Device ONNX", "LSTM inference in ~15ms"],
          ["Server Fallback", "Railway WebSocket if on-device fails"],
        ].map(([title, desc]) => (
          <View key={title} style={styles.stepRow}>
            <Ionicons name="chevron-forward" size={14} color="#4CAF50" />
            <View style={{ marginLeft: 8 }}>
              <Text style={styles.stepTitle}>{title}</Text>
              <Text style={styles.stepDesc}>{desc}</Text>
            </View>
          </View>
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#121212" },
  content: { padding: 16, paddingBottom: 48 },
  serverRow: { flexDirection: "row", alignItems: "center", marginBottom: 16 },
  dot: { width: 8, height: 8, borderRadius: 4, marginRight: 8 },
  serverText: { color: "#aaa", fontSize: 13 },
  card: {
    backgroundColor: "#1e1e1e",
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  cardTitle: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "700",
    marginBottom: 4,
  },
  cardSubtitle: {
    color: "#888",
    fontSize: 13,
    marginBottom: 20,
    lineHeight: 18,
  },
  bigLetter: {
    fontSize: 120,
    fontWeight: "900",
    color: "#fff",
    textAlign: "center",
    lineHeight: 140,
  },
  demoBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#4CAF50",
    borderRadius: 12,
    padding: 14,
    gap: 8,
    marginTop: 16,
  },
  demoBtnStop: { backgroundColor: "#f44336" },
  demoBtnText: { color: "#fff", fontWeight: "700", fontSize: 15 },
  predRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
    gap: 12,
  },
  predLetter: { color: "#fff", fontSize: 18, fontWeight: "700", width: 28 },
  barBg: {
    flex: 1,
    height: 6,
    backgroundColor: "#333",
    borderRadius: 3,
    overflow: "hidden",
  },
  barFill: { height: "100%", backgroundColor: "#4CAF50", borderRadius: 3 },
  predConf: { color: "#aaa", fontSize: 12, width: 36, textAlign: "right" },
  stepRow: { flexDirection: "row", alignItems: "flex-start", marginBottom: 12 },
  stepTitle: { color: "#fff", fontSize: 13, fontWeight: "600" },
  stepDesc: { color: "#888", fontSize: 12 },
});
