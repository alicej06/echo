import React, { useState, useEffect, useRef, useCallback } from "react";
import { View, Text, StyleSheet, Animated, Pressable, ScrollView, ActivityIndicator } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";

const KEY_SERVER_URL = "settings:serverUrl";
const DEFAULT_WS_URL = "ws://localhost:8765";
const RECONNECT_MS = 3000;
const DEMO_LETTERS = "HELLOWORLD".split("");
const DEMO_SENTENCES = ["Hello, nice to meet you.", "How are you today?", "My name is Echo."];

export default function HomeScreen() {
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS_URL);
  const [connected, setConnected] = useState(false);
  const [device, setDevice] = useState<string | null>(null);
  const [letter, setLetter] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [topK, setTopK] = useState<[string, number][]>([]);
  const [sentence, setSentence] = useState<string | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [demo, setDemo] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const demoRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const demoIdxRef = useRef(0);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    AsyncStorage.getItem(KEY_SERVER_URL).then((v) => { if (v) setWsUrl(v); });
  }, []);

  const pulse = useCallback(() => {
    Animated.sequence([
      Animated.timing(pulseAnim, { toValue: 1.18, duration: 80, useNativeDriver: true }),
      Animated.timing(pulseAnim, { toValue: 1, duration: 220, useNativeDriver: true }),
    ]).start();
  }, [pulseAnim]);

  const connect = useCallback((url: string) => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    if (wsRef.current) { wsRef.current.onclose = null; wsRef.current.close(); }
    setError(null);
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data as string) as Record<string, unknown>;
          if (msg.type === "letter") {
            const l = String(msg.letter ?? "").toUpperCase();
            const conf = Number(msg.confidence ?? 0);
            const tk = (msg.top_k as [string, number][]) ?? [];
            setLetter(l); setConfidence(conf); setTopK(tk);
            setHistory((prev) => [l, ...prev].slice(0, 30));
            pulse();
          } else if (msg.type === "sentence") {
            setSentence(String(msg.text ?? ""));
          } else if (msg.type === "status") {
            setConnected(Boolean(msg.connected));
            setDevice(msg.connected ? String(msg.device ?? "Myo") : null);
          }
        } catch { /* ignore */ }
      };
      ws.onerror = () => setError("Cannot reach " + url);
      ws.onclose = () => {
        setConnected(false); setDevice(null);
        reconnectTimer.current = setTimeout(() => connect(url), RECONNECT_MS);
      };
    } catch { setError("Invalid URL: " + url); }
  }, [pulse]);

  useEffect(() => {
    connect(wsUrl);
    return () => { wsRef.current?.close(); if (reconnectTimer.current) clearTimeout(reconnectTimer.current); };
  }, [wsUrl, connect]);

  const startDemo = useCallback(() => {
    setDemo(true); demoIdxRef.current = 0;
    demoRef.current = setInterval(() => {
      const l = DEMO_LETTERS[demoIdxRef.current % DEMO_LETTERS.length];
      const conf = 0.55 + Math.random() * 0.4;
      demoIdxRef.current += 1;
      setLetter(l); setConfidence(conf); setTopK([[l, conf]]);
      setHistory((prev) => [l, ...prev].slice(0, 30));
      pulse();
      if (demoIdxRef.current % DEMO_LETTERS.length === 0) {
        setSentence(DEMO_SENTENCES[Math.floor(demoIdxRef.current / DEMO_LETTERS.length) % DEMO_SENTENCES.length]);
      }
    }, 700);
  }, [pulse]);

  const stopDemo = useCallback(() => {
    setDemo(false);
    if (demoRef.current) { clearInterval(demoRef.current); demoRef.current = null; }
  }, []);

  useEffect(() => () => { if (demoRef.current) clearInterval(demoRef.current); }, []);

  const confColor = confidence > 0.75 ? "#4CAF50" : confidence > 0.5 ? "#FFC107" : "#f44336";

  return (
    <View style={s.container}>
      <View style={s.statusBar}>
        <View style={[s.dot, { backgroundColor: (connected || demo) ? "#4CAF50" : "#f44336" }]} />
        <Text style={s.statusText}>
          {demo ? "Demo mode" : connected ? ("Connected  " + (device ?? "Myo")) : "Connecting..."}
        </Text>
      </View>

      {error && !demo && <Text style={s.errorText}>{error}</Text>}

      <View style={s.letterBox}>
        {!letter && !demo ? (
          <>
            <ActivityIndicator size="large" color="#6c5ce7" />
            <Text style={s.waitText}>Waiting for signal...</Text>
          </>
        ) : (
          <>
            <Animated.Text style={[s.bigLetter, { transform: [{ scale: pulseAnim }] }]}>
              {letter ?? "?"}
            </Animated.Text>
            <View style={s.barTrack}>
              <View style={[s.barFill, { width: `${confidence * 100}%`, backgroundColor: confColor }]} />
            </View>
            <Text style={[s.confText, { color: confColor }]}>
              {(confidence * 100).toFixed(0)}% confidence
            </Text>
            {topK.length > 1 && (
              <Text style={s.topKText}>
                {topK.slice(0, 3).map(([l, sc]) => `${l} ${(sc * 100).toFixed(0)}%`).join("   ")}
              </Text>
            )}
          </>
        )}
      </View>

      {sentence ? (
        <View style={s.sentenceBox}>
          <Text style={s.sentenceText}>{sentence}</Text>
        </View>
      ) : null}

      <View style={s.histHeader}>
        <Text style={s.histTitle}>Letter stream</Text>
        <Pressable onPress={() => { setHistory([]); setSentence(null); }} hitSlop={8}>
          <Text style={s.clearBtn}>Clear</Text>
        </Pressable>
      </View>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={s.histScroll}>
        {history.slice(0, 12).map((l, i) => (
          <View key={i} style={[s.chip, { opacity: Math.max(0.2, 1 - i * 0.07) }]}>
            <Text style={s.chipLetter}>{l}</Text>
          </View>
        ))}
        {history.length === 0 && <Text style={s.histEmpty}>Form an ASL letter...</Text>}
      </ScrollView>

      <Pressable style={[s.demoBtn, demo && s.demoBtnActive]} onPress={demo ? stopDemo : startDemo}>
        <Ionicons name={demo ? "stop-circle" : "play-circle"} size={20} color="#fff" />
        <Text style={s.demoBtnText}>{demo ? "Stop Demo" : "Demo Mode"}</Text>
      </Pressable>
    </View>
  );
}

const s = StyleSheet.create({
  container:    { flex: 1, backgroundColor: "#0f0f1a", padding: 16 },
  statusBar:    { flexDirection: "row", alignItems: "center", marginBottom: 6 },
  dot:          { width: 8, height: 8, borderRadius: 4, marginRight: 8 },
  statusText:   { color: "#888", fontSize: 13 },
  errorText:    { color: "#f44336", fontSize: 12, marginBottom: 6 },
  letterBox:    { flex: 1, alignItems: "center", justifyContent: "center", maxHeight: 300 },
  bigLetter:    { fontSize: 160, fontWeight: "800", color: "#fff", lineHeight: 180 },
  waitText:     { color: "#555", fontSize: 13, marginTop: 12 },
  barTrack:     { width: "70%", height: 6, backgroundColor: "#222", borderRadius: 3, overflow: "hidden", marginTop: 8 },
  barFill:      { height: "100%", borderRadius: 3 },
  confText:     { fontSize: 14, marginTop: 6, fontWeight: "600" },
  topKText:     { color: "#555", fontSize: 12, marginTop: 4 },
  sentenceBox:  { backgroundColor: "#1a1a2e", borderRadius: 12, padding: 14, marginBottom: 12, borderWidth: 1, borderColor: "#2a2a4e" },
  sentenceText: { color: "#c8b8ff", fontSize: 16, fontWeight: "500", textAlign: "center" },
  histHeader:   { flexDirection: "row", justifyContent: "space-between", marginBottom: 8 },
  histTitle:    { color: "#666", fontSize: 12, fontWeight: "600" },
  clearBtn:     { color: "#6c5ce7", fontSize: 12 },
  histScroll:   { maxHeight: 52, marginBottom: 12 },
  chip:         { width: 40, height: 40, borderRadius: 8, backgroundColor: "#1e1e2e", alignItems: "center", justifyContent: "center", marginRight: 8 },
  chipLetter:   { color: "#fff", fontSize: 20, fontWeight: "700" },
  histEmpty:    { color: "#333", fontSize: 13, paddingTop: 12 },
  demoBtn:      { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8, backgroundColor: "#2a1e4a", borderRadius: 12, padding: 14 },
  demoBtnActive:{ backgroundColor: "#3a1e2e" },
  demoBtnText:  { color: "#fff", fontWeight: "600", fontSize: 14 },
});
