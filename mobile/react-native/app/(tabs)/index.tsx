import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View, Text, StyleSheet, Animated, Pressable,
  ScrollView, ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { getServerConfig } from '../../src/config/serverConfig';

const ASL_CLASSES = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const WS_RECONNECT_MS = 3000;
const N_CHANNELS = 8;
const WINDOW_SAMPLES = 40;

interface Prediction {
  class: string | null;
  confidence: number;
  latency_ms: number;
}

// --- Mock EMG frame generator (for simulator — no real BLE) ---
function makeMockFrame(): ArrayBuffer {
  const buf = new ArrayBuffer(WINDOW_SAMPLES * N_CHANNELS * 2);
  const view = new DataView(buf);
  for (let i = 0; i < WINDOW_SAMPLES * N_CHANNELS; i++) {
    const val = Math.floor((Math.random() - 0.5) * 2000);
    view.setInt16(i * 2, val, false); // big-endian
  }
  return buf;
}

export default function ASLLiveScreen() {
  const [connected, setConnected] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const streamInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;

  // Pulse animation on prediction
  const triggerPulse = useCallback(() => {
    Animated.sequence([
      Animated.timing(pulseAnim, { toValue: 1.15, duration: 100, useNativeDriver: true }),
      Animated.timing(pulseAnim, { toValue: 1, duration: 200, useNativeDriver: true }),
    ]).start();
  }, [pulseAnim]);

  const connect = useCallback(() => {
    const cfg = getServerConfig();
    const wsUrl = cfg.wsUrl;
    setError(null);

    try {
      const ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        setConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data as string) as Prediction;
          setPrediction(data);
          setLatency(data.latency_ms);
          if (data.class) {
            triggerPulse();
            setHistory(prev => [data.class!, ...prev].slice(0, 20));
          }
        } catch { /* ignore parse errors */ }
      };

      ws.onerror = () => setError('WebSocket error');
      ws.onclose = () => {
        setConnected(false);
        setStreaming(false);
        setTimeout(connect, WS_RECONNECT_MS);
      };

      wsRef.current = ws;
    } catch (e) {
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

  const toggleStream = useCallback(() => {
    if (streaming) {
      if (streamInterval.current) clearInterval(streamInterval.current);
      streamInterval.current = null;
      setStreaming(false);
    } else {
      // Send mock EMG frames every 100ms (10 Hz)
      streamInterval.current = setInterval(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(makeMockFrame());
        }
      }, 100);
      setStreaming(true);
    }
  }, [streaming]);

  const clearHistory = () => setHistory([]);

  const conf = prediction?.confidence ?? 0;
  const confColor = conf > 0.85 ? '#4CAF50' : conf > 0.6 ? '#FFC107' : '#f44336';

  return (
    <View style={styles.container}>
      {/* Connection status */}
      <View style={styles.statusBar}>
        <View style={[styles.dot, { backgroundColor: connected ? '#4CAF50' : '#f44336' }]} />
        <Text style={styles.statusText}>
          {connected ? 'Server connected' : 'Connecting...'}
        </Text>
        {latency != null && (
          <Text style={styles.latencyText}>{latency.toFixed(1)}ms</Text>
        )}
      </View>

      {error && <Text style={styles.errorText}>{error}</Text>}

      {/* Big prediction display */}
      <View style={styles.predictionContainer}>
        <Animated.Text style={[styles.predLetter, { transform: [{ scale: pulseAnim }] }]}>
          {prediction?.class ?? '?'}
        </Animated.Text>
        <View style={styles.confRow}>
          <View style={[styles.confBar, { width: `${conf * 100}%`, backgroundColor: confColor }]} />
        </View>
        <Text style={[styles.confText, { color: confColor }]}>
          {(conf * 100).toFixed(1)}% confidence
        </Text>
      </View>

      {/* Stream button */}
      <Pressable
        style={[styles.streamBtn, streaming && styles.streamBtnActive]}
        onPress={toggleStream}
        disabled={!connected}
      >
        <Ionicons
          name={streaming ? 'stop-circle' : 'radio-button-on'}
          size={24} color="#fff"
        />
        <Text style={styles.streamBtnText}>
          {streaming ? 'Stop Streaming' : 'Start Streaming (Mock EMG)'}
        </Text>
      </Pressable>

      {/* History */}
      <View style={styles.historyHeader}>
        <Text style={styles.historyTitle}>Recognition history</Text>
        <Pressable onPress={clearHistory}>
          <Text style={styles.clearBtn}>Clear</Text>
        </Pressable>
      </View>
      <ScrollView style={styles.historyScroll} horizontal showsHorizontalScrollIndicator={false}>
        {history.map((letter, i) => (
          <View key={i} style={[styles.historyChip, { opacity: 1 - i * 0.04 }]}>
            <Text style={styles.historyLetter}>{letter}</Text>
          </View>
        ))}
        {history.length === 0 && (
          <Text style={styles.historyEmpty}>
            {connected ? (streaming ? 'Waiting for prediction...' : 'Press Start to stream') : 'Connect to server first'}
          </Text>
        )}
      </ScrollView>

      {/* Note for simulator */}
      <Text style={styles.footnote}>
        Simulator mode: mock EMG data - Railway server - prediction
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#121212', padding: 16 },
  statusBar: { flexDirection: 'row', alignItems: 'center', marginBottom: 8 },
  dot: { width: 8, height: 8, borderRadius: 4, marginRight: 8 },
  statusText: { color: '#aaa', fontSize: 13, flex: 1 },
  latencyText: { color: '#666', fontSize: 12 },
  errorText: { color: '#f44336', fontSize: 13, marginBottom: 8 },
  predictionContainer: {
    flex: 1, alignItems: 'center', justifyContent: 'center',
    maxHeight: 300,
  },
  predLetter: {
    fontSize: 160, fontWeight: '800', color: '#fff',
    textShadowColor: '#4CAF5066', textShadowRadius: 40,
    lineHeight: 180,
  },
  confRow: {
    width: '80%', height: 6, backgroundColor: '#333',
    borderRadius: 3, overflow: 'hidden', marginTop: 8,
  },
  confBar: { height: '100%', borderRadius: 3 },
  confText: { fontSize: 14, marginTop: 6, fontWeight: '600' },
  streamBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    backgroundColor: '#1e3a2f', borderRadius: 12, padding: 16,
    marginVertical: 16, gap: 10,
  },
  streamBtnActive: { backgroundColor: '#3a1e1e' },
  streamBtnText: { color: '#fff', fontWeight: '600', fontSize: 15 },
  historyHeader: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 },
  historyTitle: { color: '#aaa', fontSize: 13, fontWeight: '600' },
  clearBtn: { color: '#4CAF50', fontSize: 13 },
  historyScroll: { maxHeight: 56, marginBottom: 8 },
  historyChip: {
    width: 40, height: 40, borderRadius: 8,
    backgroundColor: '#2a2a2a', alignItems: 'center', justifyContent: 'center',
    marginRight: 8,
  },
  historyLetter: { color: '#fff', fontSize: 20, fontWeight: '700' },
  historyEmpty: { color: '#555', fontSize: 13, paddingTop: 10 },
  footnote: { color: '#444', fontSize: 11, textAlign: 'center', marginTop: 4 },
});
