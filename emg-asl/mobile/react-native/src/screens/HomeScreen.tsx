/**
 * HomeScreen.tsx
 * Main recognition screen for the MAIA EMG-ASL app.
 *
 * Layout:
 *   - Header with branding and MYO connection status
 *   - Large centered prediction label with animated confidence ring
 *   - Signal quality indicator (8-channel activity)
 *   - Connect / Disconnect button
 *   - Scrollable prediction transcript (last 10 entries)
 *
 * Connects to the Thalmic MYO Armband via BLE (direct protocol) or via
 * WebSocket when using the laptop inference server (MyoConnect path).
 */

import React, { useCallback, useRef } from 'react';
import {
  Animated,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

import { useOnDeviceASL } from '../hooks/useOnDeviceASL';
import type { ConnectionStatus } from '../bluetooth/BLEManager';

// ---------------------------------------------------------------------------
// Design tokens
// ---------------------------------------------------------------------------

const C = {
  // Backgrounds
  bg:          '#070711',
  surface:     '#111120',
  surfaceHigh: '#181830',
  border:      '#23234a',

  // Accent
  accent:      '#7c6ff7',
  accentDim:   '#4e48b2',
  accentGlow:  '#7c6ff720',

  // Semantic
  success:     '#00d68f',
  warning:     '#ffb547',
  error:       '#ff6b6b',

  // Text
  textPrimary:   '#f0f0ff',
  textSecondary: '#8888aa',
  textMuted:     '#444466',

  // Bar
  barTrack: '#1e1e3c',
  barFill:  '#7c6ff7',
} as const;

// ---------------------------------------------------------------------------
// StatusPill
// ---------------------------------------------------------------------------

interface StatusPillProps {
  status: ConnectionStatus;
}

function StatusPill({ status }: StatusPillProps) {
  const label =
    status === 'disconnected' ? 'Disconnected'
    : status === 'scanning'   ? 'Scanning...'
    : status === 'connecting' ? 'Connecting...'
    : status === 'connected'  ? 'MYO Connected'
    : 'Connection Error';

  const color =
    status === 'connected'                            ? C.success
    : status === 'error'                              ? C.error
    : status === 'scanning' || status === 'connecting' ? C.warning
    : C.textMuted;

  const isPulsing = status === 'scanning' || status === 'connecting';

  return (
    <View style={[styles.pill, { borderColor: color }]}>
      <View style={[
        styles.pillDot,
        { backgroundColor: color },
        isPulsing && styles.pillDotPulsing,
      ]} />
      <Text style={[styles.pillLabel, { color }]}>{label}</Text>
    </View>
  );
}

// ---------------------------------------------------------------------------
// ConfidenceBar
// ---------------------------------------------------------------------------

interface ConfidenceBarProps {
  confidence: number; // 0..1
}

function ConfidenceBar({ confidence }: ConfidenceBarProps) {
  const pct = Math.round(confidence * 100);
  const barColor =
    pct >= 90 ? C.success
    : pct >= 75 ? C.accent
    : pct >= 50 ? C.warning
    : C.error;

  return (
    <View style={styles.barContainer}>
      <Text style={styles.barLabel}>Confidence</Text>
      <View style={styles.barTrack}>
        <View
          style={[
            styles.barFill,
            { width: `${pct}%` as `${number}%`, backgroundColor: barColor },
          ]}
        />
      </View>
      <Text style={[styles.barPct, { color: barColor }]}>{pct}%</Text>
    </View>
  );
}

// ---------------------------------------------------------------------------
// ModelBadge
// ---------------------------------------------------------------------------

interface ModelBadgeProps {
  modelLoaded: boolean;
  modelLoadError: string | null;
}

function ModelBadge({ modelLoaded, modelLoadError }: ModelBadgeProps) {
  if (modelLoaded) {
    return (
      <View style={[styles.badge, { borderColor: C.success }]}>
        <Text style={[styles.badgeText, { color: C.success }]}>On-device model</Text>
      </View>
    );
  }
  if (modelLoadError !== null) {
    return (
      <View style={[styles.badge, { borderColor: C.warning }]}>
        <Text style={[styles.badgeText, { color: C.warning }]}>Server mode</Text>
      </View>
    );
  }
  return null;
}

// ---------------------------------------------------------------------------
// HomeScreen
// ---------------------------------------------------------------------------

export default function HomeScreen() {
  const {
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
  } = useOnDeviceASL({
    deviceName: 'Myo',
    autoConnect: false,
    fallbackToServer: true,
    serverUrl: 'ws://localhost:8000/stream',
  });

  const isScanning =
    connectionStatus === 'scanning' || connectionStatus === 'connecting';

  const handleConnectPress = useCallback(async () => {
    if (isConnected || isScanning) {
      await disconnect();
    } else {
      await connect();
    }
  }, [isConnected, isScanning, connect, disconnect]);

  const connectLabel =
    isConnected  ? 'Disconnect'
    : isScanning ? 'Cancel'
    : 'Connect to MYO';

  const connectBg =
    isConnected || isScanning ? C.error : C.accent;

  const recentHistory = predictionHistory.slice(0, 10);
  const hasHistory = recentHistory.length > 0;

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" backgroundColor={C.bg} />

      {/* ------------------------------------------------------------------ */}
      {/* Header                                                              */}
      {/* ------------------------------------------------------------------ */}
      <View style={styles.header}>
        <View>
          <Text style={styles.appTitle}>MAIA</Text>
          <Text style={styles.appSubtitle}>EMG Sign Language · MYO Armband</Text>
        </View>
        <ModelBadge modelLoaded={modelLoaded} modelLoadError={modelLoadError} />
      </View>

      {/* ------------------------------------------------------------------ */}
      {/* Status pill                                                         */}
      {/* ------------------------------------------------------------------ */}
      <View style={styles.pillRow}>
        <StatusPill status={connectionStatus} />
      </View>

      {/* ------------------------------------------------------------------ */}
      {/* Prediction display                                                  */}
      {/* ------------------------------------------------------------------ */}
      <View style={styles.predictionCard}>
        <Text style={styles.predictionHint}>Recognized sign</Text>
        <Text
          style={[
            styles.predictionLabel,
            lastLabel === null && styles.predictionLabelEmpty,
          ]}
          numberOfLines={1}
          adjustsFontSizeToFit
        >
          {lastLabel !== null ? lastLabel : '—'}
        </Text>
        {lastLabel !== null && (
          <ConfidenceBar confidence={lastConfidence} />
        )}
      </View>

      {/* ------------------------------------------------------------------ */}
      {/* Connect button                                                      */}
      {/* ------------------------------------------------------------------ */}
      <TouchableOpacity
        style={[styles.connectButton, { backgroundColor: connectBg }]}
        onPress={handleConnectPress}
        activeOpacity={0.82}
        accessibilityRole="button"
        accessibilityLabel={connectLabel}
      >
        <View style={styles.connectButtonInner}>
          <View style={[
            styles.connectDot,
            { backgroundColor: isConnected ? C.success : isScanning ? C.warning : C.textPrimary },
          ]} />
          <Text style={styles.connectButtonText}>{connectLabel}</Text>
        </View>
      </TouchableOpacity>

      {/* ------------------------------------------------------------------ */}
      {/* Stats row                                                           */}
      {/* ------------------------------------------------------------------ */}
      <View style={styles.statsRow}>
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{windowsProcessed.toLocaleString()}</Text>
          <Text style={styles.statLabel}>Windows</Text>
        </View>
        <View style={styles.statDivider} />
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{predictionsAccepted.toLocaleString()}</Text>
          <Text style={styles.statLabel}>Predictions</Text>
        </View>
        <View style={styles.statDivider} />
        <View style={styles.statItem}>
          <Text style={styles.statValue}>200 Hz</Text>
          <Text style={styles.statLabel}>Sample Rate</Text>
        </View>
      </View>

      {/* ------------------------------------------------------------------ */}
      {/* Transcript                                                          */}
      {/* ------------------------------------------------------------------ */}
      <View style={styles.transcriptSection}>
        <View style={styles.transcriptHeader}>
          <Text style={styles.transcriptTitle}>Transcript</Text>
          {hasHistory && (
            <TouchableOpacity
              onPress={clearHistory}
              accessibilityRole="button"
              accessibilityLabel="Clear transcript"
              hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
            >
              <Text style={styles.clearText}>Clear</Text>
            </TouchableOpacity>
          )}
        </View>

        <ScrollView
          style={styles.transcriptScroll}
          contentContainerStyle={[
            styles.transcriptContent,
            !hasHistory && styles.transcriptContentEmpty,
          ]}
          showsVerticalScrollIndicator={false}
        >
          {!hasHistory ? (
            <View style={styles.emptyState}>
              <Text style={styles.emptyTitle}>No signs detected yet</Text>
              <Text style={styles.emptyHint}>Connect the MYO armband and begin signing</Text>
            </View>
          ) : (
            recentHistory.map((entry, i) => (
              <View
                key={String(entry.timestamp)}
                style={[styles.transcriptRow, i === 0 && styles.transcriptRowFirst]}
              >
                <Text style={styles.transcriptLabel}>{entry.label}</Text>
                <Text style={styles.transcriptConf}>
                  {Math.round(entry.confidence * 100)}%
                </Text>
                <Text style={styles.transcriptTime}>
                  {new Date(entry.timestamp).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                  })}
                </Text>
              </View>
            ))
          )}
        </ScrollView>
      </View>
    </SafeAreaView>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: C.bg,
  },

  // ── Header ──────────────────────────────────────────────────────────────
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    paddingHorizontal: 24,
    paddingTop: 20,
    paddingBottom: 4,
  },
  appTitle: {
    fontSize: 32,
    fontWeight: '900',
    color: C.textPrimary,
    letterSpacing: 3,
  },
  appSubtitle: {
    fontSize: 11,
    color: C.textMuted,
    marginTop: 3,
    letterSpacing: 0.8,
    textTransform: 'uppercase',
  },

  // ── Status pill ──────────────────────────────────────────────────────────
  pillRow: {
    paddingHorizontal: 24,
    paddingVertical: 12,
  },
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    gap: 7,
    paddingVertical: 5,
    paddingHorizontal: 12,
    borderRadius: 100,
    borderWidth: 1,
    backgroundColor: C.surface,
  },
  pillDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
  },
  pillDotPulsing: {
    opacity: 0.85,
  },
  pillLabel: {
    fontSize: 12,
    fontWeight: '600',
    letterSpacing: 0.4,
  },

  // ── Model badge ──────────────────────────────────────────────────────────
  badge: {
    paddingVertical: 4,
    paddingHorizontal: 10,
    borderRadius: 8,
    borderWidth: 1,
    backgroundColor: C.surface,
    alignSelf: 'flex-start',
    marginTop: 4,
  },
  badgeText: {
    fontSize: 11,
    fontWeight: '600',
    letterSpacing: 0.4,
  },

  // ── Prediction card ──────────────────────────────────────────────────────
  predictionCard: {
    marginHorizontal: 24,
    backgroundColor: C.surface,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: C.border,
    paddingHorizontal: 28,
    paddingTop: 20,
    paddingBottom: 24,
    marginBottom: 16,
    alignItems: 'center',
  },
  predictionHint: {
    fontSize: 11,
    color: C.textMuted,
    letterSpacing: 1,
    textTransform: 'uppercase',
    marginBottom: 8,
    alignSelf: 'flex-start',
  },
  predictionLabel: {
    fontSize: 100,
    fontWeight: '900',
    color: C.textPrimary,
    letterSpacing: 6,
    alignSelf: 'center',
    lineHeight: 110,
    marginBottom: 12,
  },
  predictionLabelEmpty: {
    color: C.textMuted,
    fontSize: 72,
    letterSpacing: 0,
  },

  // ── Confidence bar ───────────────────────────────────────────────────────
  barContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    width: '100%',
  },
  barLabel: {
    fontSize: 11,
    color: C.textSecondary,
    width: 76,
    letterSpacing: 0.3,
  },
  barTrack: {
    flex: 1,
    height: 6,
    backgroundColor: C.barTrack,
    borderRadius: 3,
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: 3,
  },
  barPct: {
    fontSize: 12,
    fontWeight: '700',
    width: 36,
    textAlign: 'right',
  },

  // ── Connect button ───────────────────────────────────────────────────────
  connectButton: {
    marginHorizontal: 24,
    borderRadius: 14,
    paddingVertical: 15,
    marginBottom: 16,
  },
  connectButtonInner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  connectDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    opacity: 0.9,
  },
  connectButtonText: {
    fontSize: 16,
    fontWeight: '700',
    color: C.textPrimary,
    letterSpacing: 0.6,
  },

  // ── Stats row ────────────────────────────────────────────────────────────
  statsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: C.surface,
    marginHorizontal: 24,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: C.border,
    paddingVertical: 14,
    marginBottom: 16,
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 18,
    fontWeight: '700',
    color: C.textPrimary,
  },
  statLabel: {
    fontSize: 10,
    color: C.textMuted,
    marginTop: 3,
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  statDivider: {
    width: 1,
    height: 30,
    backgroundColor: C.border,
  },

  // ── Transcript ───────────────────────────────────────────────────────────
  transcriptSection: {
    flex: 1,
    marginHorizontal: 24,
    marginBottom: 16,
  },
  transcriptHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  transcriptTitle: {
    fontSize: 11,
    fontWeight: '700',
    color: C.textSecondary,
    letterSpacing: 1.2,
    textTransform: 'uppercase',
  },
  clearText: {
    fontSize: 13,
    color: C.accent,
    fontWeight: '600',
  },
  transcriptScroll: {
    flex: 1,
    backgroundColor: C.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: C.border,
  },
  transcriptContent: {
    paddingHorizontal: 16,
    paddingVertical: 4,
  },
  transcriptContentEmpty: {
    flex: 1,
    justifyContent: 'center',
  },
  transcriptRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: C.border,
  },
  transcriptRowFirst: {
    // most-recent entry — subtle highlight
    backgroundColor: C.accentGlow,
    borderRadius: 8,
    paddingHorizontal: 8,
    marginHorizontal: -8,
  },
  transcriptLabel: {
    flex: 1,
    fontSize: 16,
    fontWeight: '800',
    color: C.textPrimary,
    letterSpacing: 0.5,
  },
  transcriptConf: {
    fontSize: 13,
    color: C.accent,
    fontWeight: '700',
    width: 44,
    textAlign: 'right',
  },
  transcriptTime: {
    fontSize: 11,
    color: C.textMuted,
    width: 84,
    textAlign: 'right',
  },

  // ── Empty state ──────────────────────────────────────────────────────────
  emptyState: {
    alignItems: 'center',
    paddingVertical: 32,
  },
  emptyTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: C.textSecondary,
    marginBottom: 6,
  },
  emptyHint: {
    fontSize: 12,
    color: C.textMuted,
    textAlign: 'center',
    maxWidth: 200,
    lineHeight: 18,
  },
});
