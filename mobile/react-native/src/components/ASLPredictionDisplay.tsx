/**
 * ASLPredictionDisplay.tsx
 * Main visual output component: shows the current ASL sign/letter prediction,
 * a confidence bar, and a scrolling history of the last 5 spoken signs.
 *
 * Fades in each new prediction with a spring animation.
 */

import React, { useEffect, useRef } from 'react';
import {
  Animated,
  Easing,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface HistoryEntry {
  id: number;
  label: string;
}

interface ASLPredictionDisplayProps {
  label: string | null;
  confidence: number; // 0.0 – 1.0
  history: HistoryEntry[]; // last 5 predictions (newest first), id must be stable
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const ASLPredictionDisplay: React.FC<ASLPredictionDisplayProps> = ({
  label,
  confidence,
  history,
}) => {
  // Fade animation for the main letter
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;
  const confBarAnim = useRef(new Animated.Value(0)).current;

  // Trigger fade-in whenever the label changes
  useEffect(() => {
    if (!label) return;

    // Reset and re-animate
    fadeAnim.setValue(0);
    scaleAnim.setValue(0.8);

    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 250,
        easing: Easing.out(Easing.ease),
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        friction: 6,
        tension: 80,
        useNativeDriver: true,
      }),
    ]).start();
  }, [label, fadeAnim, scaleAnim]);

  // Animate the confidence bar width
  useEffect(() => {
    Animated.timing(confBarAnim, {
      toValue: confidence,
      duration: 300,
      easing: Easing.out(Easing.ease),
      useNativeDriver: false,
    }).start();
  }, [confidence, confBarAnim]);

  const confBarWidth = confBarAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  const confColor =
    confidence > 0.75 ? '#2ecc71' : confidence > 0.5 ? '#f39c12' : '#e74c3c';

  const confPercent = Math.round(confidence * 100);

  return (
    <View style={styles.container}>
      {/* ---------------------------------------------------------------- */}
      {/* Main letter display                                              */}
      {/* ---------------------------------------------------------------- */}
      <View style={styles.letterContainer}>
        {label ? (
          <Animated.Text
            style={[
              styles.letterText,
              { opacity: fadeAnim, transform: [{ scale: scaleAnim }] },
            ]}
            accessibilityLabel={`Detected sign: ${label}`}
          >
            {label}
          </Animated.Text>
        ) : (
          <Text style={styles.placeholderText}>Waiting…</Text>
        )}
      </View>

      {/* ---------------------------------------------------------------- */}
      {/* Confidence bar                                                   */}
      {/* ---------------------------------------------------------------- */}
      <View style={styles.confSection}>
        <View style={styles.confLabelRow}>
          <Text style={styles.confLabel}>Confidence</Text>
          <Text style={[styles.confValue, { color: confColor }]}>
            {label ? `${confPercent}%` : '—'}
          </Text>
        </View>
        <View style={styles.confTrack}>
          <Animated.View
            style={[
              styles.confFill,
              { width: confBarWidth, backgroundColor: confColor },
            ]}
          />
        </View>
      </View>

      {/* ---------------------------------------------------------------- */}
      {/* Scrolling history                                                */}
      {/* ---------------------------------------------------------------- */}
      <View style={styles.historySection}>
        <Text style={styles.historyTitle}>Recent Signs</Text>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.historyScroll}
          accessibilityLabel="Recent sign history"
        >
          {history.length === 0 ? (
            <Text style={styles.historyEmpty}>No signs detected yet</Text>
          ) : (
            history.map((item, idx) => (
              <View
                key={String(item.id)}
                style={[
                  styles.historyChip,
                  idx === 0 && styles.historyChipNewest,
                ]}
              >
                <Text
                  style={[
                    styles.historyChipText,
                    idx === 0 && styles.historyChipTextNewest,
                  ]}
                >
                  {item.label}
                </Text>
              </View>
            ))
          )}
        </ScrollView>
      </View>
    </View>
  );
};

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    paddingHorizontal: 24,
    paddingTop: 32,
  },

  // Main letter
  letterContainer: {
    width: 220,
    height: 220,
    borderRadius: 110,
    backgroundColor: '#1e1e2e',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#6c5ce7',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.5,
    shadowRadius: 16,
    elevation: 10,
    marginBottom: 32,
  },
  letterText: {
    fontSize: 120,
    fontWeight: '800',
    color: '#ffffff',
    lineHeight: 140,
    includeFontPadding: false,
  },
  placeholderText: {
    fontSize: 22,
    color: '#555577',
    fontStyle: 'italic',
  },

  // Confidence
  confSection: {
    width: '100%',
    marginBottom: 32,
  },
  confLabelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  confLabel: {
    color: '#a0a0b0',
    fontSize: 13,
    fontWeight: '500',
  },
  confValue: {
    fontSize: 13,
    fontWeight: '700',
  },
  confTrack: {
    height: 8,
    backgroundColor: '#2a2a3e',
    borderRadius: 4,
    overflow: 'hidden',
  },
  confFill: {
    height: '100%',
    borderRadius: 4,
  },

  // History
  historySection: {
    width: '100%',
  },
  historyTitle: {
    color: '#a0a0b0',
    fontSize: 13,
    fontWeight: '500',
    marginBottom: 10,
  },
  historyScroll: {
    flexDirection: 'row',
    gap: 8,
    paddingRight: 8,
  },
  historyChip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    backgroundColor: '#2a2a3e',
    borderRadius: 20,
  },
  historyChipNewest: {
    backgroundColor: '#6c5ce7',
  },
  historyChipText: {
    color: '#a0a0b0',
    fontSize: 16,
    fontWeight: '600',
  },
  historyChipTextNewest: {
    color: '#ffffff',
  },
  historyEmpty: {
    color: '#555577',
    fontSize: 13,
    fontStyle: 'italic',
  },
});
