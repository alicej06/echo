/**
 * CalibrationPrompt.tsx
 * Shows the current ASL letter to calibrate along with a visual description
 * of the hand shape, a rep counter, a progress bar, and capture feedback.
 */

import React, { useEffect, useRef, useState } from 'react';
import {
  Animated,
  Easing,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

// ---------------------------------------------------------------------------
// ASL hand shape descriptions (brief guidance for each letter)
// ---------------------------------------------------------------------------

const ASL_DESCRIPTIONS: Record<string, string> = {
  A: 'Fist with thumb resting on the side of the index finger',
  B: 'Fingers together and extended upward, thumb folded across palm',
  C: 'Hand curved in a "C" shape, fingers and thumb arced',
  D: 'Index finger points up, other fingers and thumb form an "O"',
  E: 'Fingers curled downward, thumb tucked under fingers',
  F: 'Index and thumb touch at tips, other three fingers extended',
  G: 'Index finger and thumb point to the side, other fingers folded',
  H: 'Index and middle fingers extended horizontally side by side',
  I: 'Pinky finger extended upward, other fingers folded into fist',
  J: 'Pinky extended — trace a "J" shape (motion letter)',
  K: 'Index and middle fingers spread in a "V", thumb between them',
  L: 'L-shape: index finger up, thumb out to the side',
  M: 'Three fingers folded over thumb — thumb shows under ring & pinky',
  N: 'Two fingers folded over thumb — thumb shows under middle finger',
  O: 'All fingers curved to touch the thumb, forming an "O"',
  P: 'Like K but hand points downward',
  Q: 'Like G but hand points downward',
  R: 'Index and middle fingers crossed',
  S: 'Fist with thumb tucked over folded fingers',
  T: 'Fist with thumb between index and middle fingers',
  U: 'Index and middle fingers extended and held together, pointing up',
  V: 'Index and middle fingers extended in a "V" (peace sign)',
  W: 'Index, middle, and ring fingers spread upward',
  X: 'Index finger hooked or crooked',
  Y: 'Thumb and pinky extended outward (hang-loose)',
  Z: 'Index finger traces a "Z" in the air (motion letter)',
};

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface CalibrationPromptProps {
  currentSign: string;       // e.g. "A"
  repNumber: number;         // 1-based current rep
  totalReps: number;         // total reps per letter (typically 5)
  totalProgress: number;     // overall calibration progress 0–1
  onCapture: () => void;     // called when user taps "Hold Sign"
  disabled?: boolean;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const CalibrationPrompt: React.FC<CalibrationPromptProps> = ({
  currentSign,
  repNumber,
  totalReps,
  totalProgress,
  onCapture,
  disabled = false,
}) => {
  const [captured, setCaptured] = useState(false);

  // Animated values
  const letterScale = useRef(new Animated.Value(0.7)).current;
  const feedbackOpacity = useRef(new Animated.Value(0)).current;
  const progressAnim = useRef(new Animated.Value(0)).current;

  // Re-animate letter when sign changes
  useEffect(() => {
    setCaptured(false);
    feedbackOpacity.setValue(0);

    letterScale.setValue(0.7);
    Animated.spring(letterScale, {
      toValue: 1,
      friction: 5,
      tension: 70,
      useNativeDriver: true,
    }).start();
  }, [currentSign, letterScale, feedbackOpacity]);

  // Animate progress bar
  useEffect(() => {
    Animated.timing(progressAnim, {
      toValue: totalProgress,
      duration: 400,
      easing: Easing.out(Easing.ease),
      useNativeDriver: false,
    }).start();
  }, [totalProgress, progressAnim]);

  const progressWidth = progressAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  const handleCapture = () => {
    if (disabled || captured) return;

    setCaptured(true);
    onCapture();

    // Show "Captured!" then fade out
    Animated.sequence([
      Animated.timing(feedbackOpacity, {
        toValue: 1,
        duration: 150,
        useNativeDriver: true,
      }),
      Animated.delay(600),
      Animated.timing(feedbackOpacity, {
        toValue: 0,
        duration: 300,
        useNativeDriver: true,
      }),
    ]).start(() => {
      setCaptured(false);
    });
  };

  const description =
    ASL_DESCRIPTIONS[currentSign] ?? 'Follow the sign shown above.';
  const progressPercent = Math.round(totalProgress * 100);

  return (
    <View style={styles.container}>
      {/* ---------------------------------------------------------------- */}
      {/* Header / progress                                                */}
      {/* ---------------------------------------------------------------- */}
      <Text style={styles.headerTitle}>Calibration</Text>
      <Text style={styles.headerSubtitle}>{progressPercent}% complete</Text>

      <View style={styles.progressTrack}>
        <Animated.View
          style={[styles.progressFill, { width: progressWidth }]}
        />
      </View>

      {/* ---------------------------------------------------------------- */}
      {/* Large letter display                                             */}
      {/* ---------------------------------------------------------------- */}
      <Animated.View
        style={[
          styles.letterCircle,
          { transform: [{ scale: letterScale }] },
        ]}
      >
        <Text style={styles.letterText}>{currentSign}</Text>
      </Animated.View>

      {/* ---------------------------------------------------------------- */}
      {/* Hand shape description                                           */}
      {/* ---------------------------------------------------------------- */}
      <View style={styles.descriptionBox}>
        <Text style={styles.descriptionLabel}>Hand shape:</Text>
        <Text style={styles.descriptionText}>{description}</Text>
      </View>

      {/* ---------------------------------------------------------------- */}
      {/* Rep indicator                                                    */}
      {/* ---------------------------------------------------------------- */}
      <View style={styles.repRow}>
        {Array.from({ length: totalReps }).map((_, i) => (
          <View
            key={i}
            style={[
              styles.repDot,
              i < repNumber - 1
                ? styles.repDotDone
                : i === repNumber - 1
                ? styles.repDotCurrent
                : styles.repDotPending,
            ]}
          />
        ))}
      </View>
      <Text style={styles.repLabel}>
        Rep {repNumber} of {totalReps}
      </Text>

      {/* ---------------------------------------------------------------- */}
      {/* Capture button                                                   */}
      {/* ---------------------------------------------------------------- */}
      <View style={styles.captureWrapper}>
        <Animated.Text
          style={[styles.capturedFeedback, { opacity: feedbackOpacity }]}
        >
          Captured!
        </Animated.Text>

        <TouchableOpacity
          onPress={handleCapture}
          disabled={disabled || captured}
          activeOpacity={0.75}
          accessibilityRole="button"
          accessibilityLabel={`Hold sign ${currentSign} to capture`}
          style={[
            styles.captureButton,
            (disabled || captured) && styles.captureButtonDisabled,
          ]}
        >
          <Text style={styles.captureButtonText}>
            {captured ? 'Captured!' : 'Hold Sign'}
          </Text>
        </TouchableOpacity>
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
    paddingTop: 16,
  },
  headerTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#ffffff',
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 13,
    color: '#a0a0b0',
    marginBottom: 12,
  },
  progressTrack: {
    width: '100%',
    height: 6,
    backgroundColor: '#2a2a3e',
    borderRadius: 3,
    overflow: 'hidden',
    marginBottom: 32,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#6c5ce7',
    borderRadius: 3,
  },

  // Letter
  letterCircle: {
    width: 180,
    height: 180,
    borderRadius: 90,
    backgroundColor: '#1e1e2e',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#6c5ce7',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.6,
    shadowRadius: 16,
    elevation: 10,
    marginBottom: 24,
    borderWidth: 3,
    borderColor: '#6c5ce7',
  },
  letterText: {
    fontSize: 100,
    fontWeight: '800',
    color: '#ffffff',
    lineHeight: 120,
    includeFontPadding: false,
  },

  // Description
  descriptionBox: {
    backgroundColor: '#1e1e2e',
    borderRadius: 12,
    padding: 16,
    width: '100%',
    marginBottom: 24,
  },
  descriptionLabel: {
    color: '#6c5ce7',
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 4,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  descriptionText: {
    color: '#c0c0d0',
    fontSize: 15,
    lineHeight: 22,
  },

  // Rep dots
  repRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 6,
  },
  repDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  repDotDone: { backgroundColor: '#2ecc71' },
  repDotCurrent: { backgroundColor: '#6c5ce7' },
  repDotPending: { backgroundColor: '#2a2a3e' },
  repLabel: {
    color: '#a0a0b0',
    fontSize: 13,
    marginBottom: 28,
  },

  // Capture
  captureWrapper: {
    width: '100%',
    alignItems: 'center',
  },
  capturedFeedback: {
    fontSize: 18,
    fontWeight: '700',
    color: '#2ecc71',
    marginBottom: 8,
  },
  captureButton: {
    width: '100%',
    paddingVertical: 18,
    backgroundColor: '#6c5ce7',
    borderRadius: 16,
    alignItems: 'center',
    shadowColor: '#6c5ce7',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.5,
    shadowRadius: 8,
    elevation: 6,
  },
  captureButtonDisabled: {
    opacity: 0.45,
  },
  captureButtonText: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '700',
  },
});
