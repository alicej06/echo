/**
 * CalibrationScreen.tsx
 * Full guided calibration flow:
 *   1. User enters their name / ID
 *   2. Steps through all 26 ASL letters, 5 reps each
 *   3. 3-2-1 countdown before each capture
 *   4. Shows accuracy summary when finished
 *   5. Persists result to AsyncStorage
 */

import React, { useCallback, useEffect, useReducer, useRef } from "react";
import {
  Alert,
  Animated,
  Easing,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";

import { CalibrationPrompt } from "../components/CalibrationPrompt";
import {
  useCalibration,
  ASL_ALPHABET,
  REPS_PER_LETTER,
  TOTAL_SAMPLES,
  type CalibrationStats,
} from "../hooks/useCalibration";
import { useEMGConnection } from "../hooks/useEMGConnection";
import type { EMGStreamProcessor } from "../bluetooth/EMGStream";

// ---------------------------------------------------------------------------
// Phase types
// ---------------------------------------------------------------------------

type Phase =
  | "idle" // intro / name entry
  | "countdown" // 3-2-1 before capture
  | "calibrating" // actively stepping through signs
  | "done"; // results summary

const COUNTDOWN_FROM = 3;

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

type State = {
  phase: Phase;
  userId: string;
  countdown: number;
  stats: CalibrationStats | null;
  busyCapturing: boolean;
};

type Action =
  | { type: "START_COUNTDOWN" }
  | { type: "TICK"; remaining: number }
  | { type: "SET_PHASE"; phase: Phase }
  | { type: "SET_USER_ID"; userId: string }
  | { type: "FINISH"; stats: CalibrationStats }
  | { type: "SET_BUSY"; busy: boolean }
  | { type: "RESET" };

const initialState: State = {
  phase: "idle",
  userId: "",
  countdown: COUNTDOWN_FROM,
  stats: null,
  busyCapturing: false,
};

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "START_COUNTDOWN":
      return {
        ...state,
        busyCapturing: true,
        phase: "countdown",
        countdown: COUNTDOWN_FROM,
      };
    case "TICK":
      return { ...state, countdown: action.remaining };
    case "SET_PHASE":
      return { ...state, phase: action.phase };
    case "SET_USER_ID":
      return { ...state, userId: action.userId };
    case "FINISH":
      return { ...state, stats: action.stats, phase: "done" };
    case "SET_BUSY":
      return { ...state, busyCapturing: action.busy };
    case "RESET":
      return { ...initialState };
    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CalibrationScreen() {
  const router = useRouter();
  const { connectionStatus, serverUrl } = useEMGConnection();

  // We need the processor ref — in a real app this would be passed down via
  // context.  For simplicity we create a local ref here that mirrors the one
  // inside useEMGConnection.  The hook's processor is exposed via the context;
  // wiring note: wrap app in an EMGConnectionContext to share the ref cleanly.
  const processorRef = useRef<EMGStreamProcessor | null>(null);

  const {
    isCalibrating,
    currentPrompt,
    samplesRecorded,
    progress,
    startSession,
    recordSample,
    finishCalibration,
    resetCalibration,
  } = useCalibration(processorRef, serverUrl);

  const [state, dispatch] = useReducer(reducer, initialState);
  const { phase, userId, countdown, stats, busyCapturing } = state;

  const countdownAnim = useRef(new Animated.Value(1)).current;
  const countdownTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (countdownTimer.current) clearInterval(countdownTimer.current);
    };
  }, []);

  // -------------------------------------------------------------------------
  // Derived
  // -------------------------------------------------------------------------

  const letterIndex = Math.floor(samplesRecorded / REPS_PER_LETTER);
  const repNumber = (samplesRecorded % REPS_PER_LETTER) + 1;
  const currentSign = ASL_ALPHABET[letterIndex] ?? "Z";

  // -------------------------------------------------------------------------
  // Start calibration
  // -------------------------------------------------------------------------

  const handleStart = useCallback(async () => {
    if (!userId.trim()) {
      Alert.alert("Name Required", "Please enter your name or ID to begin.");
      return;
    }

    if (connectionStatus !== "connected") {
      Alert.alert(
        "Not Connected",
        "Please connect the EMG armband before calibrating.",
      );
      return;
    }

    try {
      await startSession(userId.trim());
      dispatch({ type: "SET_PHASE", phase: "calibrating" });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      Alert.alert("Error", `Failed to start calibration: ${msg}`);
    }
  }, [userId, connectionStatus, startSession]);

  // -------------------------------------------------------------------------
  // Countdown then capture
  // -------------------------------------------------------------------------

  const doCapture = useCallback(async () => {
    try {
      await recordSample(currentSign);
      if (!mountedRef.current) return;

      const newTotal = samplesRecorded + 1;
      if (newTotal >= TOTAL_SAMPLES) {
        const result = await finishCalibration();
        if (mountedRef.current && result !== null)
          dispatch({ type: "FINISH", stats: result });
      } else {
        dispatch({ type: "SET_PHASE", phase: "calibrating" });
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      Alert.alert("Capture Error", msg);
      dispatch({ type: "SET_PHASE", phase: "calibrating" });
    } finally {
      if (mountedRef.current) dispatch({ type: "SET_BUSY", busy: false });
    }
  }, [currentSign, recordSample, samplesRecorded, finishCalibration]);

  const startCountdown = useCallback(() => {
    if (phase !== "calibrating" || busyCapturing) return;

    dispatch({ type: "START_COUNTDOWN" }); // sets busyCapturing, phase, countdown atomically

    let remaining = COUNTDOWN_FROM;

    const tick = () => {
      countdownAnim.setValue(1.4);
      Animated.timing(countdownAnim, {
        toValue: 1,
        duration: 600,
        easing: Easing.out(Easing.ease),
        useNativeDriver: true,
      }).start();

      remaining -= 1;
      if (!mountedRef.current) return;
      dispatch({ type: "TICK", remaining });

      if (remaining <= 0) {
        if (countdownTimer.current) clearInterval(countdownTimer.current);
        countdownTimer.current = null;
        doCapture();
      }
    };

    countdownTimer.current = setInterval(tick, 1_000);
  }, [phase, busyCapturing, countdownAnim, doCapture]);

  // -------------------------------------------------------------------------
  // Reset
  // -------------------------------------------------------------------------

  const handleReset = useCallback(() => {
    Alert.alert(
      "Clear Calibration",
      "This will delete all saved calibration data. Continue?",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Clear",
          style: "destructive",
          onPress: async () => {
            await resetCalibration();
            if (mountedRef.current) dispatch({ type: "RESET" });
          },
        },
      ],
    );
  }, [resetCalibration]);

  // -------------------------------------------------------------------------
  // Render helpers
  // -------------------------------------------------------------------------

  const renderIdle = () => (
    <ScrollView
      contentContainerStyle={styles.idleContainer}
      keyboardShouldPersistTaps="handled"
    >
      <Ionicons name="hand-left-outline" size={72} color="#6c5ce7" />
      <Text style={styles.idleTitle}>Calibrate Your Signs</Text>
      <Text style={styles.idleBody}>
        We'll walk you through all 26 ASL letters.{"\n"}
        Hold each sign steady and tap "Hold Sign" when prompted.{"\n\n"}
        This takes about 5–10 minutes.
      </Text>

      <TextInput
        style={styles.input}
        placeholder="Your name or user ID"
        placeholderTextColor="#555577"
        value={userId}
        onChangeText={(v) => dispatch({ type: "SET_USER_ID", userId: v })}
        autoCapitalize="none"
        returnKeyType="done"
        accessibilityLabel="User name or ID"
      />

      {connectionStatus !== "connected" && (
        <View style={styles.warningBox}>
          <Ionicons name="warning-outline" size={18} color="#f39c12" />
          <Text style={styles.warningText}>
            EMG armband not connected. Connect before starting.
          </Text>
        </View>
      )}

      <TouchableOpacity
        style={[
          styles.primaryButton,
          connectionStatus !== "connected" && styles.buttonDisabled,
        ]}
        onPress={handleStart}
        disabled={connectionStatus !== "connected"}
        accessibilityRole="button"
        accessibilityLabel="Start calibration"
      >
        <Text style={styles.primaryButtonText}>Begin Calibration</Text>
      </TouchableOpacity>
    </ScrollView>
  );

  const renderCountdown = () => (
    <View style={styles.centeredOverlay}>
      <Text style={styles.countdownLabel}>Get ready…</Text>
      <Animated.Text
        style={[
          styles.countdownNumber,
          { transform: [{ scale: countdownAnim }] },
        ]}
      >
        {countdown > 0 ? countdown : "GO!"}
      </Animated.Text>
      <Text style={styles.countdownSign}>Sign: {currentSign}</Text>
    </View>
  );

  const renderCalibrating = () => (
    <CalibrationPrompt
      currentSign={currentSign}
      repNumber={repNumber}
      totalReps={REPS_PER_LETTER}
      totalProgress={progress}
      onCapture={startCountdown}
      disabled={busyCapturing}
    />
  );

  const renderDone = () => {
    const acc = stats ? Math.round(stats.accuracy * 100) : 0;

    return (
      <ScrollView contentContainerStyle={styles.doneContainer}>
        <Ionicons name="checkmark-circle" size={80} color="#2ecc71" />
        <Text style={styles.doneTitle}>Calibration Complete!</Text>
        <Text style={styles.doneAccuracy}>{acc}% Accuracy</Text>
        <Text style={styles.doneBody}>
          Your personal EMG model has been saved. You can now use EMG ASL for
          real-time sign recognition.
        </Text>

        {stats && (
          <View style={styles.statsBox}>
            <Text style={styles.statsTitle}>Per-letter accuracy</Text>
            {Object.entries(stats.perClassAccuracy).map(([letter, val]) => (
              <View key={letter} style={styles.statRow}>
                <Text style={styles.statLetter}>{letter}</Text>
                <View style={styles.statTrack}>
                  <View
                    style={[
                      styles.statFill,
                      {
                        width: `${Math.round(val * 100)}%`,
                        backgroundColor:
                          val > 0.75
                            ? "#2ecc71"
                            : val > 0.5
                              ? "#f39c12"
                              : "#e74c3c",
                      },
                    ]}
                  />
                </View>
                <Text style={styles.statPct}>{Math.round(val * 100)}%</Text>
              </View>
            ))}
          </View>
        )}

        <TouchableOpacity
          style={styles.primaryButton}
          onPress={() => router.push("/")}
          accessibilityRole="button"
        >
          <Text style={styles.primaryButtonText}>Start Using EMG ASL</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={handleReset}
          accessibilityRole="button"
        >
          <Text style={styles.secondaryButtonText}>Clear & Recalibrate</Text>
        </TouchableOpacity>
      </ScrollView>
    );
  };

  // -------------------------------------------------------------------------
  // Root render
  // -------------------------------------------------------------------------

  return (
    <SafeAreaView style={styles.safeArea}>
      {phase !== "done" && (
        <TouchableOpacity
          onPress={() => router.back()}
          style={styles.backButton}
          accessibilityLabel="Go back"
          accessibilityRole="button"
        >
          <Ionicons name="chevron-back" size={24} color="#a0a0b0" />
        </TouchableOpacity>
      )}

      {phase === "idle" && renderIdle()}
      {phase === "countdown" && renderCountdown()}
      {phase === "calibrating" && renderCalibrating()}
      {phase === "done" && renderDone()}
    </SafeAreaView>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#0f0f1a",
  },
  backButton: {
    padding: 12,
    paddingLeft: 16,
    alignSelf: "flex-start",
  },

  // Idle
  idleContainer: {
    alignItems: "center",
    padding: 28,
    gap: 16,
  },
  idleTitle: {
    fontSize: 26,
    fontWeight: "800",
    color: "#ffffff",
    textAlign: "center",
  },
  idleBody: {
    fontSize: 15,
    color: "#a0a0b0",
    textAlign: "center",
    lineHeight: 24,
  },
  input: {
    width: "100%",
    backgroundColor: "#1e1e2e",
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
    color: "#ffffff",
    fontSize: 16,
    borderWidth: 1,
    borderColor: "#2a2a3e",
  },
  warningBox: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    backgroundColor: "#2a1e0e",
    borderRadius: 10,
    padding: 12,
    width: "100%",
  },
  warningText: {
    color: "#f39c12",
    fontSize: 13,
    flex: 1,
  },

  // Countdown
  centeredOverlay: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 16,
  },
  countdownLabel: {
    fontSize: 20,
    color: "#a0a0b0",
    fontWeight: "500",
  },
  countdownNumber: {
    fontSize: 120,
    fontWeight: "900",
    color: "#6c5ce7",
    lineHeight: 130,
  },
  countdownSign: {
    fontSize: 22,
    color: "#ffffff",
    fontWeight: "600",
  },

  // Done
  doneContainer: {
    alignItems: "center",
    padding: 28,
    gap: 16,
  },
  doneTitle: {
    fontSize: 26,
    fontWeight: "800",
    color: "#ffffff",
  },
  doneAccuracy: {
    fontSize: 40,
    fontWeight: "900",
    color: "#2ecc71",
  },
  doneBody: {
    fontSize: 15,
    color: "#a0a0b0",
    textAlign: "center",
    lineHeight: 24,
  },
  statsBox: {
    width: "100%",
    backgroundColor: "#1e1e2e",
    borderRadius: 12,
    padding: 16,
    gap: 8,
  },
  statsTitle: {
    color: "#a0a0b0",
    fontSize: 12,
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: 0.8,
    marginBottom: 4,
  },
  statRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  statLetter: {
    color: "#ffffff",
    fontWeight: "700",
    width: 20,
    fontSize: 13,
  },
  statTrack: {
    flex: 1,
    height: 6,
    backgroundColor: "#2a2a3e",
    borderRadius: 3,
    overflow: "hidden",
  },
  statFill: {
    height: "100%",
    borderRadius: 3,
  },
  statPct: {
    color: "#a0a0b0",
    fontSize: 11,
    width: 34,
    textAlign: "right",
  },

  // Buttons
  primaryButton: {
    width: "100%",
    paddingVertical: 18,
    backgroundColor: "#6c5ce7",
    borderRadius: 16,
    alignItems: "center",
    shadowColor: "#6c5ce7",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.5,
    shadowRadius: 8,
    elevation: 6,
  },
  primaryButtonText: {
    color: "#ffffff",
    fontSize: 17,
    fontWeight: "700",
  },
  buttonDisabled: {
    opacity: 0.4,
  },
  secondaryButton: {
    width: "100%",
    paddingVertical: 14,
    alignItems: "center",
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#2a2a3e",
  },
  secondaryButtonText: {
    color: "#a0a0b0",
    fontSize: 15,
    fontWeight: "500",
  },
});
