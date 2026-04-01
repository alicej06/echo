/**
 * useCalibration.ts
 * React hook that drives a guided ASL calibration session.
 *
 * Flow:
 *   1. startSession(userId)          — initialises the session on the server
 *   2. recordSample(label)           — captures the live EMG window and POSTs it
 *   3. finishCalibration()           — tells the server to train & returns stats
 *
 * The hook cycles through all 26 ASL letters, collecting REPS_PER_LETTER
 * samples each (default 5), so the user makes 130 captures total.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";

import { EMGStreamProcessor } from "../bluetooth/EMGStream";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ASL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const REPS_PER_LETTER = 5;
const TOTAL_SAMPLES = ASL_ALPHABET.length * REPS_PER_LETTER; // 130

const KEY_CALIBRATION_DATA = "calibration:data";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CalibrationStats {
  accuracy: number;
  perClassAccuracy: Record<string, number>;
  totalSamples: number;
}

export interface CalibrationState {
  isCalibrating: boolean;
  currentPrompt: string | null;
  samplesRecorded: number;
  progress: number; // 0–1
  startSession: (userId: string) => Promise<void>;
  recordSample: (label: string) => Promise<void>;
  finishCalibration: () => Promise<CalibrationStats | null>;
  resetCalibration: () => Promise<void>;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useCalibration(
  processorRef: React.MutableRefObject<EMGStreamProcessor | null>,
  serverUrl: string,
): CalibrationState {
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [samplesRecorded, setSamplesRecorded] = useState(0);
  const [currentPromptIndex, setCurrentPromptIndex] = useState(0);

  const userIdRef = useRef<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // -------------------------------------------------------------------------
  // Derived state
  // -------------------------------------------------------------------------

  const letterIndex = Math.floor(currentPromptIndex / REPS_PER_LETTER);
  const repNumber = (currentPromptIndex % REPS_PER_LETTER) + 1;
  const currentPrompt = isCalibrating
    ? (ASL_ALPHABET[letterIndex] ?? null)
    : null;
  const progress = TOTAL_SAMPLES > 0 ? samplesRecorded / TOTAL_SAMPLES : 0;

  // -------------------------------------------------------------------------
  // startSession
  // -------------------------------------------------------------------------

  const startSession = useCallback(
    async (userId: string): Promise<void> => {
      userIdRef.current = userId;
      setSamplesRecorded(0);
      setCurrentPromptIndex(0);

      // Create a calibration session on the server
      const response = await fetch(`${serverUrl}/calibrate/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId }),
      });

      if (!response.ok) {
        throw new Error(
          `Server rejected calibration start: ${response.status}`,
        );
      }

      const data = (await response.json()) as Record<string, unknown>;
      sessionIdRef.current = (data.session_id as string | undefined) ?? null;

      if (mountedRef.current) {
        setIsCalibrating(true);
      }
    },
    [serverUrl],
  );

  // -------------------------------------------------------------------------
  // recordSample
  // -------------------------------------------------------------------------

  const recordSample = useCallback(
    async (label: string): Promise<void> => {
      if (!isCalibrating) return;

      const window = processorRef.current?.getWindow();
      if (!window) {
        throw new Error(
          "No EMG window available — ensure the armband is connected.",
        );
      }

      const payload = {
        session_id: sessionIdRef.current,
        user_id: userIdRef.current,
        label,
        window: Array.from(window),
        shape: [40, 8],
        timestamp: Date.now(),
      };

      const response = await fetch(`${serverUrl}/calibrate/sample`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Failed to save sample: ${response.status}`);
      }

      const nextIndex = currentPromptIndex + 1;

      if (mountedRef.current) {
        setSamplesRecorded((n) => n + 1);
        setCurrentPromptIndex(nextIndex);
      }
    },
    [isCalibrating, currentPromptIndex, serverUrl, processorRef],
  );

  // -------------------------------------------------------------------------
  // finishCalibration
  // -------------------------------------------------------------------------

  const finishCalibration =
    useCallback(async (): Promise<CalibrationStats | null> => {
      if (!sessionIdRef.current) return null;

      const response = await fetch(`${serverUrl}/calibrate/finish`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionIdRef.current,
          user_id: userIdRef.current,
        }),
      });

      if (!response.ok) {
        throw new Error(`Calibration finish failed: ${response.status}`);
      }

      const stats = (await response.json()) as CalibrationStats;

      // Persist the calibration metadata locally
      await AsyncStorage.setItem(
        KEY_CALIBRATION_DATA,
        JSON.stringify({
          userId: userIdRef.current,
          sessionId: sessionIdRef.current,
          completedAt: new Date().toISOString(),
          stats,
        }),
      );

      if (mountedRef.current) {
        setIsCalibrating(false);
      }

      return stats;
    }, [serverUrl]);

  // -------------------------------------------------------------------------
  // resetCalibration
  // -------------------------------------------------------------------------

  const resetCalibration = useCallback(async (): Promise<void> => {
    await AsyncStorage.removeItem(KEY_CALIBRATION_DATA);
    sessionIdRef.current = null;
    userIdRef.current = null;

    if (mountedRef.current) {
      setIsCalibrating(false);
      setSamplesRecorded(0);
      setCurrentPromptIndex(0);
    }
  }, []);

  // -------------------------------------------------------------------------
  // Return
  // -------------------------------------------------------------------------

  return {
    isCalibrating,
    currentPrompt,
    samplesRecorded,
    progress,
    startSession,
    recordSample,
    finishCalibration,
    resetCalibration,
  };
}

// Expose the rep number and letter index separately for consumers who need them
export { ASL_ALPHABET, REPS_PER_LETTER, TOTAL_SAMPLES };
