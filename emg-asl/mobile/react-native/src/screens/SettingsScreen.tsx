/**
 * SettingsScreen.tsx
 * Configuration for server URL, TTS rate, device name, and data management.
 */

import React, { useCallback, useEffect, useReducer } from "react";
import {
  Alert,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Slider from "@react-native-community/slider";

import { useEMGConnection } from "../hooks/useEMGConnection";
import { speechEngine } from "../tts/SpeechEngine";

// ---------------------------------------------------------------------------
// Storage keys (must match those used elsewhere)
// ---------------------------------------------------------------------------

const KEY_SERVER_URL = "settings:serverUrl";
const KEY_DEVICE_NAME = "settings:deviceName";
const KEY_TTS_RATE = "settings:ttsRate";
const KEY_AUTO_SPEAK = "settings:autoSpeak";
const KEY_CALIBRATION_DATA = "calibration:data";

const DEFAULT_SERVER_URL = "ws://localhost:8765";
const DEFAULT_DEVICE_NAME = "EMG-Band";
const DEFAULT_TTS_RATE = 1.0;

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface SettingRowProps {
  label: string;
  children: React.ReactNode;
}

const SettingRow: React.FC<SettingRowProps> = ({ label, children }) => (
  <View style={styles.settingRow}>
    <Text style={styles.settingLabel}>{label}</Text>
    <View style={styles.settingControl}>{children}</View>
  </View>
);

interface SectionHeaderProps {
  title: string;
}

const SectionHeader: React.FC<SectionHeaderProps> = ({ title }) => (
  <Text style={styles.sectionHeader}>{title}</Text>
);

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

type State = {
  serverUrl: string;
  deviceName: string;
  ttsRate: number;
  autoSpeak: boolean;
  saving: boolean;
  hasCalibration: boolean;
};

type Action =
  | {
      type: "LOAD";
      serverUrl: string;
      deviceName: string;
      ttsRate: number;
      autoSpeak: boolean;
      hasCalibration: boolean;
    }
  | { type: "SET_SERVER_URL"; value: string }
  | { type: "SET_DEVICE_NAME"; value: string }
  | { type: "SET_TTS_RATE"; value: number }
  | { type: "SET_AUTO_SPEAK"; value: boolean }
  | { type: "SET_SAVING"; value: boolean }
  | { type: "SET_HAS_CALIBRATION"; value: boolean }
  | { type: "RESET" };

const initialState: State = {
  serverUrl: DEFAULT_SERVER_URL,
  deviceName: DEFAULT_DEVICE_NAME,
  ttsRate: DEFAULT_TTS_RATE,
  autoSpeak: true,
  saving: false,
  hasCalibration: false,
};

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "LOAD":
      return {
        ...state,
        serverUrl: action.serverUrl,
        deviceName: action.deviceName,
        ttsRate: action.ttsRate,
        autoSpeak: action.autoSpeak,
        hasCalibration: action.hasCalibration,
      };
    case "SET_SERVER_URL":
      return { ...state, serverUrl: action.value };
    case "SET_DEVICE_NAME":
      return { ...state, deviceName: action.value };
    case "SET_TTS_RATE":
      return { ...state, ttsRate: action.value };
    case "SET_AUTO_SPEAK":
      return { ...state, autoSpeak: action.value };
    case "SET_SAVING":
      return { ...state, saving: action.value };
    case "SET_HAS_CALIBRATION":
      return { ...state, hasCalibration: action.value };
    case "RESET":
      return { ...initialState };
    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function SettingsScreen() {
  const router = useRouter();
  const { setServerUrl, setDeviceName } = useEMGConnection();
  const [state, dispatch] = useReducer(reducer, initialState);
  const { serverUrl, deviceName, ttsRate, autoSpeak, saving, hasCalibration } =
    state;

  // Load persisted settings — single dispatch replaces 5 cascading setStates
  useEffect(() => {
    (async () => {
      const [storedUrl, storedName, storedRate, storedAutoSpeak, calData] =
        await Promise.all([
          AsyncStorage.getItem(KEY_SERVER_URL),
          AsyncStorage.getItem(KEY_DEVICE_NAME),
          AsyncStorage.getItem(KEY_TTS_RATE),
          AsyncStorage.getItem(KEY_AUTO_SPEAK),
          AsyncStorage.getItem(KEY_CALIBRATION_DATA),
        ]);

      dispatch({
        type: "LOAD",
        serverUrl: storedUrl ?? DEFAULT_SERVER_URL,
        deviceName: storedName ?? DEFAULT_DEVICE_NAME,
        ttsRate: storedRate ? parseFloat(storedRate) : DEFAULT_TTS_RATE,
        autoSpeak: storedAutoSpeak !== null ? storedAutoSpeak === "true" : true,
        hasCalibration: !!calData,
      });
    })();
  }, []);

  // -------------------------------------------------------------------------
  // Save
  // -------------------------------------------------------------------------

  const handleSave = useCallback(async () => {
    if (!serverUrl.trim()) {
      Alert.alert("Invalid URL", "Server URL cannot be empty.");
      return;
    }

    dispatch({ type: "SET_SAVING", value: true });
    try {
      await Promise.all([
        AsyncStorage.setItem(KEY_SERVER_URL, serverUrl.trim()),
        AsyncStorage.setItem(KEY_DEVICE_NAME, deviceName.trim()),
        AsyncStorage.setItem(KEY_TTS_RATE, String(ttsRate)),
        AsyncStorage.setItem(KEY_AUTO_SPEAK, String(autoSpeak)),
      ]);

      // Propagate to hook singletons
      await Promise.all([
        setServerUrl(serverUrl.trim()),
        setDeviceName(deviceName.trim()),
      ]);
      speechEngine.setRate(ttsRate);

      Alert.alert("Saved", "Settings have been saved successfully.");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      Alert.alert("Error", `Failed to save settings: ${msg}`);
    } finally {
      dispatch({ type: "SET_SAVING", value: false });
    }
  }, [serverUrl, deviceName, ttsRate, autoSpeak, setServerUrl, setDeviceName]);

  // -------------------------------------------------------------------------
  // Clear calibration
  // -------------------------------------------------------------------------

  const handleClearCalibration = useCallback(() => {
    Alert.alert(
      "Clear Calibration Data",
      "This will permanently delete your personal EMG calibration model. You will need to recalibrate before using sign recognition.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          style: "destructive",
          onPress: async () => {
            await AsyncStorage.removeItem(KEY_CALIBRATION_DATA);
            dispatch({ type: "SET_HAS_CALIBRATION", value: false });
            Alert.alert("Cleared", "Calibration data has been deleted.");
          },
        },
      ],
    );
  }, []);

  // -------------------------------------------------------------------------
  // Clear all data
  // -------------------------------------------------------------------------

  const handleClearAll = useCallback(() => {
    Alert.alert(
      "Reset All Data",
      "This will delete all settings and calibration data and cannot be undone.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Reset Everything",
          style: "destructive",
          onPress: async () => {
            await AsyncStorage.clear();
            dispatch({ type: "RESET" });
            Alert.alert("Reset", "All data cleared. App is back to defaults.");
          },
        },
      ],
    );
  }, []);

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  const rateLabel = ttsRate.toFixed(1) + "×";

  return (
    <SafeAreaView style={styles.safeArea}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          onPress={() => router.back()}
          style={styles.backButton}
          accessibilityLabel="Go back"
          accessibilityRole="button"
        >
          <Ionicons name="chevron-back" size={24} color="#a0a0b0" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Settings</Text>
        <TouchableOpacity
          onPress={handleSave}
          disabled={saving}
          style={[styles.saveButton, saving && styles.saveButtonDisabled]}
          accessibilityLabel="Save settings"
          accessibilityRole="button"
        >
          <Text style={styles.saveButtonText}>
            {saving ? "Saving…" : "Save"}
          </Text>
        </TouchableOpacity>
      </View>

      <ScrollView
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        {/* -------------------------------------------------------------- */}
        {/* Connection                                                      */}
        {/* -------------------------------------------------------------- */}
        <SectionHeader title="Connection" />

        <View style={styles.card}>
          <SettingRow label="Server URL">
            <TextInput
              style={styles.textInput}
              value={serverUrl}
              onChangeText={(v) =>
                dispatch({ type: "SET_SERVER_URL", value: v })
              }
              placeholder={DEFAULT_SERVER_URL}
              placeholderTextColor="#555577"
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
              returnKeyType="done"
              accessibilityLabel="Inference server URL"
            />
          </SettingRow>

          <View style={styles.divider} />

          <SettingRow label="Device Name">
            <TextInput
              style={styles.textInput}
              value={deviceName}
              onChangeText={(v) =>
                dispatch({ type: "SET_DEVICE_NAME", value: v })
              }
              placeholder={DEFAULT_DEVICE_NAME}
              placeholderTextColor="#555577"
              autoCapitalize="none"
              autoCorrect={false}
              returnKeyType="done"
              accessibilityLabel="Bluetooth device name to scan for"
            />
          </SettingRow>
        </View>

        {/* -------------------------------------------------------------- */}
        {/* Text-to-Speech                                                  */}
        {/* -------------------------------------------------------------- */}
        <SectionHeader title="Text-to-Speech" />

        <View style={styles.card}>
          <SettingRow label={`Speech Rate  ${rateLabel}`}>
            <Slider
              style={styles.slider}
              minimumValue={0.5}
              maximumValue={2.0}
              step={0.1}
              value={ttsRate}
              onValueChange={(val: number) => {
                dispatch({
                  type: "SET_TTS_RATE",
                  value: parseFloat(val.toFixed(1)),
                });
              }}
              onSlidingComplete={(val: number) => {
                const rounded = parseFloat(val.toFixed(1));
                dispatch({ type: "SET_TTS_RATE", value: rounded });
                speechEngine.setRate(rounded);
              }}
              minimumTrackTintColor="#6c5ce7"
              maximumTrackTintColor="#2a2a3e"
              thumbTintColor="#6c5ce7"
              accessibilityLabel="Speech rate"
            />
          </SettingRow>

          <View style={styles.divider} />

          <SettingRow label="Auto-speak predictions">
            <Switch
              value={autoSpeak}
              onValueChange={(v) =>
                dispatch({ type: "SET_AUTO_SPEAK", value: v })
              }
              trackColor={{ false: "#2a2a3e", true: "#6c5ce7" }}
              thumbColor={autoSpeak ? "#ffffff" : "#555577"}
              accessibilityLabel="Auto-speak new sign predictions"
            />
          </SettingRow>
        </View>

        {/* -------------------------------------------------------------- */}
        {/* Data Management                                                 */}
        {/* -------------------------------------------------------------- */}
        <SectionHeader title="Data Management" />

        <View style={styles.card}>
          <TouchableOpacity
            onPress={handleClearCalibration}
            disabled={!hasCalibration}
            style={[
              styles.dangerButton,
              !hasCalibration && styles.dangerButtonDisabled,
            ]}
            accessibilityRole="button"
            accessibilityLabel="Clear calibration data"
          >
            <Ionicons
              name="trash-outline"
              size={18}
              color={hasCalibration ? "#e74c3c" : "#555577"}
            />
            <Text
              style={[
                styles.dangerButtonText,
                !hasCalibration && styles.dangerButtonTextDisabled,
              ]}
            >
              {hasCalibration
                ? "Clear Calibration Data"
                : "No Calibration Data Saved"}
            </Text>
          </TouchableOpacity>

          <View style={styles.divider} />

          <TouchableOpacity
            onPress={handleClearAll}
            style={styles.dangerButton}
            accessibilityRole="button"
            accessibilityLabel="Reset all app data"
          >
            <Ionicons name="nuclear-outline" size={18} color="#e74c3c" />
            <Text style={styles.dangerButtonText}>Reset All App Data</Text>
          </TouchableOpacity>
        </View>

        {/* -------------------------------------------------------------- */}
        {/* About                                                           */}
        {/* -------------------------------------------------------------- */}
        <SectionHeader title="About" />
        <View style={styles.card}>
          <View style={styles.aboutRow}>
            <Text style={styles.aboutLabel}>App</Text>
            <Text style={styles.aboutValue}>EMG ASL</Text>
          </View>
          <View style={styles.divider} />
          <View style={styles.aboutRow}>
            <Text style={styles.aboutLabel}>Version</Text>
            <Text style={styles.aboutValue}>1.0.0</Text>
          </View>
          <View style={styles.divider} />
          <View style={styles.aboutRow}>
            <Text style={styles.aboutLabel}>Organisation</Text>
            <Text style={styles.aboutValue}>MAIA Biotech</Text>
          </View>
        </View>

        <View style={styles.bottomSpacer} />
      </ScrollView>
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
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 12,
    paddingTop: 8,
    paddingBottom: 8,
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: "#ffffff",
  },
  saveButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: "#6c5ce7",
    borderRadius: 10,
  },
  saveButtonDisabled: {
    opacity: 0.5,
  },
  saveButtonText: {
    color: "#ffffff",
    fontWeight: "600",
    fontSize: 14,
  },
  scrollContent: {
    paddingHorizontal: 16,
    paddingTop: 8,
    gap: 4,
  },
  sectionHeader: {
    color: "#6c5ce7",
    fontSize: 11,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 1.2,
    marginTop: 20,
    marginBottom: 6,
    paddingLeft: 4,
  },
  card: {
    backgroundColor: "#1e1e2e",
    borderRadius: 14,
    overflow: "hidden",
  },
  divider: {
    height: 1,
    backgroundColor: "#2a2a3e",
    marginHorizontal: 16,
  },
  settingRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 14,
    gap: 12,
  },
  settingLabel: {
    color: "#c0c0d0",
    fontSize: 15,
    flex: 1,
  },
  settingControl: {
    flex: 1,
    alignItems: "flex-end",
  },
  textInput: {
    backgroundColor: "#2a2a3e",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    color: "#ffffff",
    fontSize: 14,
    width: "100%",
    textAlign: "right",
  },
  slider: {
    width: "100%",
    height: 32,
  },
  dangerButton: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingHorizontal: 16,
    paddingVertical: 16,
  },
  dangerButtonDisabled: {
    opacity: 0.5,
  },
  dangerButtonText: {
    color: "#e74c3c",
    fontSize: 15,
    fontWeight: "500",
  },
  dangerButtonTextDisabled: {
    color: "#555577",
  },
  aboutRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingVertical: 14,
  },
  aboutLabel: {
    color: "#a0a0b0",
    fontSize: 15,
  },
  aboutValue: {
    color: "#ffffff",
    fontSize: 15,
    fontWeight: "500",
  },
  bottomSpacer: {
    height: 48,
  },
});
