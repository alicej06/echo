/**
 * ConnectionStatusBar.tsx
 * Compact status bar that shows the current BLE connection state with an
 * animated indicator dot.  Tapping when disconnected triggers reconnection.
 */

import React, { useEffect, useRef } from 'react';
import {
  Animated,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  Easing,
  AccessibilityRole,
} from 'react-native';
import type { ConnectionStatus } from '../bluetooth/BLEManager';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface ConnectionStatusBarProps {
  status: ConnectionStatus;
  deviceName?: string;
  onPressReconnect?: () => void;
}

// ---------------------------------------------------------------------------
// Colour map
// ---------------------------------------------------------------------------

const STATUS_COLORS: Record<ConnectionStatus, string> = {
  disconnected: '#e74c3c',
  scanning: '#3498db',
  connecting: '#f39c12',
  connected: '#2ecc71',
  error: '#e74c3c',
};

const STATUS_LABELS: Record<ConnectionStatus, string> = {
  disconnected: 'Disconnected',
  scanning: 'Scanning…',
  connecting: 'Connecting…',
  connected: 'Connected',
  error: 'Connection Error',
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const ConnectionStatusBar: React.FC<ConnectionStatusBarProps> = ({
  status,
  deviceName,
  onPressReconnect,
}) => {
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const pulseLoop = useRef<Animated.CompositeAnimation | null>(null);

  // Pulse animation for scanning state
  useEffect(() => {
    if (status === 'scanning' || status === 'connecting') {
      pulseLoop.current = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 0.3,
            duration: 700,
            easing: Easing.inOut(Easing.ease),
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 700,
            easing: Easing.inOut(Easing.ease),
            useNativeDriver: true,
          }),
        ]),
      );
      pulseLoop.current.start();
    } else {
      pulseLoop.current?.stop();
      pulseLoop.current = null;
      Animated.timing(pulseAnim, {
        toValue: 1,
        duration: 200,
        useNativeDriver: true,
      }).start();
    }

    return () => {
      pulseLoop.current?.stop();
    };
  }, [status, pulseAnim]);

  const dotColor = STATUS_COLORS[status];
  const label = STATUS_LABELS[status];
  const canReconnect = status === 'disconnected' || status === 'error';
  const subtitle = deviceName ? `Device: ${deviceName}` : undefined;

  const accessibilityHint = canReconnect ? 'Tap to reconnect' : undefined;
  const accessibilityRole: AccessibilityRole = canReconnect ? 'button' : 'text';

  return (
    <TouchableOpacity
      onPress={canReconnect ? onPressReconnect : undefined}
      disabled={!canReconnect}
      activeOpacity={canReconnect ? 0.7 : 1}
      accessibilityRole={accessibilityRole}
      accessibilityLabel={`BLE status: ${label}`}
      accessibilityHint={accessibilityHint}
      style={styles.container}
    >
      {/* Animated dot */}
      <Animated.View
        style={[
          styles.dot,
          { backgroundColor: dotColor, opacity: pulseAnim },
        ]}
      />

      {/* Text section */}
      <View style={styles.textContainer}>
        <Text style={styles.statusText}>{label}</Text>
        {subtitle ? (
          <Text style={styles.subtitleText}>{subtitle}</Text>
        ) : null}
      </View>

      {/* Reconnect hint */}
      {canReconnect && (
        <Text style={styles.reconnectHint}>Tap to reconnect</Text>
      )}
    </TouchableOpacity>
  );
};

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1e1e2e',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 12,
    marginHorizontal: 16,
    marginTop: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 4,
  },
  dot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 10,
  },
  textContainer: {
    flex: 1,
  },
  statusText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
  },
  subtitleText: {
    color: '#a0a0b0',
    fontSize: 11,
    marginTop: 1,
  },
  reconnectHint: {
    color: '#7f8c8d',
    fontSize: 11,
    fontStyle: 'italic',
  },
});
