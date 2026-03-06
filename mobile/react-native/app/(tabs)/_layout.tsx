/**
 * app/(tabs)/_layout.tsx
 * Tab navigator with Home, Calibration, and Settings tabs.
 */

import React from 'react';
import { Tabs } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { Platform } from 'react-native';

const TAB_BAR_BG = '#1e1e2e';
const TAB_ACTIVE = '#6c5ce7';
const TAB_INACTIVE = '#555577';

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: TAB_BAR_BG,
          borderTopColor: '#2a2a3e',
          borderTopWidth: 1,
          paddingBottom: Platform.OS === 'ios' ? 20 : 6,
          paddingTop: 6,
          height: Platform.OS === 'ios' ? 80 : 60,
        },
        tabBarActiveTintColor: TAB_ACTIVE,
        tabBarInactiveTintColor: TAB_INACTIVE,
        tabBarLabelStyle: {
          fontSize: 11,
          fontWeight: '600',
        },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'Home',
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="home-outline" color={color} size={size} />
          ),
        }}
      />
      <Tabs.Screen
        name="calibration"
        options={{
          title: 'Calibrate',
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="hand-left-outline" color={color} size={size} />
          ),
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: 'Settings',
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="settings-outline" color={color} size={size} />
          ),
        }}
      />
    </Tabs>
  );
}
