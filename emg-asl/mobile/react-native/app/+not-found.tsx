import { Link, Stack } from 'expo-router';
import { View, Text, StyleSheet } from 'react-native';

export default function NotFoundScreen() {
  return (
    <>
      <Stack.Screen options={{ title: 'Not Found' }} />
      <View style={styles.container}>
        <Text style={styles.text}>Screen not found</Text>
        <Link href="/" style={styles.link}>Go home</Link>
      </View>
    </>
  );
}
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#121212', alignItems: 'center', justifyContent: 'center' },
  text: { color: '#fff', fontSize: 18, marginBottom: 16 },
  link: { color: '#4CAF50', fontSize: 16 },
});
