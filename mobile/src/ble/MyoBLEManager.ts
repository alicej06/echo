/**
 * MyoBLEManager — connects to a Myo Armband via BLE and streams raw EMG data.
 *
 * Protocol reference: thalmiclabs/myo-bluetooth-protocol
 *
 * Flow:
 *   1. scan() — scan for device named "Myo" or "myomote"
 *   2. connect() — connect and discover services
 *   3. enableEMG() — write set_mode command to control characteristic
 *   4. subscribe() — subscribe to all 4 EMG characteristics (50Hz each × 2 samples = 200Hz total)
 *   5. onData callback fires per notification → caller feeds EMGWindowBuffer
 */

import {
  BleManager,
  Device,
  State,
  Characteristic,
} from "react-native-ble-plx";
import { Platform, PermissionsAndroid } from "react-native";
import { Buffer } from "buffer";

// Myo BLE UUIDs
const MYO_SERVICE_UUID = "d5060001-a904-deb9-4748-2c7f4a124842";
const MYO_CONTROL_UUID = "d5060401-a904-deb9-4748-2c7f4a124842";
const MYO_EMG_CHAR_UUIDS = [
  "d5060105-a904-deb9-4748-2c7f4a124842",
  "d5060205-a904-deb9-4748-2c7f4a124842",
  "d5060305-a904-deb9-4748-2c7f4a124842",
  "d5060405-a904-deb9-4748-2c7f4a124842",
];

// set_mode command: raw EMG, IMU off, classifier off
const ENABLE_EMG_CMD = Buffer.from([0x01, 0x03, 0x02, 0x00, 0x00]).toString(
  "base64",
);

export type EMGDataCallback = (data: Uint8Array) => void;
export type StatusCallback = (status: MyoStatus) => void;

export type MyoStatus =
  | "idle"
  | "scanning"
  | "connecting"
  | "connected"
  | "streaming"
  | "disconnected"
  | "error";

export class MyoBLEManager {
  private manager: BleManager;
  private device: Device | null = null;
  private subscriptions: Array<{ remove: () => void }> = [];
  private onEMGData: EMGDataCallback;
  private onStatus: StatusCallback;

  constructor(onEMGData: EMGDataCallback, onStatus: StatusCallback) {
    this.manager = new BleManager();
    this.onEMGData = onEMGData;
    this.onStatus = onStatus;
  }

  /** Request Android BLE permissions (iOS permissions are declared in Info.plist). */
  async requestPermissions(): Promise<boolean> {
    if (Platform.OS !== "android") return true;
    const granted = await PermissionsAndroid.requestMultiple([
      PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN,
      PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT,
      PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
    ]);
    return Object.values(granted).every(
      (r) => r === PermissionsAndroid.RESULTS.GRANTED,
    );
  }

  /** Wait for BLE adapter to be powered on. */
  private waitForPoweredOn(): Promise<void> {
    return new Promise((resolve, reject) => {
      const sub = this.manager.onStateChange((state) => {
        if (state === State.PoweredOn) {
          sub.remove();
          resolve();
        }
        if (state === State.PoweredOff || state === State.Unsupported) {
          sub.remove();
          reject(new Error(`BLE state: ${state}`));
        }
      }, true);
    });
  }

  /** Scan for a Myo device and connect to the first one found. */
  async connect(timeoutMs = 15000): Promise<void> {
    await this.requestPermissions();
    await this.waitForPoweredOn();

    this.onStatus("scanning");

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.manager.stopDeviceScan();
        this.onStatus("error");
        reject(new Error("Scan timeout — make sure Myo is awake (wave arm)"));
      }, timeoutMs);

      this.manager.startDeviceScan(
        null,
        { allowDuplicates: false },
        async (err, device) => {
          if (err) {
            clearTimeout(timer);
            this.onStatus("error");
            reject(err);
            return;
          }
          if (!device) return;

          const name = device.name?.toLowerCase() ?? "";
          if (!name.includes("myo") && !name.includes("myomote")) return;

          this.manager.stopDeviceScan();
          clearTimeout(timer);
          this.onStatus("connecting");

          try {
            this.device = await device.connect({ autoConnect: false });
            await this.device.discoverAllServicesAndCharacteristics();
            this.onStatus("connected");

            // Watch for disconnection
            this.device.onDisconnected(() => {
              this.onStatus("disconnected");
              this.cleanup();
            });

            await this.enableEMGStreaming();
            await this.subscribeToEMG();
            this.onStatus("streaming");
            resolve();
          } catch (e) {
            this.onStatus("error");
            reject(e);
          }
        },
      );
    });
  }

  /** Write the set_mode command to enable raw EMG streaming. */
  private async enableEMGStreaming(): Promise<void> {
    if (!this.device) throw new Error("No device");
    await this.device.writeCharacteristicWithResponseForService(
      MYO_SERVICE_UUID,
      MYO_CONTROL_UUID,
      ENABLE_EMG_CMD,
    );
  }

  /** Subscribe to all 4 EMG characteristics. Each fires at ~50Hz with 2 samples. */
  private async subscribeToEMG(): Promise<void> {
    if (!this.device) throw new Error("No device");

    for (const charUUID of MYO_EMG_CHAR_UUIDS) {
      const sub = this.device.monitorCharacteristicForService(
        MYO_SERVICE_UUID,
        charUUID,
        (err: Error | null, char: Characteristic | null) => {
          if (err || !char?.value) return;
          const bytes = new Uint8Array(Buffer.from(char.value, "base64"));
          this.onEMGData(bytes);
        },
      );
      this.subscriptions.push(sub);
    }
  }

  /** Disconnect and clean up all subscriptions. */
  async disconnect(): Promise<void> {
    this.cleanup();
    await this.device?.cancelConnection();
    this.device = null;
    this.onStatus("idle");
  }

  private cleanup(): void {
    for (const sub of this.subscriptions) sub.remove();
    this.subscriptions = [];
  }

  destroy(): void {
    this.cleanup();
    this.manager.destroy();
  }
}
