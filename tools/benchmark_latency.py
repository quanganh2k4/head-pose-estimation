#!/usr/bin/env python3
"""
tools/benchmark_latency.py — Pipeline Latency & Throughput Benchmark
Measures end-to-end latency from MQTT message timestamp to calculation output.
"""
import time
import json
import argparse
from collections import deque
import paho.mqtt.client as mqtt

class BenchmarkClient:
    def __init__(self, broker: str = "127.0.0.1", port: int = 1883, topic: str = "gaze/+/metadata"):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.latencies = deque(maxlen=1000)
        self.message_count = 0
        self.start_time = None

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"[INFO] Connected to MQTT Broker {self.broker}:{self.port}")
            client.subscribe(self.topic)
            print(f"[INFO] Subscribed to {self.topic}")
        else:
            print(f"[ERROR] Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        now = time.time()
        if self.start_time is None:
            self.start_time = now

        self.message_count += 1
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            ts = payload.get("timestamp") or payload.get("ts")
            if ts:
                # Calculate latency if timestamp is embedded
                lat_ms = (now - float(ts)) * 1000.0
                if 0 < lat_ms < 5000:
                    self.latencies.append(lat_ms)
        except Exception:
            pass

        if self.message_count % 100 == 0:
            elapsed = now - self.start_time
            fps = self.message_count / elapsed if elapsed > 0 else 0
            avg_lat = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
            print(f"[STATS] Msgs: {self.message_count} | Throughput: {fps:.1f} msg/s | Avg Latency: {avg_lat:.2f} ms")

    def run(self):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(self.broker, self.port, 60)
        print("[INFO] Starting benchmark. Press Ctrl+C to exit.")
        try:
            client.loop_forever()
        except KeyboardInterrupt:
            print("\n[INFO] Benchmark stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MQTT pipeline latency")
    parser.add_argument("--broker", default="127.0.0.1", help="MQTT Broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT Broker port")
    parser.add_argument("--topic", default="gaze/+/metadata", help="Topic to benchmark")
    args = parser.parse_args()

    bench = BenchmarkClient(broker=args.broker, port=args.port, topic=args.topic)
    bench.run()
