#!/usr/bin/env python3
"""
tools/rtsp_simulator.py — Local RTSP Video Stream Simulator
Streams an MP4 video or test pattern in a loop over RTSP to MediaMTX.
Useful for testing/evaluating without physical IP cameras.
"""
import argparse
import subprocess
import sys
import shutil

def run_ffmpeg_rtsp_stream(video_path: str, rtsp_url: str, fps: int = 30):
    """Streams a video file repeatedly to an RTSP server using ffmpeg."""
    if not shutil.which("ffmpeg"):
        print("[ERROR] ffmpeg is not installed on your system. Please install ffmpeg or use MediaMTX file publisher.")
        sys.exit(1)

    cmd = [
        "ffmpeg",
        "-re",
        "-stream_loop", "-1",
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-b:v", "2M",
        "-r", str(fps),
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_url
    ]

    print(f"[INFO] Streaming {video_path} -> {rtsp_url} (looping)...")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped RTSP stream.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate an RTSP camera stream from an MP4 file")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input video file (e.g. sample.mp4)")
    parser.add_argument("--url", "-u", type=str, default="rtsp://localhost:8554/cam0", help="Target RTSP URL")
    parser.add_argument("--fps", type=int, default=30, help="Target stream FPS")
    args = parser.parse_args()

    run_ffmpeg_rtsp_stream(args.input, args.url, args.fps)
