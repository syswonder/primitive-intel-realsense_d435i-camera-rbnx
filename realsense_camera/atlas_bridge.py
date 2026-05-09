#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""realsense_camera_rbnx — atlas bridge (driver-init lifecycle).

Same shape as mid360_lidar_rbnx: start.sh runs THIS process; we register
the cap + declare ONLY `primitive/camera/driver` on atlas; `rbnx boot`
calls `Driver(CMD_INIT, config_json)` and only THEN do we:
  1. parse config (camera_name, resolutions, align_depth, enable_imu),
  2. spawn `ros2 launch realsense2_camera rs_launch.py` with those args,
  3. wait for the first RGB frame on the configured topic,
  4. DeclareInterface for `primitive/camera/{rgb, depth}` on atlas.

The D435i has an internal IMU but we deliberately do NOT register it on
the `primitive/imu/imu` contract — the Ranger Mini's MID-360 IMU is the
canonical IMU for that contract (better noise, co-located with the
lidar for SLAM). Anyone who needs the camera IMU directly can subscribe
to `/<camera_name>/imu`; it's still published, just not atlas-routed.

Config (passed via Driver(CMD_INIT, config_json)):
    camera_name        default "camera_435i"
    rgb_topic          default "/<camera_name>/color/image_raw"
    depth_topic        default "/<camera_name>/aligned_depth_to_color/image_raw"
    rgb_profile        default "640x480x30"
    depth_profile      default "848x480x30"   (sensor native; rs picks closest)
    align_depth        default true           (depth aligned to RGB optical)
    enable_imu         default true           (driver publishes; not atlas-routed)
    enable_sync        default true
    sentinel_timeout_s default 30.0
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent import futures
from pathlib import Path

logging.basicConfig(level=os.environ.get("REALSENSE_LOG_LEVEL", "INFO"),
                    format="[realsense] %(message)s")
log = logging.getLogger("realsense")


def _ensure_proto_gen() -> None:
    d = Path(__file__).resolve().parent
    while d.parent != d:
        pg = d / "rbnx-build" / "codegen" / "proto_gen"
        if pg.is_dir() and (pg / "atlas_pb2.py").exists():
            sys.path.insert(0, str(pg))
            return
        d = d.parent


_ensure_proto_gen()

import grpc  # noqa: E402
import atlas_pb2 as pb  # noqa: E402
import atlas_pb2_grpc as pb_grpc  # noqa: E402
import lifecycle_pb2  # noqa: E402
import robonix_contracts_pb2_grpc as contracts_grpc  # noqa: E402

CMD_INIT = 0
CMD_ACTIVATE = 1
CMD_DEACTIVATE = 2
CMD_SHUTDOWN = 3


# ── shared state populated by Init ───────────────────────────────────────────
_state_lock = threading.Lock()
_atlas_stub: pb_grpc.AtlasStub | None = None
_cap_id: str = ""
_pkg_root: Path = Path(__file__).resolve().parent.parent
_rs_proc: subprocess.Popen | None = None
_initialized = False


def _spawn_realsense(cfg: dict) -> None:
    """Launch ros2 launch realsense2_camera rs_launch.py with config args."""
    global _rs_proc
    cam = cfg.get("camera_name", "camera_435i")
    args = [
        "ros2", "launch", "realsense2_camera", "rs_launch.py",
        f"camera_namespace:=/",
        f"camera_name:={cam}",
        f"enable_imu:={'true' if cfg.get('enable_imu', True) else 'false'}",
        f"enable_gyro:={'true' if cfg.get('enable_imu', True) else 'false'}",
        f"enable_accel:={'true' if cfg.get('enable_imu', True) else 'false'}",
        "unite_imu_method:=2",
        f"align_depth.enable:={'true' if cfg.get('align_depth', True) else 'false'}",
        f"enable_sync:={'true' if cfg.get('enable_sync', True) else 'false'}",
        "publish_tf:=true",  # rtabmap consumes camera_link → optical_frame TFs
        "temporal_filter.enable:=true",
        "hole_filling_filter.enable:=true",
        f"rgb_camera.color_profile:={cfg.get('rgb_profile', '640x480x30')}",
        f"depth_module.depth_profile:={cfg.get('depth_profile', '848x480x30')}",
    ]
    log_path = _pkg_root / "rbnx-build" / "data" / "realsense.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "ab", buffering=0)
    log.info("spawning realsense driver (cam=%s) → %s", cam, log_path)
    log.debug("launch args: %s", " ".join(args))
    _rs_proc = subprocess.Popen(
        args,
        stdout=log_fh, stderr=log_fh,
        start_new_session=True,
    )


def _kill_realsense() -> None:
    p = _rs_proc
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        p.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _wait_for_image(topic: str, timeout_s: float) -> bool:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import Image
    except ImportError as e:
        log.warning("rclpy unavailable (%s); skipping sentinel wait", e)
        return True
    rclpy.init(args=None)
    node = Node("realsense_atlas_sentinel")
    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    seen = threading.Event()
    node.create_subscription(Image, topic, lambda _m: seen.set(), qos)
    log.info("waiting for first frame on %s — up to %.1fs", topic, timeout_s)
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            if seen.is_set():
                break
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass
    return seen.is_set()


def _decl_topic_out(contract_id: str, topic: str, qos_profile: str = "best_effort") -> None:
    if _atlas_stub is None:
        return
    _atlas_stub.DeclareInterface(pb.DeclareInterfaceRequest(
        capability_id=_cap_id,
        contract_id=contract_id,
        transport=pb.TRANSPORT_ROS2,
        endpoint=topic,
        params=pb.TransportParams(ros2=pb.Ros2Params(qos_profile=qos_profile)),
    ))


class _CameraDriverServicer(contracts_grpc.PrimitiveCameraDriverServicer):
    def Driver(self, request, context):
        cmd = int(request.command)
        if cmd == CMD_INIT:
            try:
                cfg = json.loads(request.config_json) if request.config_json else {}
            except json.JSONDecodeError as e:
                return lifecycle_pb2.Driver_Response(
                    ok=False, state="error", error=f"bad config_json: {e}"
                )
            return self._init(cfg)
        if cmd == CMD_ACTIVATE:
            # primitives do all bring-up in CMD_INIT; ACTIVATE
            # is a framework no-op that flips the cap to ACTIVE
            # so consumers may begin calling.
            return lifecycle_pb2.Driver_Response(ok=True, state="active", error="")
        if cmd == CMD_DEACTIVATE:
            # framework no-op back to INACTIVE; v1 doesn't evict.
            return lifecycle_pb2.Driver_Response(ok=True, state="inactive", error="")
        if cmd == CMD_SHUTDOWN:
            _kill_realsense()
            return lifecycle_pb2.Driver_Response(ok=True, state="terminated", error="")
        return lifecycle_pb2.Driver_Response(
            ok=False, state="error", error=f"invalid command {cmd}"
        )

    def _init(self, cfg: dict):
        global _initialized
        with _state_lock:
            if _initialized:
                return lifecycle_pb2.Driver_Response(ok=True, state="inactive", error="")

        cam = cfg.get("camera_name", "camera_435i")
        rgb_topic = cfg.get("rgb_topic", f"/{cam}/color/image_raw")
        depth_topic = cfg.get("depth_topic",
                              f"/{cam}/aligned_depth_to_color/image_raw")
        sentinel_timeout = float(cfg.get("sentinel_timeout_s", 30.0))

        try:
            _spawn_realsense(cfg)
        except Exception as e:  # noqa: BLE001
            return lifecycle_pb2.Driver_Response(
                ok=False, state="error", error=f"spawn realsense failed: {e}"
            )

        if not _wait_for_image(rgb_topic, sentinel_timeout):
            _kill_realsense()
            return lifecycle_pb2.Driver_Response(
                ok=False, state="error",
                error=f"no Image on {rgb_topic} within {sentinel_timeout:.1f}s",
            )

        try:
            _decl_topic_out("robonix/primitive/camera/rgb",   rgb_topic)
            _decl_topic_out("robonix/primitive/camera/depth", depth_topic)
        except grpc.RpcError as e:
            if e.code() != grpc.StatusCode.ALREADY_EXISTS:
                return lifecycle_pb2.Driver_Response(
                    ok=False, state="error", error=f"declare failed: {e.details()}"
                )

        with _state_lock:
            _initialized = True
        log.info("init complete: rgb=%s depth=%s", rgb_topic, depth_topic)
        return lifecycle_pb2.Driver_Response(ok=True, state="inactive", error="")


def _start_driver_grpc(port: int) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    contracts_grpc.add_PrimitiveCameraDriverServicer_to_server(
        _CameraDriverServicer(), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("LifecycleDriver gRPC serving on 0.0.0.0:%d", port)


def _decl_driver_iface(port: int) -> None:
    if _atlas_stub is None:
        return
    _atlas_stub.DeclareInterface(pb.DeclareInterfaceRequest(
        capability_id=_cap_id,
        contract_id="robonix/primitive/camera/driver",
        transport=pb.TRANSPORT_GRPC,
        endpoint=f"127.0.0.1:{port}",
        params=pb.TransportParams(grpc=pb.GrpcParams(
            proto_file="robonix_contracts.proto",
            service_name="PrimitiveCameraDriver",
            method="Driver",
        )),
    ))


def _heartbeat_loop() -> None:
    while True:
        time.sleep(15.0)
        if _atlas_stub is None:
            continue
        try:
            _atlas_stub.Heartbeat(pb.HeartbeatRequest(capability_id=_cap_id))
        except Exception as e:  # noqa: BLE001
            log.debug("heartbeat: %s", e)


def _on_signal(signum, _frame):
    log.info("signal %d — shutting down", signum)
    _kill_realsense()
    sys.exit(0)


def main() -> None:
    global _atlas_stub, _cap_id
    atlas_addr = os.environ.get("ROBONIX_ATLAS", "127.0.0.1:50051")
    driver_port = int(os.environ.get("REALSENSE_DRIVER_PORT", "50232"))
    _cap_id = os.environ.get(
        "ROBONIX_CAPABILITY_ID", "com.robonix.ranger.realsense_camera"
    )

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    _start_driver_grpc(driver_port)

    channel = grpc.insecure_channel(atlas_addr)
    _atlas_stub = pb_grpc.AtlasStub(channel)
    pkg_dir = os.environ.get("ROBONIX_PKG_HOST_DIR", "")
    md_path = f"{pkg_dir}/CAPABILITY.md" if pkg_dir else ""
    try:
        _atlas_stub.RegisterCapability(pb.RegisterCapabilityRequest(
            capability_id=_cap_id,
            namespace="robonix/primitive/camera",
            capability_md_path=md_path,
        ))
        _decl_driver_iface(driver_port)
        log.info("registered cap %s, driver iface on :%d (awaiting INIT)",
                 _cap_id, driver_port)
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.ALREADY_EXISTS:
            log.info("cap %s already registered (re-deploy); ok", _cap_id)
        else:
            log.warning("atlas registration failed: %s", e)

    threading.Thread(target=_heartbeat_loop, daemon=True).start()
    log.info("ready — awaiting Driver(CMD_INIT)")
    try:
        while True:
            time.sleep(60.0)
    except KeyboardInterrupt:
        pass
    finally:
        _kill_realsense()


if __name__ == "__main__":
    main()
