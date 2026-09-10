#!/usr/bin/env python3
"""Run F54's self-contained page in a local Firefox through Marionette.

No HTTP server or network navigation is used. The output directory must not
exist, preventing a failed run from reusing an older passing result.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time


class Marionette:
    def __init__(self, port: int, timeout: float):
        deadline = time.monotonic() + timeout
        last_error = None
        while time.monotonic() < deadline:
            try:
                self.socket = socket.create_connection(("127.0.0.1", port), 2)
                break
            except OSError as error:
                last_error = error
                time.sleep(0.2)
        else:
            raise TimeoutError(f"Marionette did not listen: {last_error}")
        self.socket.settimeout(timeout)
        self.buffer = bytearray()
        self.next_id = 1
        self.hello = self._read()

    def _read(self):
        while b":" not in self.buffer:
            chunk = self.socket.recv(65536)
            if not chunk:
                raise EOFError("Marionette closed")
            self.buffer += chunk
        size_bytes, _, remainder = self.buffer.partition(b":")
        size = int(size_bytes)
        self.buffer = bytearray(remainder)
        while len(self.buffer) < size:
            chunk = self.socket.recv(65536)
            if not chunk:
                raise EOFError("Marionette closed mid-packet")
            self.buffer += chunk
        payload = bytes(self.buffer[:size])
        self.buffer = self.buffer[size:]
        return json.loads(payload.decode())

    def command(self, name, parameters=None):
        message_id = self.next_id
        self.next_id += 1
        body = json.dumps([0, message_id, name, parameters or {}]).encode()
        self.socket.sendall(str(len(body)).encode() + b":" + body)
        while True:
            packet = self._read()
            if not isinstance(packet, list) or len(packet) != 4:
                continue
            _, reply_id, error, result = packet
            if reply_id != message_id:
                continue
            if error:
                raise RuntimeError(f"{name} failed: {error}")
            return result

    def close(self):
        self.socket.close()


def write_profile(path: Path, port: int):
    preferences = {
        "marionette.port": port,
        "browser.shell.checkDefaultBrowser": False,
        "browser.aboutwelcome.enabled": False,
        "browser.startup.homepage_override.mstone": "ignore",
        "browser.tabs.warnOnClose": False,
        "datareporting.policy.dataSubmissionEnabled": False,
        "toolkit.telemetry.reportingpolicy.firstRun": False,
        "app.update.auto": False,
        "app.update.enabled": False,
        "browser.search.update": False,
        "extensions.update.enabled": False,
        "network.prefetch-next": False,
        "network.dns.disablePrefetch": True,
        "webgl.force-enabled": True,
        "webgl.disable-fail-if-major-performance-caveat": False,
    }
    lines = [f"user_pref({json.dumps(key)}, {json.dumps(value)});"
             for key, value in sorted(preferences.items())]
    (path / "user.js").write_text("\n".join(lines) + "\n")
    return preferences


def hardware_renderer(result):
    renderer = (result.get("webgl2", {}).get("unmaskedRenderer") or
                result.get("webgl2", {}).get("renderer") or "").lower()
    return renderer and not any(token in renderer for token in
                                ("llvmpipe", "softpipe", "swiftshader",
                                 "software rasterizer", "lavapipe"))


def validate(result, require_hardware, require_g17p_zink):
    failures = []
    if result.get("schema") != 1 or not result.get("ok"):
        failures.append(result.get("error") or "page result was not ok")
    if not result.get("webgl1", {}).get("exact"):
        failures.append("WebGL1 exact indexed draw failed")
    draws = result.get("webgl2", {}).get("draws") or []
    if len(draws) != 12 or not all(draw.get("exact") for draw in draws):
        failures.append("WebGL2 12-draw vertex-texture churn failed")
    if require_hardware and not hardware_renderer(result):
        failures.append("renderer was missing or software")
    if require_g17p_zink:
        renderer = (result.get("webgl2", {}).get("unmaskedRenderer") or
                    result.get("webgl2", {}).get("renderer") or "").lower()
        if "zink" not in renderer or not any(token in renderer for token in
                                              ("g17p", "a18 pro", "honeykrisp")):
            failures.append("renderer did not prove Zink on Honeykrisp G17P")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firefox", default=shutil.which("firefox"))
    parser.add_argument("--page", default=str(Path(__file__).with_name(
        "firefox-g17p-offline.html")))
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--port", type=int, default=2828)
    parser.add_argument("--require-hardware", action="store_true")
    parser.add_argument("--require-g17p-zink", action="store_true")
    args = parser.parse_args()

    if not args.firefox or not Path(args.firefox).is_file():
        parser.error("--firefox must name a local Firefox executable")
    page = Path(args.page).resolve()
    if not page.is_file():
        parser.error(f"missing page: {page}")
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(f"refusing existing output: {output}")
    if not 5 <= args.timeout <= 180:
        parser.error("--timeout must be in [5, 180]")
    output.mkdir(parents=True)

    work = Path(tempfile.mkdtemp(prefix="f54-firefox-"))
    profile = work / "profile"
    profile.mkdir()
    preferences = write_profile(profile, args.port)
    stderr_path = output / "firefox-stderr.log"
    argv = [args.firefox, "--new-instance", "--no-remote", "--profile",
            str(profile), "--marionette", "-remote-allow-system-access"]
    environment = dict(os.environ)
    environment.setdefault("MOZ_ENABLE_WAYLAND", "1")
    client = None
    process = None
    environment_keys = (
        "MOZ_ENABLE_WAYLAND", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE",
        "LD_LIBRARY_PATH", "LIBGL_DRIVERS_PATH", "GBM_BACKENDS_PATH",
        "MESA_LOADER_DRIVER_OVERRIDE", "MESA_SHADER_CACHE_DISABLE",
        "VK_DRIVER_FILES", "VK_ICD_FILENAMES",
        "__EGL_VENDOR_LIBRARY_FILENAMES", "ZINK_DEBUG",
    )
    report = {
        "status": "error",
        "argv": argv,
        "preferences": preferences,
        "environment": {key: environment.get(key) for key in environment_keys},
        "pageSha256": hashlib.sha256(page.read_bytes()).hexdigest(),
    }
    try:
        report["firefoxVersion"] = subprocess.run(
            [args.firefox, "--version"], check=True, capture_output=True,
            text=True, timeout=5).stdout.strip()
    except Exception as error:
        report["firefoxVersionError"] = f"{type(error).__name__}: {error}"
    try:
        with stderr_path.open("wb") as stderr:
            process = subprocess.Popen(argv, stdout=stderr,
                                       stderr=subprocess.STDOUT,
                                       env=environment)
        client = Marionette(args.port, args.timeout)
        report["marionetteHello"] = client.hello
        client.command("WebDriver:NewSession", {"capabilities": {}})
        client.command("WebDriver:SetTimeouts",
                       {"script": int(args.timeout * 1000)})
        client.command("Marionette:SetContext", {"value": "content"})
        client.command("WebDriver:Navigate", {"url": page.as_uri()})
        script = """
          const done = arguments[arguments.length - 1];
          window.__f54ResultPromise.then(
            value => done(JSON.stringify(value)),
            error => done(JSON.stringify({schema:1, ok:false,
                                          error:String(error)})));
        """
        response = client.command("WebDriver:ExecuteAsyncScript",
                                  {"script": script, "args": []})
        report["result"] = json.loads(response["value"])
        screenshot = client.command("WebDriver:TakeScreenshot", {})
        encoded = screenshot.get("value") or screenshot.get("data")
        if encoded:
            (output / "firefox-f54.png").write_bytes(
                base64.b64decode(encoded, validate=True))
        failures = validate(report["result"], args.require_hardware,
                            args.require_g17p_zink)
        forbidden = ("AGX backend cannot encode", "no grounded AGX3 encoding",
                     "VK_ERROR_FEATURE_NOT_PRESENT", "VK_ERROR_DEVICE_LOST",
                     "pipeline creation failed", "WebGL context lost",
                     "GL_EXT_shader_texture_lod' unsupported")
        stderr_text = stderr_path.read_text(errors="replace")
        failures += [f"stderr contains {needle}" for needle in forbidden
                     if needle.lower() in stderr_text.lower()]
        report["failures"] = failures
        report["status"] = "ok" if not failures else "error"
    except Exception as error:  # fail closed and preserve partial evidence
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if client:
            try:
                client.command("Marionette:Quit", {"flags": ["eForceQuit"]})
            except Exception:
                pass
            client.close()
        if process:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        shutil.rmtree(work, ignore_errors=True)

    (output / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output / "result.json")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
