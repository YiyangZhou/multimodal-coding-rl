"""Sandbox-only browser operations; no model API credentials or source inspection.

Chromium runs independently so successive command invocations retain the page.
Only fixed browser operations are exposed: no shell, JavaScript or file tools.
"""
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit
import urllib.request

WORK = Path("/workspace/.mmrl")
CDP = "http://127.0.0.1:9222"


def connect(pw):
    # Resolve the local endpoint directly; sandbox proxy environment settings
    # must not route Chromium's local discovery request through an HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(CDP + "/json/version", timeout=2) as response:
        endpoint = json.load(response)["webSocketDebuggerUrl"]
    parsed = urlsplit(endpoint)
    if parsed.scheme != "ws" or parsed.hostname != "127.0.0.1" or parsed.port != 9222:
        raise RuntimeError("Unexpected local browser endpoint")
    return pw.chromium.connect_over_cdp(endpoint, timeout=5000)


def validate_action(action):
    if not isinstance(action, dict) or not isinstance(action.get("op"), str):
        raise ValueError("Expected a browser operation")
    fields = {"snapshot": set(), "click": {"locator"}, "fill": {"locator", "value"},
              "select": {"locator", "value"}, "check": {"locator"}, "uncheck": {"locator"},
              "press": {"key"}, "back": set(), "reload": set(), "read_text": {"locator"},
              "screenshot": {"fullpage"}, "start": {"width", "height"}}
    op = action["op"]
    if op not in fields or set(action) != {"op", *fields[op]}:
        raise ValueError("Unsupported browser operation or extra arguments")
    if "locator" in action:
        locator = action["locator"]
        if not isinstance(locator, dict) or set(locator) != {"kind", "name", "role"}:
            raise ValueError("Invalid browser locator")
        if locator["kind"] not in {"role", "text", "label", "placeholder"}:
            raise ValueError("Unsupported browser locator kind")
        if any(not isinstance(locator[k], str) or len(locator[k]) > 1000 for k in ("name", "role")):
            raise ValueError("Invalid browser locator text")
        if not locator["name"] or (locator["kind"] == "role" and not locator["role"]):
            raise ValueError("An exact named browser element is required")
    if "value" in action and (not isinstance(action["value"], str) or len(action["value"]) > 8192):
        raise ValueError("Invalid form value")
    if op == "press" and action["key"] not in {"Enter", "Escape", "Tab", "Space", "Backspace", "Delete",
                                               "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "PageUp", "PageDown", "Home", "End"}:
        raise ValueError("Unsupported keyboard input")
    if op == "screenshot" and type(action["fullpage"]) is not bool:
        raise ValueError("Screenshot scope must be boolean")
    if op == "start" and any(type(action[k]) is not int or not 100 <= action[k] <= 4096 for k in ("width", "height")):
        raise ValueError("Invalid browser viewport")
    return action


def locate(page, value):
    if value["kind"] == "role":
        return page.get_by_role(value["role"], name=value["name"], exact=True)
    return {"text": page.get_by_text, "label": page.get_by_label,
            "placeholder": page.get_by_placeholder}[value["kind"]](value["name"], exact=True)


def app_url(url):
    parsed = urlsplit(url)
    return (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}
            and parsed.port == 3000 and not parsed.username and not parsed.password)


def execute(action):
    from playwright.sync_api import Error, TimeoutError as BrowserTimeout, sync_playwright

    action = validate_action(action)
    with sync_playwright() as pw:
        if action["op"] == "start":
            # A fresh browser per workflow; state persists between its cases.
            if (WORK / "browser.pid").exists():
                raise RuntimeError("An active workflow browser already exists")
            WORK.mkdir(exist_ok=True)
            (WORK / "viewport.json").write_text(json.dumps({"width": action["width"], "height": action["height"]}))
            with (WORK / "browser.log").open("wb") as log:
                proc = subprocess.Popen([pw.chromium.executable_path, "--headless", "--no-sandbox",
                    "--disable-dev-shm-usage", "--remote-debugging-port=9222", "--remote-debugging-address=127.0.0.1",
                    "--user-data-dir=" + str(WORK / "profile"), "about:blank"],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            (WORK / "browser.pid").write_text(str(proc.pid))
            browser = None
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                try:
                    browser = connect(pw)
                    break
                except (Error, OSError) as exc:
                    # Local Chromium transport only; this sandbox has no judge
                    # credentials. Keep the error for diagnosing runtime failures.
                    (WORK / "connect-error.txt").write_text(str(exc)[:4000])
                    if proc.poll() is not None:
                        raise RuntimeError("Browser startup failed") from None
                    time.sleep(.2)
            if browser is None:
                raise RuntimeError("Browser endpoint unavailable")
            page = browser.contexts[0].pages[0]
            page.set_viewport_size({"width": action["width"], "height": action["height"]})
            page.set_default_timeout(3000)
            try:
                response = page.goto("http://localhost:3000", wait_until="domcontentloaded", timeout=3000)
                if response is not None and response.status >= 400:
                    return {"status": "app_failure", "reason": "http_error", "http_status": response.status}
            except (BrowserTimeout, Error):
                return {"status": "app_failure", "reason": "initial_navigation_failed"}
        else:
            browser = connect(pw)
            page = browser.contexts[0].pages[0]
            page.set_default_timeout(3000)
        # CDP emulation belongs to the connection, not the persistent page.
        # Reapply the dataset viewport after every reconnect.
        page.set_viewport_size(json.loads((WORK / "viewport.json").read_text()))
        if not app_url(page.url):
            return {"status": "app_failure", "reason": "unexpected_navigation"}
        try:
            op = action["op"]
            if op in {"click", "check", "uncheck"}:
                getattr(locate(page, action["locator"]), op)(timeout=3000)
            elif op in {"fill", "select"}:
                getattr(locate(page, action["locator"]), "fill" if op == "fill" else "select_option")(action["value"], timeout=3000)
            elif op == "press":
                page.keyboard.press(action["key"])
            elif op in {"back", "reload"}:
                getattr(page, "go_back" if op == "back" else "reload")(wait_until="domcontentloaded", timeout=3000)
            elif op == "read_text":
                return {"status": "ok", "text": locate(page, action["locator"]).inner_text(timeout=3000)[:32000]}
            if not app_url(page.url):
                return {"status": "app_failure", "reason": "unexpected_navigation"}
            if op == "screenshot":
                page.screenshot(path=str(WORK / "screenshot.png"), full_page=action["fullpage"], timeout=3000)
                return {"status": "ok", "screenshot": "/workspace/.mmrl/screenshot.png"}
            state = page.locator("body").aria_snapshot(timeout=3000)
            return {"status": "ok", "url": page.url, "snapshot": state[:32000], "snapshot_truncated": len(state) > 32000}
        except BrowserTimeout:
            return {"status": "app_failure", "reason": "action_timeout"}


def stop():
    from playwright.sync_api import sync_playwright
    import shutil
    with sync_playwright() as pw:
        browser = connect(pw)
        browser.close()
    # Only this verifier's owned browser files are removed.
    (WORK / "browser.pid").unlink()
    (WORK / "viewport.json").unlink()
    shutil.rmtree(WORK / "profile", ignore_errors=True)


if __name__ == "__main__":
    WORK.mkdir(exist_ok=True)
    try:
        if sys.argv[1:] == ["--stop"]:
            stop()
            result = {"status": "ok"}
        elif sys.argv[1:] == ["--request"]:
            result = execute(json.loads((WORK / "request.json").read_text()))
        else:
            raise ValueError("Unknown browser invocation")
    except Exception as exc:
        result = {"status": "infrastructure_error", "reason": type(exc).__name__}
        if isinstance(exc, RuntimeError) and str(exc) in {
            "Unexpected local browser endpoint", "Browser startup failed",
            "Browser endpoint unavailable", "An active workflow browser already exists",
        }:
            result["detail"] = str(exc)
    (WORK / "reply.json").write_text(json.dumps(result))
    raise SystemExit(30 if result["status"] == "infrastructure_error" else 0)
