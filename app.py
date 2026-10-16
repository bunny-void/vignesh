"""
Human Emergency Detector - experimental fall-detection prototype.

Run with:  streamlit run app.py

How it works (short version)
----------------------------
1. streamlit-webrtc streams webcam frames to PoseProcessor.recv() (a background thread).
2. MediaPipe Pose finds body landmarks; we draw a skeleton on the frame.
3. FallDetector (plain Python, easy to read) combines THREE signals:
     a) a sudden downward movement of the body,
     b) a big change of torso angle toward horizontal,
     c) staying down (horizontal) for HOLD_SECONDS.
   Only when all three agree is a "Possible Fall" reported.
4. The background thread never touches Streamlit widgets. It only puts events into
   a lock-protected SharedState. The dashboard (a Streamlit fragment) polls that
   state twice a second and runs the Monitoring -> Possible Fall -> Alert Active
   state machine.

Nothing is recorded or saved. This is NOT a medical device.
"""

import io
import math
import struct
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import streamlit as st

# Heavy / optional dependencies. If any is missing the dashboard and Demo Mode
# still work, and a clear message tells the user what to install.
DEPENDENCY_ERROR = None
try:
    import av
    import cv2
    import mediapipe as mp
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
except Exception as exc:  # noqa: BLE001 - we want to show any import problem
    DEPENDENCY_ERROR = f"{type(exc).__name__}: {exc}"
    VideoProcessorBase = object  # lets the class below be defined without the library


# ----------------------------------------------------------------------------
# SETTINGS - change these numbers to tune the detector and the alert flow
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DetectorConfig:
    # Landmarks with visibility below this are ignored (0..1).
    MIN_VISIBILITY: float = 0.5
    # Torso shorter than this fraction of the frame height = person too far away.
    MIN_TORSO_FRACTION: float = 0.05

    # Signal A - sudden downward movement.
    # Body centre must move down by this fraction of the frame height ...
    DROP_THRESHOLD: float = 0.15
    # ... within this many seconds.
    DROP_WINDOW_SEC: float = 0.8

    # Signal B - orientation change.
    # Torso angle (0 = upright, 90 = lying flat) must increase by this many degrees ...
    ANGLE_CHANGE_DEG: float = 35.0
    # ... compared with the smallest angle seen in this many seconds.
    ANGLE_WINDOW_SEC: float = 4.0

    # Signal C - staying down.
    # Torso counts as "down" at or above this angle (it stays "down" until the angle
    # falls 10 degrees below it, which stops jitter from resetting the timer).
    DOWN_ANGLE_DEG: float = 60.0
    DOWN_HYSTERESIS_DEG: float = 10.0
    # Person must stay down this long (the short confirmation period).
    HOLD_SECONDS: float = 2.0

    # The drop and the orientation change must both have happened this recently.
    SIGNAL_MEMORY_SEC: float = 4.0
    # If the person is not visible for longer than this, tracking restarts.
    MISSING_GRACE_SEC: float = 1.0
    # After a detection the detector stays quiet for this long.
    DETECTOR_COOLDOWN_SEC: float = 15.0


@dataclass(frozen=True)
class AlertConfig:
    COUNTDOWN_SECONDS: int = 10       # time to press "I'm OK" before the alert goes active
    COOLDOWN_AFTER_CANCEL_SEC: float = 15.0   # ignore new live detections after "I'm OK"
    COOLDOWN_AFTER_RESET_SEC: float = 10.0    # ignore new live detections after Reset
    NO_FRAMES_WARNING_SEC: float = 4.0        # warn if the video stream goes quiet


DET = DetectorConfig()
ALERT = AlertConfig()

SAFETY_NOTICE = (
    "This is an experimental fall-detection prototype. It cannot diagnose medical shock "
    "or guarantee that an emergency will be detected. Do not rely on it for personal "
    "safety or medical decisions. In a real emergency, contact local emergency services."
)

MONITORING = "monitoring"
POSSIBLE_FALL = "possible_fall"
ALERT_ACTIVE = "alert_active"
LIVE = "live"
DEMO = "demo"


# ----------------------------------------------------------------------------
# Fall detection (plain Python, no Streamlit, no MediaPipe - easy to test)
# ----------------------------------------------------------------------------
class FallDetector:
    """Rule-based, explainable fall detector fed with pose landmarks.

    Call update() once per frame. It returns a dict describing what it saw;
    result["triggered"] is True for exactly one frame when a possible fall is confirmed.
    """

    # MediaPipe Pose landmark indexes
    L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24

    def __init__(self, config: DetectorConfig = DET):
        self.cfg = config
        self.reset()

    def reset(self):
        self.history = deque()          # (time, body_center_y, torso_angle)
        self.last_drop_t = None         # last time Signal A was seen
        self.last_orient_t = None       # last time Signal B was seen
        self.down_since = None          # when the torso became "down"
        self.is_down = False
        self.last_seen_t = None
        self.cooldown_until = 0.0

    # -- helpers ----------------------------------------------------------
    def _visible_mean(self, landmarks, indexes):
        pts = [landmarks[i] for i in indexes if landmarks[i][2] >= self.cfg.MIN_VISIBILITY]
        if not pts:
            return None
        return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)

    def measure(self, landmarks, width, height):
        """Return (body_center_y, torso_angle_deg) or None if the torso is not usable.

        landmarks: list of (x, y, visibility) with x, y normalised 0..1.
        """
        if not landmarks or len(landmarks) < 25:
            return None
        shoulders = self._visible_mean(landmarks, (self.L_SHOULDER, self.R_SHOULDER))
        hips = self._visible_mean(landmarks, (self.L_HIP, self.R_HIP))
        if shoulders is None or hips is None:
            return None
        dx = (shoulders[0] - hips[0]) * width      # pixels, so the angle is not stretched
        dy = (shoulders[1] - hips[1]) * height
        torso_len = math.hypot(dx, dy)
        if torso_len < self.cfg.MIN_TORSO_FRACTION * height:
            return None
        angle = math.degrees(math.atan2(abs(dx), abs(dy)))   # 0 = upright, 90 = flat
        center_y = (shoulders[1] + hips[1]) / 2.0
        return center_y, angle

    # -- main entry point -------------------------------------------------
    def update(self, landmarks, width, height, now):
        cfg = self.cfg
        result = {
            "person_found": False, "angle": None, "center_y": None,
            "drop": False, "orientation": False, "is_down": self.is_down,
            "hold": 0.0, "triggered": False,
        }
        m = self.measure(landmarks, width, height) if landmarks else None

        if m is None:
            # Missing landmarks / poor lighting: tolerate brief gaps, then restart.
            if self.last_seen_t is not None and now - self.last_seen_t > cfg.MISSING_GRACE_SEC:
                cooldown = self.cooldown_until
                self.reset()
                self.cooldown_until = cooldown
            return result

        center_y, angle = m
        self.last_seen_t = now
        result.update(person_found=True, angle=angle, center_y=center_y)

        self.history.append((now, center_y, angle))
        keep = max(cfg.ANGLE_WINDOW_SEC, cfg.SIGNAL_MEMORY_SEC) + 1.0
        while self.history and now - self.history[0][0] > keep:
            self.history.popleft()

        # Signal A: sudden downward movement of the body centre (image y grows downward).
        recent_y = [y for t, y, _ in self.history if now - t <= cfg.DROP_WINDOW_SEC]
        if recent_y and center_y - min(recent_y) >= cfg.DROP_THRESHOLD:
            self.last_drop_t = now

        # Signal B: torso angle jumped toward horizontal.
        window_angles = [a for t, _, a in self.history if now - t <= cfg.ANGLE_WINDOW_SEC]
        if window_angles and angle - min(window_angles) >= cfg.ANGLE_CHANGE_DEG:
            self.last_orient_t = now

        # Signal C: stay down (with hysteresis so small jitter does not reset the timer).
        down_limit = cfg.DOWN_ANGLE_DEG - (cfg.DOWN_HYSTERESIS_DEG if self.is_down else 0.0)
        self.is_down = angle >= down_limit
        if self.is_down:
            if self.down_since is None:
                self.down_since = now
        else:
            self.down_since = None

        drop_recent = self.last_drop_t is not None and now - self.last_drop_t <= cfg.SIGNAL_MEMORY_SEC
        orient_recent = self.last_orient_t is not None and now - self.last_orient_t <= cfg.SIGNAL_MEMORY_SEC
        hold = (now - self.down_since) if self.down_since is not None else 0.0
        result.update(drop=drop_recent, orientation=orient_recent, is_down=self.is_down, hold=hold)

        if (drop_recent and orient_recent and hold >= cfg.HOLD_SECONDS
                and now >= self.cooldown_until):
            result["triggered"] = True
            self.cooldown_until = now + cfg.DETECTOR_COOLDOWN_SEC
            self.last_drop_t = self.last_orient_t = None
            self.down_since = None
        return result


# ----------------------------------------------------------------------------
# State shared between the video thread and the Streamlit script
# ----------------------------------------------------------------------------
class SharedState:
    """Thread-safe mailbox. The video thread writes, the dashboard reads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._events = []
        self._info = {}
        self._error = None
        self._last_frame_t = None
        self._frames = 0
        self._started_t = None

    def push_event(self, event):
        with self._lock:
            self._events.append(event)

    def pop_events(self):
        with self._lock:
            events, self._events = self._events, []
        return events

    def clear_events(self):
        with self._lock:
            self._events = []

    def frame_done(self, info):
        now = time.monotonic()
        with self._lock:
            if self._started_t is None or (self._last_frame_t and now - self._last_frame_t > 3):
                self._started_t, self._frames = now, 0   # new stream session -> restart fps count
            self._frames += 1
            self._last_frame_t = now
            self._info = info

    def set_error(self, message):
        with self._lock:
            self._error = message

    def snapshot(self):
        now = time.monotonic()
        with self._lock:
            age = None if self._last_frame_t is None else now - self._last_frame_t
            fps = 0.0
            if self._started_t is not None and self._last_frame_t and self._last_frame_t > self._started_t:
                fps = self._frames / (self._last_frame_t - self._started_t)
            return {"info": dict(self._info), "error": self._error, "frame_age": age, "fps": fps}


# ----------------------------------------------------------------------------
# Video processing (runs in a background thread - NO Streamlit calls in here)
# ----------------------------------------------------------------------------
class PoseProcessor(VideoProcessorBase):
    def __init__(self, shared: SharedState):
        self.shared = shared
        self.detector = FallDetector(DET)
        self.pose = None
        self.last_trigger_t = 0.0
        self.error_count = 0
        try:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,            # lightest model
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as exc:  # noqa: BLE001
            self.shared.set_error(f"Pose model could not start: {exc}")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            img = self._process(img)
        except Exception as exc:  # noqa: BLE001 - never kill the stream because of one bad frame
            self.error_count += 1
            self.shared.set_error(f"Frame processing error: {exc}")
            cv2.putText(img, "Processing error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _process(self, img):
        height, width = img.shape[:2]
        now = time.monotonic()
        info = {"person_found": False}

        if self.pose is None:
            cv2.putText(img, "Pose model unavailable", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)
            self.shared.frame_done(info)
            return img

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)

        landmarks = None
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
            landmarks = [(p.x, p.y, p.visibility) for p in results.pose_landmarks.landmark]

        res = self.detector.update(landmarks, width, height, now)
        info = dict(res)
        if res["triggered"]:
            self.last_trigger_t = now
            self.shared.push_event({
                "kind": "possible_fall",
                "time": datetime.now(),
                "angle": res["angle"],
            })

        # Small status overlay on the video
        if res["person_found"]:
            line = f"Torso angle {res['angle']:.0f} deg | down {res['hold']:.1f}/{DET.HOLD_SECONDS:.1f}s"
            color = (0, 200, 0)
        else:
            line = "No person detected (show shoulders and hips)"
            color = (0, 165, 255)
        cv2.putText(img, line, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if now - self.last_trigger_t < 4.0:
            cv2.rectangle(img, (0, 0), (width - 1, height - 1), (0, 0, 255), 8)
            cv2.putText(img, "POSSIBLE FALL DETECTED", (10, height - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        self.shared.frame_done(info)
        return img

    def __del__(self):
        try:
            if self.pose is not None:
                self.pose.close()
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------------
# Alert state machine (plain functions on a dict-like state, so they are testable)
# ----------------------------------------------------------------------------
def init_state(s):
    defaults = {
        "mode": MONITORING,
        "source": None,          # LIVE or DEMO for the current non-monitoring state
        "deadline": None,        # when the countdown ends (time.time())
        "cooldown_until": 0.0,   # live detections ignored until then (time.time())
        "log": [],
        "notice": None,
    }
    for key, value in defaults.items():
        if key not in s:
            s[key] = value


def add_log(s, source, message):
    s["log"].insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "source": source, "message": message})
    del s["log"][200:]


def trigger_possible_fall(s, source, now, note=""):
    """Start the countdown. Returns False if ignored (busy or in cooldown)."""
    if s["mode"] != MONITORING:
        return False
    if source == LIVE and now < s["cooldown_until"]:
        return False
    s["mode"], s["source"] = POSSIBLE_FALL, source
    s["deadline"] = now + ALERT.COUNTDOWN_SECONDS
    add_log(s, source, "Possible Fall Detected" + (f" ({note})" if note else ""))
    return True


def tick(s, now):
    """Move from Possible Fall to Alert Active when the countdown runs out."""
    if s["mode"] == POSSIBLE_FALL and s["deadline"] is not None and now >= s["deadline"]:
        s["mode"] = ALERT_ACTIVE
        add_log(s, s["source"], "Alert Active - no cancellation received")


def cancel_alert(s, now):
    if s["mode"] not in (POSSIBLE_FALL, ALERT_ACTIVE):
        return False
    source = s["source"]
    add_log(source=source, s=s, message="Alert cancelled - person reported OK")
    s["mode"], s["source"], s["deadline"] = MONITORING, None, None
    if source == LIVE:
        s["cooldown_until"] = now + ALERT.COOLDOWN_AFTER_CANCEL_SEC
    return True


def reset_all(s, now):
    add_log(s, s["source"] or LIVE, "Reset - back to monitoring")
    s["mode"], s["source"], s["deadline"] = MONITORING, None, None
    s["cooldown_until"] = now + ALERT.COOLDOWN_AFTER_RESET_SEC
    s["notice"] = None


def real_alert_in_progress(s):
    return s["mode"] != MONITORING and s["source"] == LIVE


def demo_normal(s):
    if real_alert_in_progress(s):
        return False
    s["mode"], s["source"], s["deadline"] = MONITORING, None, None
    add_log(s, DEMO, "Normal monitoring (simulated)")
    return True


def demo_fall(s, now):
    if real_alert_in_progress(s):
        return False
    s["mode"], s["source"], s["deadline"] = MONITORING, None, None   # restart the demo cleanly
    return trigger_possible_fall(s, DEMO, now, "simulated")


def demo_cancel(s, now):
    if s["source"] != DEMO or s["mode"] == MONITORING:
        return False
    return cancel_alert(s, now)


def demo_confirm(s, now):
    """Simulate a confirmed alert: jump straight to Alert Active."""
    if real_alert_in_progress(s):
        return False
    if s["mode"] == MONITORING:
        trigger_possible_fall(s, DEMO, now, "simulated")
    if s["mode"] == POSSIBLE_FALL:
        s["deadline"] = now          # countdown ends immediately
        tick(s, now)
    return s["mode"] == ALERT_ACTIVE


# ----------------------------------------------------------------------------
# Alert sound: a short beep pattern generated in memory (no files, no downloads)
# ----------------------------------------------------------------------------
@st.cache_data
def make_alert_wav() -> bytes:
    rate = 22050
    frames = bytearray()
    for _ in range(3):                                   # three beeps, then a pause
        for i in range(int(rate * 0.18)):
            value = int(14000 * math.sin(2 * math.pi * 880 * i / rate))
            frames += struct.pack("<h", value)
        frames += b"\x00\x00" * int(rate * 0.12)
    frames += b"\x00\x00" * int(rate * 0.5)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
STYLE = """
<style>
.status-card {border-radius: 14px; padding: 18px 20px; margin-bottom: 8px; color: #fff;}
.status-card h2 {margin: 0; font-size: 1.6rem; line-height: 1.25; color: #fff;}
.status-card p {margin: 6px 0 0 0; font-size: 1rem; opacity: .95;}
.s-monitoring {background: #1f8a4c;}
.s-possible {background: #c77700;}
.s-alert {background: #c62828; animation: pulse 1s infinite;}
@keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(198,40,40,.6);} 70% {box-shadow: 0 0 0 16px rgba(198,40,40,0);} 100% {box-shadow: 0 0 0 0 rgba(198,40,40,0);}}
.badge {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 700; margin-bottom: 8px; background: rgba(255,255,255,.25);}
</style>
"""


def is_local_session() -> bool:
    try:
        host = st.context.headers.get("Host", "")
        return host.startswith(("localhost", "127.0.0.1", "[::1]"))
    except Exception:  # noqa: BLE001
        return True


def render_status_card(s, now):
    mode, source = s["mode"], s["source"]
    badge = ""
    if source == DEMO:
        badge = '<div class="badge">DEMO - SIMULATED, NOT A REAL DETECTION</div>'
    elif source == LIVE:
        badge = '<div class="badge">LIVE WEBCAM DETECTION</div>'

    if mode == MONITORING:
        css, title, body = "s-monitoring", "Monitoring", "No fall detected."
    elif mode == POSSIBLE_FALL:
        remaining = max(0, math.ceil(s["deadline"] - now))
        css, title = "s-possible", "Possible Fall Detected"
        body = f"Alert becomes active in {remaining} s unless cancelled."
    else:
        css, title = "s-alert", "Alert Active - Check the Person Immediately"
        body = "No cancellation was received."
    st.markdown(f'<div class="status-card {css}">{badge}<h2>{title}</h2><p>{body}</p></div>',
                unsafe_allow_html=True)


def render_camera_info(snap):
    age, info = snap["frame_age"], snap["info"]
    if snap["error"]:
        st.error(snap["error"])
    if age is not None and age < 2.0:
        if info.get("person_found"):
            st.caption(f"Camera live ({snap['fps']:.0f} fps). Person found - torso angle "
                       f"{info['angle']:.0f} deg. Down {info['hold']:.1f}s / {DET.HOLD_SECONDS:.1f}s. "
                       f"Signals: drop {'yes' if info['drop'] else 'no'}, "
                       f"orientation change {'yes' if info['orientation'] else 'no'}.")
        else:
            st.caption(f"Camera live ({snap['fps']:.0f} fps). No person found - make sure your "
                       "shoulders and hips are visible and the room is well lit.")
    elif age is not None and age >= ALERT.NO_FRAMES_WARNING_SEC:
        st.warning("The video stream stopped or was interrupted. Press STOP then START on the "
                   "camera panel to reconnect.")
    else:
        st.caption("Camera is not running. Press START on the camera panel to begin live monitoring.")


@st.fragment(run_every=0.5)
def dashboard(shared: SharedState):
    s = st.session_state
    now = time.time()

    # 1. Collect events from the video thread (real detections only).
    snap = shared.snapshot()
    for event in shared.pop_events():
        if event["kind"] == "possible_fall":
            trigger_possible_fall(s, LIVE, now, f"torso angle {event['angle']:.0f} deg")
    tick(s, now)

    # 2. Status
    render_status_card(s, now)
    if s["mode"] != MONITORING:
        st.audio(make_alert_wav(), format="audio/wav", autoplay=True, loop=True)
        st.caption("Alert sound is playing. If you hear nothing, your browser blocked autoplay - "
                   "press play on the audio bar.")

    # 3. Buttons
    c1, c2 = st.columns(2)
    if c1.button("I'm OK - Cancel Alert", type="primary", use_container_width=True,
                 disabled=s["mode"] == MONITORING, key="btn_cancel"):
        cancel_alert(s, time.time())
        st.rerun(scope="fragment")
    if c2.button("Reset to Monitoring", use_container_width=True, key="btn_reset"):
        reset_all(s, time.time())
        shared.clear_events()
        st.rerun(scope="fragment")

    render_camera_info(snap)

    # 4. Demo mode
    with st.expander("Demo Mode (simulated events - no webcam needed)", expanded=False):
        st.caption("These buttons only simulate events for presentations. Simulated events are "
                   "labelled DEMO in the status card and the event log.")
        d1, d2 = st.columns(2)
        clicked = None
        if d1.button("Normal monitoring", use_container_width=True, key="demo_normal"):
            clicked = demo_normal(s)
        if d2.button("Possible fall detected", use_container_width=True, key="demo_fall"):
            clicked = demo_fall(s, time.time())
        if d1.button("Alert cancelled", use_container_width=True, key="demo_cancel"):
            clicked = demo_cancel(s, time.time())
        if d2.button("Alert confirmed", use_container_width=True, key="demo_confirm"):
            clicked = demo_confirm(s, time.time())
        if clicked is False:
            if real_alert_in_progress(s):
                st.warning("A real webcam alert is in progress. Cancel or reset it before using Demo Mode.")
            else:
                st.info("Nothing to do: that demo step does not apply in the current state.")
        elif clicked:
            st.rerun(scope="fragment")

    # 5. Event log
    st.subheader("Event log")
    with st.container(height=220):
        if not s["log"]:
            st.caption("No events yet.")
        for entry in s["log"]:
            tag = "DEMO" if entry["source"] == DEMO else "LIVE"
            icon = "🟣" if entry["source"] == DEMO else "🟢"
            st.markdown(f"`{entry['time']}` {icon} **{tag}** - {entry['message']}")


def main():
    st.set_page_config(page_title="Human Emergency Detector", page_icon="🚨", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    init_state(st.session_state)
    if "shared" not in st.session_state:
        st.session_state["shared"] = SharedState()
    shared = st.session_state["shared"]

    st.title("🚨 Human Emergency Detector")
    st.caption("Webcam posture monitoring that flags a possible fall - educational prototype.")
    st.warning(SAFETY_NOTICE)
    if is_local_session():
        st.caption("Privacy: video frames are processed in memory on this computer. "
                   "Nothing is recorded, saved or sent to any AI service.")
    else:
        st.error("This app is running on a remote server: your webcam frames are sent to and "
                 "processed by that server (in memory only, nothing is saved by this app). "
                 "Do not use it if you are not comfortable with that.")

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Camera")
        if DEPENDENCY_ERROR:
            st.error("Live webcam monitoring is unavailable because a required package could not "
                     f"be loaded:\n\n`{DEPENDENCY_ERROR}`\n\nRun `pip install -r requirements.txt` "
                     "inside your virtual environment (Python 3.10-3.12). Demo Mode still works.")
        else:
            webrtc_streamer(
                key="fall-detector",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: PoseProcessor(shared),
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                media_stream_constraints={
                    "video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                              "frameRate": {"ideal": 15}},
                    "audio": False,
                },
                async_processing=True,
                video_html_attrs={"autoPlay": True, "playsInline": True, "muted": True,
                                  "controls": False, "style": {"width": "100%", "borderRadius": "12px"}},
            )
            st.caption("Press START to begin monitoring and STOP to end it. Your browser will ask "
                       "for camera permission the first time. Stand so your shoulders and hips "
                       "are in view, 2-3 metres from the camera.")

    with right:
        st.subheader("Status")
        dashboard(shared)


main()
