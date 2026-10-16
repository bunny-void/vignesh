# Human Emergency Detector

An **experimental, educational** Streamlit app that watches a webcam, draws a body skeleton with MediaPipe Pose, and shows a **"Possible Fall Detected"** warning when a fall-like pattern is seen.

> **Safety notice:** This is an experimental fall-detection prototype. It cannot diagnose medical shock or guarantee that an emergency will be detected. Do not rely on it for personal safety or medical decisions. In a real emergency, contact local emergency services.

**Privacy:** frames are processed in memory by the computer running the app. Nothing is recorded or saved, and no footage is sent to any AI service. If you host the app on a remote server, your frames are processed by that server (the app shows a red notice in that case). The WebRTC connection uses Google's public STUN server only to set up the connection; video is not sent to it.

## Files

- `app.py` - the whole app (settings are at the top)
- `requirements.txt` - pinned package versions
- `README.md` - this file

## Run it on Windows

1. **Install Python 3.10, 3.11 or 3.12** from python.org (tick *Add python.exe to PATH*). **Do not use Python 3.13** - the pinned MediaPipe version does not support it. Check with `py --list`.
2. Open **Command Prompt** or PowerShell and go to the project folder:
   ```
   cd path\to\human-emergency-detector
   ```
3. Create and activate a virtual environment (use `-3.11` or `-3.12` to match what you installed):
   ```
   py -3.11 -m venv venv
   venv\Scripts\activate
   ```
   In PowerShell, if activation is blocked, run `Set-ExecutionPolicy -Scope Process Bypass` first.
4. Install the dependencies:
   ```
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Start the app:
   ```
   streamlit run app.py
   ```
   Your browser opens at **http://localhost:8501**.

## Test it with your webcam

1. Close other apps that use the camera (Zoom, Teams, etc.).
2. Press **START** in the Camera panel and allow camera access. You should see your skeleton and a line such as "Torso angle 5 deg".
3. Stand 2-3 metres back so your **shoulders and hips** are visible, in a well-lit room, with some floor space around you.
4. Simulate a fall safely (soft surface, clear area): drop quickly to the floor so your torso ends up horizontal, and stay down for about 3 seconds.
5. The status card turns amber: **Possible Fall Detected**, an alert tone plays, and a 10-second countdown runs.
   - Click **I'm OK - Cancel Alert** to return to monitoring, or
   - wait and it changes to **Alert Active - Check the Person Immediately**.
6. **Reset to Monitoring** always returns to normal.

Things that should *not* trigger an alert: sitting down, bending to pick something up, slowly lying down, or walking out of view.

### Demo Mode (no webcam needed)

Open **Demo Mode** under the status card. Four buttons simulate: normal monitoring, possible fall, alert cancelled, alert confirmed. Demo events are labelled **DEMO** in the status card and the event log (purple dot), live webcam events **LIVE** (green dot). Demo buttons refuse to override a real webcam alert.

## How detection works

A possible fall is reported only when **all three** signals agree:

1. **Sudden downward movement** - the body centre drops by 15% of the frame height within 0.8 s.
2. **Orientation change** - the torso angle swings at least 35 degrees toward horizontal within the last 4 s.
3. **Staying down** - the torso stays near horizontal (60 degrees or more from vertical) for 2 s.

A cooldown prevents repeated alerts for the same event. Tune all numbers in `DetectorConfig` and `AlertConfig` at the top of `app.py`.

## Limitations

- Needs the shoulders and hips visible; poor lighting, clutter, or furniture hiding the body reduce accuracy.
- Falls straight toward or away from the camera, or behind furniture, may be missed. Other fast movements can cause false alarms.
- One person in view works best.
- The alert sound may be blocked by browser autoplay rules; the page then shows an audio bar you can press.
- No phone calls, SMS, GPS, or emergency-service integration.

## Troubleshooting

- **`ModuleNotFoundError` / install fails on mediapipe:** you are probably on Python 3.13 or 32-bit Python. Use 64-bit Python 3.10-3.12.
- **Camera does not start:** allow the camera in the browser address bar, close other apps using it, or try Chrome/Edge.
- **Black or frozen video:** press STOP then START.
- **`streamlit` not recognised:** activate the virtual environment (step 3).
