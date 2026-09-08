"""MESBAHI — عداد استغفار ذكي من الكاميرا مع شرح قابل للتفسير (XAI).

Live demo:
    python main.py                  # use your webcam
    python main.py --simulate       # scripted hand, no camera required
    python main.py --camera 1       # pick another camera index

Controls (focus window):
    Q / ESC   quit       R   reset session   S   save session to CSV
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import threading
import time

# Windows console: speak Arabic properly
import sys
if sys.platform == "win32" and sys.stdout.encoding:
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import arabic_reshaper
from bidi.algorithm import get_display

import core

AR_NAMES = ["إبهام", "سبابة", "وسطى", "بنصر", "خنصر"]
BARFC = (32, 74, 135)   # deep blue
BARO  = (170, 20, 20)   # crimson

# --------------------------------------------------------------------- Arabic
_FONT = None
def _f(sz: int) -> ImageFont.FreeTypeFont:
    global _FONT
    return ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", sz)


def ar(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))


def overlay_arabic(frame_bgr, x, y, text, fsize=34, color=(235, 235, 235)):
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.text((x, y), ar(text), font=_f(fsize), fill=color)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def draw_gauge(frame, cx, cy, value, color):
    """33-bead tasbih gauge: value ∈ [0, 33]. Returns an unmodified copy."""
    prog = (value % 33) / 33
    cv2.putText(frame, f"{value % 33}/33", (cx - 40, cy - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 210, 235), 2)
    cv2.circle(frame, (cx, cy), 110, (70, 90, 110), 4)
    for b in range(33):
        a = -90 + b * 360 / 33
        on = (b + 1) <= (value % 33) or value % 33 == 0 and b == 0
        c = color if on else (70, 90, 110)
        x = int(cx + 96 * np.cos(np.radians(a)))
        y = int(cy + 96 * np.sin(np.radians(a)))
        cv2.circle(frame, (x, y), 6, c, -1)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="MESBAHI — camera istighfar counter with XAI")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--simulate", action="store_true", help="scripted hand, no camera")
    ap.add_argument("--goal", type=int, default=100)
    args = ap.parse_args()

    model, explainer = core.build_count_model()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

    sim = args.simulate
    cap = cv2.VideoCapture(args.camera) if not sim else None
    sim_t0, sim_count = time.time(), 0

    total, current = 0, 0
    prev_open, last_change = -1, time.time()
    session_rows, started = [], dt.datetime.now()

    def save_csv():
        fn = f"mesbahi_{started:%Y%m%d_%H%M}.csv"
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "count", "shap_thumb", "shap_index",
                        "shap_middle", "shap_ring", "shap_pinky"])
            w.writerows(session_rows)
        print(f"[اتصلت الجلسة] → {fn}")

    print(ar("مِسباح — اضغط Q للخروج، R لإعادة العدّ، S لحفظ الجلسة"))
    while True:
        if sim:
            t = time.time() - sim_t0
            mask = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0][int(t // 2) % 12]
            lm = core.synthetic_hand(mask, seed=int(t))
            frame = np.zeros((480, 640, 3), np.uint8)
            frame[:] = (30, 36, 48)
            feats = core.finger_features(lm)
            has_hand = True
        else:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            has_hand = bool(res.multi_hand_landmarks)
            if has_hand:
                lm = np.array([[p.x, p.y, p.z] for p in
                               res.multi_hand_landmarks[0].landmark])
                feats = core.finger_features(lm)
            else:
                feats = None

        if has_hand:
            current = core.count_with_stable_finger(model, feats, prev_open)[0]
            if current != prev_open:          # a finger toggled — count it
                if current > prev_open:
                    total += current - prev_open
                last_change = time.time()
                prev_open = current
        else:
            prev_open = -1

        # ----- shimmer when a new istighfar is counted
        warm = max(0, 1.0 - (time.time() - last_change) / 1.6)
        glow = (48, 120, 235) if warm > 0 else (60, 66, 82)

        frame = overlay_arabic(frame, 24, 14, "مِسباح — عداد استغفار ذكي", 40, (245, 245, 245))
        frame = overlay_arabic(frame, 24, 66, "افتح إصبعاً: كل إصبع = استغفار ☝️", 20, (190, 210, 230))

        # ----- big session counter
        cv2.putText(frame, str(total), (250, 150), cv2.FONT_HERSHEY_DUPLEX, 2.4, glow, 8)
        frame = overlay_arabic(frame, 30, 128, "إجمالي الاستغفار", 26, (210, 220, 230))

        # ----- current round (33-bead tasbih gauge)
        frame = overlay_arabic(frame, 470, 128, f"متبقي في الحلقة: {33 - (total % 33)}", 22, (200, 210, 235))
        frame = draw_gauge(frame, 555, 330, total, glow)

        # ----- SHAP explanation strip (why n?)
        if has_hand:
            shap_vals = core.explain_count(explainer, feats)
            y0 = 440
            frame = overlay_arabic(frame, 24, y0 - 34, f"لماذا عُدّ {current}؟  (SHAP)", 22, (150, 235, 170))
            names = list(shap_vals)
            width = 172
            maxc = max(abs(v) for v in shap_vals.values()) or 1e-6
            for i, name in enumerate(names):
                v = shap_vals[name]; bx = 24 + i * (width + 8)
                bw = int(width / 2 * abs(v) / maxc)
                if v >= 0:
                    cv2.rectangle(frame, (bx + width // 2, y0), (bx + width // 2 + bw, y0 + 30), BARO, -1)
                else:
                    cv2.rectangle(frame, (bx, y0), (bx + width // 2 - bw, y0 + 30), BARFC, -1)
                cv2.rectangle(frame, (bx, y0), (bx + width, y0 + 30), (120, 160, 200), 1)
                frame = overlay_arabic(frame, bx, y0 + 38, AR_NAMES[i], 16, (200, 210, 220))
            frame = overlay_arabic(frame, 24, y0 + 72, "أحمر=يزيد العدّ   أزرق=يُعوّق العدّ", 16, (170, 200, 210))
            cv2.putText(frame, f"open: {current}", (24, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, glow, 2)

        # ----- footer keys
        frame = overlay_arabic(frame, 190, 578, "R تصفير  ·  S حفظ  ·  Q خروج", 16, (120, 150, 170))

        cv2.imshow("MESBAHI — مِسباح", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (113, 27):
            break
        if key == ord("r"):
            total, current, prev_open = 0, 0, -1
        if key == ord("s"):
            threading.Thread(target=save_csv, daemon=True).start()

        if has_hand:
            sv = core.explain_count(explainer, feats)
            session_rows.append([dt.datetime.now().isoformat(), current] +
                                [round(sv[n], 4) for n in core.FINGER_NAMES])

    cap.release() if cap else None
    hands.close()
    cv2.destroyAllWindows()
    print(ar(f"جلسة انتهت — إجمالي: {total} استغفار"))


if __name__ == "__main__":
    main()