"""MESBAHI — عداد تسبيح ذكي من الكاميرا (قبضة اليد) مع شرح قابل للتفسير (XAI).

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
FONT_PATH = r"C:\Windows\Fonts\segoeui.ttf"

# --------------------------------------------------------------------- Arabic
_FONTS: dict[int, ImageFont.FreeTypeFont] = {}


def _f(sz: int) -> ImageFont.FreeTypeFont:
    if sz not in _FONTS:
        _FONTS[sz] = ImageFont.truetype(FONT_PATH, sz)
    return _FONTS[sz]


def ar(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))


def begin_layer(frame_bgr):
    """Convert once per frame to a PIL draw surface (fonts cached)."""
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def layer_text(pil, x, y, text, fsize=34, color=(235, 235, 235)):
    d = ImageDraw.Draw(pil)
    d.text((x, y), ar(text), font=_f(fsize), fill=color)
    return pil


def end_layer(pil):
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def draw_gauge(frame, cx, cy, value, color):
    """33-bead tasbih gauge: value ∈ [0, 33]."""
    value %= 33
    cv2.putText(frame, f"{value}/33", (cx - 40, cy - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 210, 235), 2)
    cv2.circle(frame, (cx, cy), 110, (70, 90, 110), 4)
    for b in range(33):
        a = -90 + b * 360 / 33
        x = int(cx + 96 * np.cos(np.radians(a)))
        y = int(cy + 96 * np.sin(np.radians(a)))
        cv2.circle(frame, (x, y), 6, color if b < value else (70, 90, 110), -1)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="MESBAHI — camera istighfar counter with XAI")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--simulate", action="store_true", help="scripted hand, no camera")
    ap.add_argument("--goal", type=int, default=100, help="target istighfar (default 100)")
    args = ap.parse_args()
    GOAL = args.goal

    model, explainer = core.build_count_model()
    open_th, close_th = core.count_thresholds()
    counter = core.FistCounter(open_th, close_th, hold_frames=3,
                               alpha=0.6, fist_hold=4)
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    sim = args.simulate
    cap = cv2.VideoCapture(args.camera) if not sim else None
    sim_t0 = time.time()

    current = 0
    last_change, last_total = time.time(), 0
    lost_frames = 0
    goal_celebrated = False
    session_rows, started = [], dt.datetime.now()
    mask_seq = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    seq_i = 0

    def save_csv():
        fn = f"mesbahi_{started:%Y%m%d_%H%M}.csv"
        with open(fn, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "count", "shap_thumb", "shap_index",
                        "shap_middle", "shap_ring", "shap_pinky"])
            w.writerows(session_rows)
        print(f"[جلسة محفوظة] → {fn}")

    print(ar("مِسباح — اضغط Q للخروج، R لإعادة العدّ، S لحفظ الجلسة"))
    while True:
        # ---------------- acquire frame + landmarks
        if sim:
            seq_i += 1
            t = time.time() - sim_t0
            mask = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0][int(t) % 11]
            lm = core.synthetic_hand(mask, seed=int(t * 10))
            aspect = 1.0
            feats = core.finger_features(lm, aspect)
            has_hand = True
            frame = np.zeros((480, 640, 3), np.uint8)
            frame[:] = (30, 36, 48)
            h, w = 480, 640
        else:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            has_hand = bool(res.multi_hand_landmarks)
            if has_hand:
                lm = np.array([[p.x, p.y, p.z] for p in
                               res.multi_hand_landmarks[0].landmark])
                aspect = w / h                    # correct mediapipe's axis split
                feats = core.finger_features(lm, aspect)
            else:
                feats = None

        # ---------------- count logic: accurate per-finger state machine
        if has_hand:
            lost_frames = 0
            current = counter.update(feats)
        else:
            lost_frames += 1
            if lost_frames >= 5:            # hand gone ~0.5s -> full reset
                counter.reset()
                current = 0
        if counter.total > last_total:      # a new istighfar was committed
            last_change = time.time()
        last_total = counter.total

        # ---------------- visual composition (single PIL conversion + cached fonts)
        warm = max(0, 1.0 - (time.time() - last_change) / 1.6)
        glow = (48, 120, 235) if warm > 0 else (80, 90, 105)

        if not goal_celebrated and counter.total >= GOAL:
            goal_celebrated = True
            print(ar(f"🎉 أتممتَ {GOAL} تسبيحة — جزاك الله خيراً!"))

        # goal progress bar (toward --goal, default 100)
        pct = min(counter.total / GOAL, 1.0)
        bar_col = (40, 200, 120) if pct >= 1.0 else glow
        cv2.rectangle(frame, (24, 108), (616, 124), (60, 70, 85), 2)
        cv2.rectangle(frame, (26, 110), (26 + int(588 * pct), 122), bar_col, -1)

        pil = begin_layer(frame)
        pil = layer_text(pil, 24, 14, "مِسباح — عداد تسبيح بقبضة اليد", 40, (245, 245, 245))
        pil = layer_text(pil, 24, 64, "افتح يدك ثم اقبضها: كل قبضة = تسبيحة", 20, (190, 210, 230))
        pil = layer_text(pil, 30, 128, "إجمالي التسبيحات", 26, (210, 220, 230))
        pil = layer_text(pil, 470, 128, f"متبقي في الحلقة: {33 - counter.total % 33}", 22, (200, 210, 235))
        pil = layer_text(pil, 316, 96, f"الهدف {counter.total}/{GOAL}", 16,
                         (150, 255, 180) if pct >= 1.0 else (230, 235, 240))
        if pct >= 1.0 and warm > 0:
            pil = layer_text(pil, 180, 60, "استغفر الله العظيم ✨", 20, (245, 230, 120))
        frame = end_layer(pil)

        cv2.putText(frame, str(counter.total), (250, 150), cv2.FONT_HERSHEY_DUPLEX, 2.4, glow, 8)
        frame = draw_gauge(frame, 555, 330, counter.total, glow)

        # ---------------- SHAP explanation strip (fist state?)
        if has_hand:
            sv = core.explain_count(explainer, feats)
            y0 = 440
            state_word = "قبضة ✓" if current == 0 else f"مفتوحة ({current})"
            pil = begin_layer(frame)
            pil = layer_text(pil, 24, y0 - 34, f"حالة اليد: {state_word}  (SHAP)", 22, (150, 235, 170))
            width = 172
            maxc = max(abs(v) for v in sv.values()) or 1e-6
            for i, name in enumerate(core.FINGER_NAMES):
                v = sv[name]
                bx = 24 + i * (width + 8)
                bw = int(width / 2 * abs(v) / maxc)
                if v >= 0:
                    cv2.rectangle(frame, (bx + width // 2, y0), (bx + width // 2 + bw, y0 + 30), BARO, -1)
                else:
                    cv2.rectangle(frame, (bx, y0), (bx + width // 2 - bw, y0 + 30), BARFC, -1)
                cv2.rectangle(frame, (bx, y0), (bx + width, y0 + 30), (120, 160, 200), 1)
                pil = layer_text(pil, bx, y0 + 38, AR_NAMES[i], 16, (200, 210, 220))
            pil = layer_text(pil, 24, y0 + 72, "أحمر=يزيد العدّ   أزرق=يُعوّق العدّ", 16, (170, 200, 210))
            frame = end_layer(pil)
            cv2.putText(frame, f"fist: {1 if current == 0 else 0}", (24, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, glow, 2)

            session_rows.append([dt.datetime.now().isoformat(), current] +
                                [round(sv[n], 4) for n in core.FINGER_NAMES])
        # ---------------- footer + keys
        pil = begin_layer(frame)
        pil = layer_text(pil, 190, 578, "R تصفير  ·  S حفظ  ·  Q خروج", 18, (120, 150, 170))
        frame = end_layer(pil)

        cv2.imshow("MESBAHI — مِسباح", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (113, 27):
            break
        if key == ord("r"):
            counter.reset()
            current = 0
            goal_celebrated = False
        if key == ord("s"):
            threading.Thread(target=save_csv, daemon=True).start()

    cap.release() if cap else None
    hands.close()
    cv2.destroyAllWindows()
    print(ar(f"جلسة انتهت — إجمالي: {counter.total} تسبيحة"))


if __name__ == "__main__":
    main()