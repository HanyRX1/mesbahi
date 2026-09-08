"""MESBAHI — عداد تسبيح ذكي من الكاميرا (قبضة اليد) مع شرح SHAP.

Live demo:
    python main.py                  # use your webcam
    python main.py --simulate       # scripted hand, no camera required
    python main.py --camera 1       # pick another camera index

Controls:
    Q / ESC  exit    R  reset session    S  save session CSV
"""

from __future__ import annotations
import argparse, csv, datetime as dt, threading, time, sys

if sys.platform == "win32" and sys.stdout.encoding:
    sys.stdout.reconfigure(encoding="utf-8")

import cv2, mediapipe as mp, numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import core

# ─── Palette ─────────────────────────────────────────────────────
BG        = (32, 36, 44)
PANEL     = (40, 46, 58)
TEXT_W    = (240, 242, 248)
TEXT_M    = (160, 170, 190)
TEXT_L    = (100, 110, 130)
ACCENT    = (60, 180, 160)       # teal
ACCENT2   = (80, 140, 220)       # blue
GREEN_OK  = (80, 200, 130)
GOLD      = (240, 200, 80)
WARM_ON   = (80, 140, 235)
WARM_OFF  = (70, 78, 95)
SHAP_POS  = (220, 90, 70)
SHAP_NEG  = (70, 120, 200)
GAUGE_OFF = (50, 60, 75)
BAR_BG    = (55, 62, 78)

FONT_PATH = r"C:\Windows\Fonts\segoeui.ttf"
AR_NAMES  = ["الإبهام", "السبابة", "الوسطى", "البنصر", "الخنصر"]

# ─── Font cache ──────────────────────────────────────────────────
_FONTS: dict[int, ImageFont.FreeTypeFont] = {}

def _f(sz: int) -> ImageFont.FreeTypeFont:
    if sz not in _FONTS:
        _FONTS[sz] = ImageFont.truetype(FONT_PATH, sz)
    return _FONTS[sz]

def ar(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))

# ─── PIL helpers ─────────────────────────────────────────────────
def begin_layer(frame_bgr):
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

def layer_text(pil, x, y, text, size=34, color=TEXT_W):
    d = ImageDraw.Draw(pil)
    d.text((x, y), ar(text), font=_f(size), fill=color)
    return pil

def end_layer(pil):
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

# ─── Gauge ───────────────────────────────────────────────────────
def draw_gauge(frame, cx, cy, value, color):
    val33 = value % 33
    # ring
    cv2.circle(frame, (cx, cy), 100, GAUGE_OFF, 3, cv2.LINE_AA)
    for b in range(33):
        a = -90 + b * 360 / 33
        x = int(cx + 86 * np.cos(np.radians(a)))
        y = int(cy + 86 * np.sin(np.radians(a)))
        filled = b < val33
        col = color if filled else GAUGE_OFF
        cv2.circle(frame, (x, y), 5 if not filled else 7, col, -1, cv2.LINE_AA)
    # center text
    cv2.putText(frame, f"{val33}", (cx - 16, cy + 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, TEXT_W, 2, cv2.LINE_AA)
    return frame

# ─── Progress bar ────────────────────────────────────────────────
def draw_bar(frame, x, y, w, h, pct, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), BAR_BG, -1)
    if pct > 0:
        cv2.rectangle(frame, (x, y), (x + int(w * min(pct, 1.0)), y + h),
                      color, -1)
    return frame


def open_camera(idx: int, attempts: int = 5):
    """Open the camera with a working-frame probe so we never hang.

    DirectShow first (responsive on Windows), then the default backend;
    the device is only accepted once it actually delivers a frame.
    """
    for _ in range(attempts):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
        time.sleep(0.5)
    for _ in range(attempts):                    # last resort: default backend
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
        time.sleep(0.5)
    return None


def _make_hands():
    return mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ─── Main ────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="MESBAHI fist istighfar counter")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--goal", type=int, default=100)
    args = ap.parse_args()
    GOAL = args.goal

    model, explainer = core.build_count_model()
    open_th, close_th = core.count_thresholds()
    counter = core.FistCounter(open_th, close_th, hold_frames=3,
                               alpha=0.6, fist_hold=4)

    sim = args.simulate
    demo_auto = False
    cap = None
    hands = None
    if not sim:
        cap = open_camera(args.camera)
        if cap is None:
            print(ar("⚠️ تعذّر فتح الكاميرا — تشغيل الوضع التجريبي تلقائياً"))
            sim = True
            demo_auto = True
    if not sim:
        hands = _make_hands()
    sim_t0 = time.time()

    last_change = time.time()
    last_total = 0
    current = 0
    lost_frames = 0
    goal_celebrated = False
    session_rows, started = [], dt.datetime.now()
    seq_i = 0

    def save_csv():
        fn = f"mesbahi_{started:%Y%m%d_%H%M}.csv"
        with open(fn, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "count", "shap_thumb", "shap_index",
                        "shap_middle", "shap_ring", "shap_pinky"])
            w.writerows(session_rows)
        print(f"[جلسة محفوظة] → {fn}")

    print(ar("مِسباح — Q خروج · R تصفير · S حفظ"))

    while True:
        # ── frame + landmarks ─────────────────────────────────
        if sim:
            seq_i += 1
            t = time.time() - sim_t0
            mask = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0][int(t) % 11]
            lm = core.synthetic_hand(mask, seed=int(t * 10))
            feats = core.finger_features(lm, 1.0)
            has_hand = True
            frame = np.zeros((480, 640, 3), np.uint8)
            frame[:] = BG
            h, w = 480, 640
            if demo_auto:
                pil = begin_layer(frame)
                pil = layer_text(pil, 160, 12, "وضع العرض التوضيحي — بدون كاميرا", 18, GOLD)
                frame = end_layer(pil)
        else:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            has_hand = bool(res.multi_hand_landmarks)
            if has_hand:
                lm = np.array([[p.x, p.y, p.z]
                               for p in res.multi_hand_landmarks[0].landmark])
                feats = core.finger_features(lm, w / h)
            else:
                feats = None

        # ── fist counter ──────────────────────────────────────
        if has_hand:
            lost_frames = 0
            current = counter.update(feats)
        else:
            lost_frames += 1
            if lost_frames >= 5:
                counter.reset(keep_total=True)   # clear gesture state only,
                current = 0                      # keep tasbih progress

        if counter.total > last_total:
            last_change = time.time()
        last_total = counter.total

        warm = max(0.0, 1.0 - (time.time() - last_change) / 1.4)
        glow = WARM_ON if warm > 0 else WARM_OFF

        # ── goal check ────────────────────────────────────────
        if not goal_celebrated and counter.total >= GOAL:
            goal_celebrated = True
            print(ar(f"🎉 أتممتَ {GOAL} تسبيحة — جزاك الله خيراً!"))

        pct = min(counter.total / GOAL, 1.0)

        # ── overlay (dark tint) ───────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 105), BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # ── top panel ─────────────────────────────────────────
        pil = begin_layer(frame)
        pil = layer_text(pil, 24, 12, "مِسباح", 32, ACCENT)
        pil = layer_text(pil, 110, 16, "عداد تسبيح بقبضة اليد", 22, TEXT_M)
        pil = layer_text(pil, 24, 62, "افتح يدك ثم اقبضها — كل قبضة = تسبيحة", 16, TEXT_L)
        frame = end_layer(pil)

        # ── progress bar ──────────────────────────────────────
        bar_color = GREEN_OK if pct >= 1.0 else ACCENT
        draw_bar(frame, 24, 92, w - 48, 6, pct, bar_color)

        # ── main count ────────────────────────────────────────
        count_color = GREEN_OK if pct >= 1.0 else glow
        cv2.putText(frame, str(counter.total), (30, 190),
                    cv2.FONT_HERSHEY_DUPLEX, 3.0, count_color, 5, cv2.LINE_AA)

        pil = begin_layer(frame)
        pil = layer_text(pil, 30, 200, "تسبيحة", 18, TEXT_M)
        frame = end_layer(pil)

        # ── goal label ────────────────────────────────────────
        goal_col = GREEN_OK if pct >= 1.0 else TEXT_M
        pil = begin_layer(frame)
        pil = layer_text(pil, 170, 200, f"الهدف {counter.total} / {GOAL}", 16, goal_col)
        if pct >= 1.0 and warm > 0:
            pil = layer_text(pil, 160, 230, "استغفر الله العظيم", 18, GOLD)
        frame = end_layer(pil)

        # ── tasbih gauge (left panel) ─────────────────────────
        draw_gauge(frame, 130, 340, counter.total, glow)
        pil = begin_layer(frame)
        pil = layer_text(pil, 86, 280, f"حلقة {counter.total % 33} / 33", 14, TEXT_L)
        frame = end_layer(pil)

        # ── fist state indicator ──────────────────────────────
        fist_col = ACCENT if current == 0 else TEXT_L
        fist_text = "● قبضة" if current == 0 else "○ مفتوحة"
        pil = begin_layer(frame)
        pil = layer_text(pil, 30, 270, fist_text, 16, fist_col)
        frame = end_layer(pil)

        # ── SHAP explanation ──────────────────────────────────
        if has_hand:
            sv = core.explain_count(explainer, feats)
            bar_w = 140
            bar_h = 24
            gap = 10
            x0 = 280
            y0 = 260

            pil = begin_layer(frame)
            pil = layer_text(pil, x0, y0 - 26, "تحليل الشكل اليدوي  (SHAP)", 16, ACCENT)
            frame = end_layer(pil)

            maxc = max(abs(v) for v in sv.values()) or 1e-6
            for i, name in enumerate(core.FINGER_NAMES):
                v = sv[name]
                by = y0 + i * (bar_h + gap)
                # label
                pil = begin_layer(frame)
                pil = layer_text(pil, x0, by + 3, AR_NAMES[i], 13, TEXT_M)
                frame = end_layer(pil)
                # bar
                bx0 = x0 + 100
                bw = int((bar_w / 2) * abs(v) / maxc)
                col = SHAP_POS if v >= 0 else SHAP_NEG
                cv2.rectangle(frame, (bx0, by), (bx0 + bar_w, by + bar_h),
                              BAR_BG, -1)
                if v >= 0:
                    cv2.rectangle(frame, (bx0 + bar_w // 2, by + 2),
                                  (bx0 + bar_w // 2 + bw, by + bar_h - 2), col, -1)
                else:
                    cv2.rectangle(frame, (bx0 + bar_w // 2 - bw, by + 2),
                                  (bx0 + bar_w // 2, by + bar_h - 2), col, -1)
                cv2.line(frame, (bx0 + bar_w // 2, by),
                         (bx0 + bar_w // 2, by + bar_h), TEXT_L, 1, cv2.LINE_AA)

            pil = begin_layer(frame)
            pil = layer_text(pil, x0, y0 + 5 * (bar_h + gap) + 6,
                             "   يزّيد العدّ      يُعوّق العدّ", 12, TEXT_L)
            frame = end_layer(pil)

            session_rows.append(
                [dt.datetime.now().isoformat(), current] +
                [round(sv[n], 4) for n in core.FINGER_NAMES])

        # ── footer ────────────────────────────────────────────
        pil = begin_layer(frame)
        pil = layer_text(pil, 24, h - 36, "R تصفير  ·  S حفظ  ·  Q خروج", 13, TEXT_L)
        frame = end_layer(pil)

        cv2.imshow("MESBAHI — مِسباح", frame)
        key = cv2.waitKey(1) & 0xFF
        win_name = "MESBAHI — مِسباح"
        win_visible = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE)
        if key in (113, 27) or win_visible < 1:
            break
        if key == ord("r"):
            counter.reset()
            goal_celebrated = False
        if key == ord("s"):
            threading.Thread(target=save_csv, daemon=True).start()

    cap.release() if cap else None
    hands.close() if hands else None
    cv2.destroyAllWindows()
    print(ar(f"جلسة انتهت — إجمالي: {counter.total} تسبيحة"))

if __name__ == "__main__":
    main()
