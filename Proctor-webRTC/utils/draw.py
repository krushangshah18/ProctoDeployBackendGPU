import cv2


def draw_audio_status(frame, active: bool) -> None:
    """MIC indicator — bottom-right corner to avoid overlap with score panel."""
    h, w = frame.shape[:2]
    cx, cy = w - 28, h - 28
    color  = (60, 60, 220) if active else (55, 55, 55)
    border = (120, 120, 255) if active else (100, 100, 100)
    cv2.circle(frame, (cx, cy), 12, color,  -1)
    cv2.circle(frame, (cx, cy), 12, border,  1)
    label_color = (200, 200, 255) if active else (110, 110, 110)
    cv2.putText(frame, "MIC", (w - 52, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, label_color, 1, cv2.LINE_AA)


def draw_alerts(frame, warnings: list[str], alerts: list[str]) -> None:
    """
    warnings — amber pill badge + amber text on translucent dark background
    alerts   — red pill badge + bright red text on translucent dark background
    """
    font     = cv2.FONT_HERSHEY_SIMPLEX
    scale    = 0.48
    thick    = 1
    pad_x    = 10
    pad_y    = 6
    bar_w    = 4    # thin left accent stripe
    gap      = 5
    x0       = 10
    y        = 32
    bg_alpha = 0.72  # background transparency

    def _draw_row(label, label_color, msg, msg_color, bg, accent):
        nonlocal y
        # Measure label and message separately for two-tone text
        (lw, lh), _ = cv2.getTextSize(label, font, scale, thick)
        (mw, _),  _ = cv2.getTextSize(msg,   font, scale, thick)
        tw = lw + 8 + mw   # 8px gap between label and message

        row_h  = lh + pad_y * 2
        x1, y1 = x0, y - lh - pad_y
        x2, y2 = x1 + bar_w + pad_x + tw + pad_x, y + pad_y

        # Translucent background
        roi = frame[y1:y2, x1:x2]
        if roi.size:
            overlay = roi.copy()
            overlay[:] = bg
            cv2.addWeighted(overlay, bg_alpha, roi, 1 - bg_alpha, 0, roi)
            frame[y1:y2, x1:x2] = roi

        # Left accent stripe (fully opaque)
        cv2.rectangle(frame, (x1, y1), (x1 + bar_w, y2), accent, -1)

        # Label (e.g. "WARN" / "ALERT") in accent colour
        tx = x1 + bar_w + pad_x
        cv2.putText(frame, label, (tx, y), font, scale, label_color, thick, cv2.LINE_AA)

        # Message in lighter colour
        cv2.putText(frame, msg, (tx + lw + 8, y), font, scale, msg_color, thick, cv2.LINE_AA)

        y += row_h + gap

    # ── Warnings: amber ───────────────────────────────────────────────────────
    for msg in warnings:
        _draw_row(
            label="WARN", label_color=(30, 180, 255),   # amber-orange
            msg=msg,       msg_color=(200, 220, 255),    # soft white-blue
            bg=(10, 25, 45), accent=(30, 150, 255),
        )

    # ── Alerts: red ───────────────────────────────────────────────────────────
    for msg in alerts:
        _draw_row(
            label="ALERT", label_color=(80, 80, 255),   # bright red
            msg=msg,        msg_color=(200, 210, 255),   # soft white
            bg=(10, 10, 40), accent=(60, 60, 240),
        )


def draw_detections(frame, detections):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.40
    thick = 1
    # colour per class (BGR)
    _COLORS = {
        "person"    : (80,  200,  80),
        "cell_phone": (60,  60,  255),
        "book"      : (255, 160,  40),
        "headphone" : (255, 100, 200),
        "earbud"    : (200,  80, 255),
    }
    _DEFAULT = (140, 200, 140)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls   = det["class"]
        conf  = det.get("confidence", 1.0)
        color = _COLORS.get(cls, _DEFAULT)
        label = f"{cls}  {conf:.2f}"

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # Label pill above box
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        lx1, ly1 = x1, max(0, y1 - th - 6)
        lx2, ly2 = x1 + tw + 8, y1
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
        txt_color = (10, 10, 10) if sum(color) > 400 else (240, 240, 240)
        cv2.putText(frame, label, (lx1 + 4, ly2 - 3),
                    font, scale, txt_color, thick, cv2.LINE_AA)
