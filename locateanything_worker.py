#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_with_token_prob.py

Object detection using LocateAnything with per-token generation probability
as confidence score.

Usage:
    python detect_with_token_prob.py <image> [cat1,cat2,...]

Example:
    python detect_with_token_prob.py bus.jpg person,car,bicycle
"""

import sys
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 1. Parse model output
# ============================================================

def parse_detections(answer, image_width, image_height):
    """
    Extract detection boxes from the model's text answer.
    Returns a list of dicts, each with keys:
        label, x1, y1, x2, y2  (pixel coordinates)
    """
    detections = []
    current_label = None

    # Collect all <ref> and <box> matches with their positions
    events = []
    for m in re.finditer(r"<ref>(.*?)</ref>", answer):
        events.append((m.start(), "ref", m.group(1)))
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        events.append((m.start(), "box", [int(g) for g in m.groups()]))
    events.sort(key=lambda e: e[0])

    for _, kind, value in events:
        if kind == "ref":
            current_label = value
        elif kind == "box" and current_label is not None:
            x1n, y1n, x2n, y2n = value
            detections.append({
                "label": current_label,
                "x1": x1n / 1000.0 * image_width,
                "y1": y1n / 1000.0 * image_height,
                "x2": x2n / 1000.0 * image_width,
                "y2": y2n / 1000.0 * image_height,
            })

    return detections


# ============================================================
# 2. Map token probabilities to each box
# ============================================================

def compute_box_probs(answer, token_strs, token_probs):
    """
    Map per-token generation probabilities to each <box> region.

    Strategy:
        1. Reconstruct the answer string from token_strs.
        2. Walk through the answer and find the character ranges
           that correspond to each <box>...and <ref>...</ref> region.
        3. For each box, gather the probabilities of all tokens that
           fall within its character span, and take their geometric
           mean (product of probs ^ 1/N).
        4. The label probability uses the same approach for the
           <ref>...</ref> span.

    Args:
        answer:      full decoded string from the model.
        token_strs:  list of token strings (one per generated step).
        token_probs: list of float probabilities (same length).

    Returns:
        box_probs:  list of float, one per detected box.
        label_probs: list of float, one per detected box (label confidence).
    """
    # Build a cumulative string from tokens to map character positions
    # to token indices.
    cum_chars = ""          # cumulative string
    char_to_token = []      # char_to_token[i] = token index for char position i

    for tok_idx, tok in enumerate(token_strs):
        for ch in tok:
            char_to_token.append(tok_idx)
            cum_chars += ch

    # Find the character span of each <box> and <ref> in cum_chars
    box_spans = []
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", cum_chars):
        box_spans.append((m.start(), m.end()))

    ref_spans = []
    for m in re.finditer(r"<ref>(.*?)</ref>", cum_chars):
        ref_spans.append((m.start(), m.end()))

    # Collect token indices for each span
    def tokens_in_span(start, end):
        """Return the set of unique token indices that overlap [start, end)."""
        tok_ids = set()
        for pos in range(start, min(end, len(char_to_token))):
            tok_ids.add(char_to_token[pos])
        return tok_ids

    box_probs = []
    for span_start, span_end in box_spans:
        tok_ids = tokens_in_span(span_start, span_end)
        if not tok_ids:
            box_probs.append(0.0)
            continue
        # Geometric mean of token probabilities
        probs = [token_probs[i] for i in tok_ids if i < len(token_probs)]
        if probs:
            # Use geometric mean: product^(1/n)
            log_probs = [np.log(max(p, 1e-10)) for p in probs]
            geo_mean = np.exp(np.mean(log_probs))
            box_probs.append(round(float(geo_mean), 4))
        else:
            box_probs.append(0.0)

    label_probs = []
    for span_start, span_end in ref_spans:
        tok_ids = tokens_in_span(span_start, span_end)
        if not tok_ids:
            label_probs.append(0.0)
            continue
        probs = [token_probs[i] for i in tok_ids if i < len(token_probs)]
        if probs:
            log_probs = [np.log(max(p, 1e-10)) for p in probs]
            geo_mean = np.exp(np.mean(log_probs))
            label_probs.append(round(float(geo_mean), 4))
        else:
            label_probs.append(0.0)

    return box_probs, label_probs


def assign_probs_to_detections(detections, box_probs, label_probs):
    """
    Assign probabilities to each detection dict.
    Each detection gets:
        'box_prob'   — confidence of the box coordinates
        'label_prob' — confidence of the label
        'prob'       — combined confidence (min of both)
    """
    for i, det in enumerate(detections):
        bp = box_probs[i] if i < len(box_probs) else 0.0
        lp = label_probs[i] if i < len(label_probs) else 0.0
        det["box_prob"] = bp
        det["label_prob"] = lp
        # Combined: geometric mean of box and label probs
        det["prob"] = round(float(np.exp(
            (np.log(max(bp, 1e-10)) + np.log(max(lp, 1e-10))) / 2.0
        )), 4)
    return detections


# ============================================================
# 3. Drawing utilities
# ============================================================

BOX_COLORS = [
    (220, 20, 60),    # Crimson
    (30, 144, 255),   # DodgerBlue
    (0, 201, 87),     # Green
    (255, 165, 0),    # Orange
    (148, 103, 189),  # Purple
    (0, 255, 255),    # Cyan
    (255, 0, 255),    # Magenta
    (255, 215, 0),    # Gold
    (127, 255, 0),    # Chartreuse
    (255, 105, 180),  # HotPink
]


def load_font(size=18):
    """Try to load a TrueType font, fallback to default."""
    for path in [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_detections(image_path, detections, output_path):
    """
    Draw bounding boxes with probability labels on the image.

    Each box label shows: "<label> <probability>%"
    Line width scales with confidence.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    font_main = load_font(18)

    for i, det in enumerate(detections):
        color = BOX_COLORS[i % len(BOX_COLORS)]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label = det["label"]
        prob = det["prob"]

        # Line width: thicker = more confident (2..6 px)
        line_w = max(2, int(prob * 6))

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        # Prepare label text: "person 85%"
        label_text = f"{label} {prob:.0%}"

        # Measure text
        try:
            bbox = draw.textbbox((0, 0), label_text, font=font_main)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 120, 18

        # Label background at top of box
        pad = 4
        bg_y1 = max(0, y1 - th - 2 * pad)
        draw.rectangle(
            [x1, bg_y1, x1 + tw + 2 * pad, bg_y1 + th + 2 * pad],
            fill=color,
        )
        draw.text((x1 + pad, bg_y1 + pad), label_text, fill="white", font=font_main)

        # Corner markers
        cs = 5
        for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            draw.rectangle([cx - cs, cy - cs, cx + cs, cy + cs], fill=color)

    img.save(output_path, quality=95)
    print(f"\nAnnotated image saved to: {output_path}")
    return output_path


# ============================================================
# 4. Console output
# ============================================================

def print_results(detections, raw_answer, token_count, avg_token_prob):
    """Print a formatted results table to the console."""

    print("\n" + "=" * 100)
    print(f"{'#':<4} {'Label':<12} {'Prob':>8} {'BoxConf':>9} {'LblConf':>9} "
          f"{'Bounding Box (px)':<36} {'Level'}")
    print("-" * 100)

    for i, det in enumerate(detections):
        loc = f"({det['x1']:.0f},{det['y1']:.0f})->({det['x2']:.0f},{det['y2']:.0f})"
        p = det["prob"]

        if p >= 0.9:
            level = "Very High"
        elif p >= 0.7:
            level = "High"
        elif p >= 0.5:
            level = "Medium"
        elif p >= 0.3:
            level = "Low"
        else:
            level = "Very Low"

        print(f"{i+1:<4} {det['label']:<12} {p:>7.1%} "
              f"{det['box_prob']:>8.1%} {det['label_prob']:>8.1%}   "
              f"{loc:<36} {level}")

    print("=" * 100)
    print(f"Total detections:    {len(detections)}")
    print(f"Generated tokens:    {token_count}")
    print(f"Avg token prob:      {avg_token_prob:.1%}")
    avg = np.mean([d["prob"] for d in detections]) if detections else 0
    print(f"Avg box prob:        {avg:.1%}")
    print("=" * 100)


# ============================================================
# 5. Main
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_with_token_prob.py <image> "
              "[cat1,cat2,...]")
        print("Example: python detect_with_token_prob.py bus.jpg "
              "person,car,bicycle")
        return

    image_path = sys.argv[1]
    categories = sys.argv[2].split(",") if len(sys.argv) > 2 \
        else ["person", "car", "bicycle"]

    # Get image dimensions
    with Image.open(image_path) as img:
        w, h = img.size

    print("=" * 60)
    print("LocateAnything — Token Generation Probability")
    print("=" * 60)
    print(f"Image:       {image_path} ({w}x{h})")
    print(f"Categories:  {', '.join(categories)}")
    print("-" * 60)

    # ---- Load model ----
    print("\nLoading model ...")
    from locateanything_worker import LocateAnythingWorker
    worker = LocateAnythingWorker("nvidia/LocateAnything-3B")

    image = Image.open(image_path).convert("RGB")

    # ---- Single detection with scores ----
    print("Running detection with token scores ...")
    result = worker.detect_with_scores(image, categories)

    answer = result["answer"]
    token_strs = result["tokens"]
    token_probs = result["probs"]

    # ---- Show raw output and per-token probabilities ----
    print("\n" + "=" * 60)
    print("Raw model output:")
    print("=" * 60)
    print(answer)

    print("\n" + "=" * 60)
    print(f"Per-token probabilities ({len(token_strs)} tokens):")
    print("=" * 60)
    # Print tokens in groups of 10 for readability
    line = ""
    for i, (tok, p) in enumerate(zip(token_strs, token_probs)):
        # Replace newlines for display
        display_tok = tok.replace("\n", "\\n")
        cell = f"{display_tok:>6}({p:.2f})"
        line += cell + "  "
        if (i + 1) % 6 == 0:
            print(f"  {line}")
            line = ""
    if line:
        print(f"  {line}")

    # ---- Map token probs to boxes ----
    print("\nMapping token probabilities to detection boxes ...")
    box_probs, label_probs = compute_box_probs(answer, token_strs, token_probs)

    if not box_probs:
        print("No boxes found in model output. Exiting.")
        return

    # ---- Parse detections and assign probabilities ----
    detections = parse_detections(answer, w, h)
    detections = assign_probs_to_detections(detections, box_probs, label_probs)

    # ---- Console output ----
    avg_tok = np.mean(token_probs) if token_probs else 0
    print_results(detections, answer, len(token_strs), avg_tok)

    # ---- Draw annotated image ----
    output_image = f"output_{image_path}"
    draw_detections(image_path, detections, output_image)

    # ---- Save detailed log ----
    log_path = f"detection_log_{image_path.split('.')[0]}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Detection with Token Generation Probability\n")
        f.write("=" * 60 + "\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Categories: {categories}\n")
        f.write(f"Generated tokens: {len(token_strs)}\n")
        f.write(f"Average token prob: {avg_tok:.4f}\n\n")
        f.write("Raw output:\n")
        f.write(answer + "\n\n")
        f.write("Per-token probabilities:\n")
        for i, (tok, p) in enumerate(zip(token_strs, token_probs)):
            f.write(f"  [{i:3d}] {tok:>8}  prob={p:.4f}\n")
        f.write("\nDetection results:\n")
        for i, det in enumerate(detections):
            f.write(f"  [{i+1}] {det['label']:>10}  "
                    f"prob={det['prob']:.4f}  "
                    f"box_conf={det['box_prob']:.4f}  "
                    f"lbl_conf={det['label_prob']:.4f}  "
                    f"({det['x1']:.0f},{det['y1']:.0f})"
                    f"->({det['x2']:.0f},{det['y2']:.0f})\n")
    print(f"\nDetailed log saved to: {log_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
