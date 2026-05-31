#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_and_draw.py

Object detection with LocateAnything — per-token generation probability.
Uses an embedding-layer hook to inject vision features into the LLM
without triggering numpy dtype or dual-argument errors.

Usage:
    python detect_and_draw.py <image> [cat1,cat2,...]
    python detect_and_draw.py bus.jpg person,car,bicycle
"""

import sys
import os
import re
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

print("[INIT] Starting...", flush=True)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "offline"


class LocateAnythingWorker:

    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        from transformers import AutoModel, AutoTokenizer, AutoProcessor

        print("[Worker] Loading tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True)

        print("[Worker] Loading processor...", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True)

        print("[Worker] Loading model...", flush=True)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True,
            local_files_only=True).to(device).eval()
        print("[Worker] Done.", flush=True)

    def _extract_vision(self, pixel_values, image_grid_hws):
        """Extract and project image features (same as generate)."""
        grid = image_grid_hws
        if isinstance(grid, np.ndarray):
            grid = torch.from_numpy(grid).to(
                pixel_values.device, dtype=torch.int32)
        vit_embeds = self.model.extract_feature(pixel_values, grid)
        if isinstance(vit_embeds, (list, tuple)):
            vit_embeds = torch.cat(vit_embeds, dim=0)
        vit_embeds = self.model.mlp1(vit_embeds)
        return vit_embeds

    def _get_logits(self, full_ids, vit_embeds):
        """
        Run the language model on the full sequence (prompt + answer)
        and return logits for every position.

        Uses a forward hook on the embedding layer to replace image
        token embeddings with vision features.  This avoids both the
        numpy dtype crash and the dual-argument error.
        """
        llm = self.model.language_model
        image_token_id = self.model.config.image_token_index
        embed_layer = llm.get_input_embeddings()

        is_img = (full_ids == image_token_id)[0]

        def hook_fn(module, inp, output):
            out = output.clone()
            out[0, is_img] = vit_embeds[:is_img.sum()]
            return out

        hook = embed_layer.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                outputs = llm(input_ids=full_ids)
        finally:
            hook.remove()

        return outputs.logits

    @torch.no_grad()
    def detect_with_scores(self, image, categories, **kwargs):
        """
        Detect objects AND compute per-token generation probabilities.

        Steps:
          1. generate() -> answer text
          2. extract vision features
          3. encode answer separately -> answer_ids
          4. concatenate prompt_ids + answer_ids -> full_ids
          5. run llm(input_ids=full_ids) with embedding hook
          6. for each generated token, compute softmax probability
        """
        cats = "</c>".join(categories)
        question = (f"Locate all the instances that matches "
                    f"the following description: {cats}.")

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        prompt_text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[prompt_text], images=images, videos=videos,
            return_tensors="pt").to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        image_grid_hws = inputs.get("image_grid_hws", None)
        prompt_len = input_ids.shape[1]

        # Step 1: generate
        print("  [1/5] Generating...", flush=True)
        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=kwargs.get("max_new_tokens", 2048),
            use_cache=True,
            generation_mode=kwargs.get("generation_mode", "hybrid"),
            temperature=kwargs.get("temperature", 0.7),
            do_sample=True, top_p=0.9, repetition_penalty=1.1,
            verbose=kwargs.get("verbose", False),
        )
        answer = response[0] if isinstance(response, tuple) else response

        # Step 2: tokenize answer separately
        answer_ids = self.tokenizer.encode(
            answer, add_special_tokens=False,
            return_tensors="pt").to(self.device)
        answer_len = answer_ids.shape[1]
        print(f"  [2/5] prompt={prompt_len} + answer={answer_len}"
              f" = {prompt_len + answer_len}", flush=True)

        # Step 3: extract vision features
        print("  [3/5] Extracting vision...", flush=True)
        vit_embeds = self._extract_vision(pixel_values, image_grid_hws)

        # Step 4: build full_ids and run LLM
        full_ids = torch.cat([input_ids, answer_ids], dim=1)
        print(f"  [4/5] LLM forward ({full_ids.shape[1]} tokens)...",
              flush=True)
        logits = self._get_logits(full_ids, vit_embeds)
        print(f"       logits shape: {logits.shape}", flush=True)

        # Step 5: per-token probabilities
        tokens = []
        probs = []
        for i in range(answer_len):
            pos = prompt_len - 1 + i
            if pos >= logits.shape[1]:
                break
            tok_logits = logits[0, pos].float()
            tok_probs = torch.softmax(tok_logits, dim=-1)
            tok_id = answer_ids[0, i].item()
            tok_prob = tok_probs[tok_id].item()
            tok_str = self.tokenizer.decode([tok_id])
            tokens.append(tok_str)
            probs.append(round(tok_prob, 5))

        n_pos = sum(1 for p in probs if p > 0)
        print(f"  [5/5] {len(tokens)} tokens, "
              f"{n_pos} with prob>0", flush=True)

        return {"answer": answer, "tokens": tokens, "probs": probs}

    def detect(self, image, categories, **kwargs):
        cats = "</c>".join(categories)
        prompt = (f"Locate all the instances that matches "
                  f"the following description: {cats}.")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.py_apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = self.processor.process_vision_info(msgs)
        inp = self.processor(text=[text], images=imgs, videos=vids,
                             return_tensors="pt").to(self.device)
        out = self.model.generate(
            pixel_values=inp["pixel_values"].to(self.dtype),
            input_ids=inp["input_ids"],
            attention_mask=inp["attention_mask"],
            image_grid_hws=inp.get("image_grid_hws"),
            tokenizer=self.tokenizer,
            max_new_tokens=2048, use_cache=True,
            generation_mode="hybrid", temperature=0.7,
            do_sample=True, top_p=0.9, repetition_penalty=1.1,
            verbose=False,
        )
        return {"answer": out[0] if isinstance(out, tuple) else out}


# ============================================================
# Parse, score, draw
# ============================================================
def parse_detections(answer, w, h):
    dets, cur = [], None
    evts = []
    for m in re.finditer(r"<ref>(.*?)</ref>", answer):
        evts.append((m.start(), "ref", m.group(1)))
    for m in re.finditer(
            r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        evts.append((m.start(), "box", [int(g) for g in m.groups()]))
    evts.sort()
    for _, k, v in evts:
        if k == "ref":
            cur = v
        elif k == "box" and cur:
            x1, y1, x2, y2 = v
            dets.append({"label": cur,
                         "x1": x1/1000*w, "y1": y1/1000*h,
                         "x2": x2/1000*w, "y2": y2/1000*h})
    return dets


def compute_box_probs(answer, toks, prs):
    c2t = []
    for i, t in enumerate(toks):
        c2t.extend([i] * len(t))
    cs = "".join(toks)

    def tids(s, e):
        return {c2t[p] for p in range(s, min(e, len(c2t)))}

    def gm(ps):
        return float(np.exp(np.mean(
            [np.log(max(p, 1e-10)) for p in ps]))) if ps else 0.0

    bp, lp = [], []
    for m in re.finditer(r"<box><\d+><\d+><\d+><\d+></box>", cs):
        ps = [prs[i] for i in tids(m.start(), m.end()) if i < len(prs)]
        bp.append(round(gm(ps), 4))
    for m in re.finditer(r"<ref>(.*?)</ref>", cs):
        ps = [prs[i] for i in tids(m.start(), m.end()) if i < len(prs)]
        lp.append(round(gm(ps), 4))
    return bp, lp


def assign(dets, bp, lp):
    for i, d in enumerate(dets):
        b = bp[i] if i < len(bp) else 0.0
        l = lp[i] if i < len(lp) else 0.0
        d["box_p"], d["lbl_p"] = b, l
        d["prob"] = (round(float(np.exp(
            (np.log(max(b, 1e-10)) + np.log(max(l, 1e-10))) / 2)),
            4) if b > 0 and l > 0 else max(b, l))
    return dets


COLS = [(220,20,60),(30,144,255),(0,201,87),(255,165,0),
        (148,103,189),(0,255,255),(255,0,255),(255,215,0),
        (127,255,0),(255,105,180)]


def load_font(sz=18):
    for p in ["arial.ttf", "C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_dets(path, dets, out):
    img = Image.open(path).convert("RGB")
    dr = ImageDraw.Draw(img)
    ft = load_font()
    for i, d in enumerate(dets):
        c = COLS[i % len(COLS)]
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        p, lb = d["prob"], d["label"]
        dr.rectangle([x1, y1, x2, y2], outline=c, width=max(2, int(p*6)))
        txt = f"{lb} {p:.0%}"
        try:
            bb = dr.textbbox((0, 0), txt, font=ft)
            tw, th = bb[2]-bb[0], bb[3]-bb[1]
        except Exception:
            tw, th = 120, 18
        bg = max(0, y1 - th - 8)
        dr.rectangle([x1, bg, x1+tw+8, bg+th+8], fill=c)
        dr.text((x1+4, bg+2), txt, fill="white", font=ft)
        for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            dr.rectangle([cx-5, cy-5, cx+5, cy+5], fill=c)
    img.save(out, quality=95)


def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_and_draw.py <image> [cat1,cat2,...]")
        return
    path = sys.argv[1]
    cats = (sys.argv[2].split(",") if len(sys.argv) > 2
            else ["person", "car", "bicycle"])
    with Image.open(path) as im:
        w, h = im.size
    print("=" * 60, flush=True)
    print("LocateAnything - Token Generation Probability", flush=True)
    print("=" * 60, flush=True)
    print(f"  Image: {path} ({w}x{h})", flush=True)
    print(f"  Categories: {', '.join(cats)}", flush=True)
    print("-" * 60, flush=True)

    print("\n[1/5] Loading model...", flush=True)
    wkr = LocateAnythingWorker("nvidia/LocateAnything-3B")
    img = Image.open(path).convert("RGB")

    print("\n[2/5] Detecting...", flush=True)
    res = wkr.detect_with_scores(img, cats, verbose=False)
    ans, toks, prs = res["answer"], res["tokens"], res["probs"]

    print(f"\n[3/5] Raw output:", flush=True)
    print("-" * 50, flush=True)
    print(ans, flush=True)
    print("-" * 50, flush=True)

    has_pr = any(p > 0 for p in prs)
    if has_pr:
        print(f"\n  Token probabilities ({len(toks)} tokens):", flush=True)
        ln = ""
        for i, (t, p) in enumerate(zip(toks, prs)):
            display_tok = t.replace("\n", "\\n")
            ln += f"{display_tok:>6}({p:.2f})  "
            if (i + 1) % 5 == 0:
                print(f"    {ln}", flush=True)
                ln = ""
        if ln:
            print(f"    {ln}", flush=True)

        bp, lp = compute_box_probs(ans, toks, prs)
        dets = assign(parse_detections(ans, w, h), bp, lp)
    else:
        print("\n  (Token prob unavailable - using box-only output)",
              flush=True)
        dets = parse_detections(ans, w, h)
        for d in dets:
            d["box_p"] = d["lbl_p"] = d["prob"] = 0.0

    if not dets:
        print("No detections found!", flush=True)
        return

    # Print results table
    print("\n" + "=" * 95, flush=True)
    print(f"{'#':<4} {'Label':<12} {'Prob':>8} {'Box':>8} {'Lbl':>8}"
          f"  {'Position (px)':<36} {'Level'}", flush=True)
    print("-" * 95, flush=True)
    for i, d in enumerate(dets):
        loc = (f"({d['x1']:.0f},{d['y1']:.0f})->"
               f"({d['x2']:.0f},{d['y2']:.0f})")
        p = d["prob"]
        if p >= 0.9:
            lv = "Very High"
        elif p >= 0.7:
            lv = "High"
        elif p >= 0.5:
            lv = "Medium"
        elif p >= 0.3:
            lv = "Low"
        else:
            lv = "Very Low"
        print(f"{i+1:<4} {d['label']:<12} {p:>7.1%}"
              f"{d['box_p']:>7.1%}{d['lbl_p']:>7.1%}  "
              f"{loc:<36}{lv}", flush=True)
    print("=" * 95, flush=True)
    avg_p = np.mean([d["prob"] for d in dets])
    print(f"Total: {len(dets)}, Average prob: {avg_p:.1%}", flush=True)

    # Draw
    print(f"\n[5/5] Drawing results...", flush=True)
    out = f"output_{path}"
    draw_dets(path, dets, out)
    print(f"  Saved: {out}", flush=True)
    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
