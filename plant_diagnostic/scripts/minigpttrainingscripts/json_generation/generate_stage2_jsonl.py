#!/usr/bin/env python3
"""
Fail-proof MiniGPT-4 stage-2 JSONL generator (always writes something).

- One image -> one record, guaranteed.
- Label = parent folder (drought|frost|healthy|overwatered|root_rot).
- If API succeeds and returns JSON { "report": "<text>" }, we use it.
- If JSON is malformed or any error happens, we write a fallback doctor-grade report.
- No visibility/contradiction/length hard gates.
- Resumable: skips images already present in the output file.
- Temperature is sent only for models that support it (o3 does NOT).

Usage example:
  python scripts/minigpttrainingscripts/json_generation/generate_stage2_jsonl.py \
    --img-root data/train \
    --out datasets/strawberry_stage2_train_v3.jsonl \
    --model o3 \
    --sleep 0.6
"""

import argparse, base64, json, logging, os, sys, time
from pathlib import Path
from typing import Set, Optional

VALID_LABELS = {"drought","frost","healthy","overwatered","root_rot"}
DEFAULT_MODEL = os.getenv("OPENAI_VISION_MODEL", "o3")
MAX_RETRIES = 2
BACKOFF = 1.25
DEFAULT_SLEEP = 0.6

def setup_logger(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)]
    )

def to_data_url(p: Path) -> str:
    mime = {
        ".jpg":"image/jpeg",".jpeg":"image/jpeg",".png":"image/png",
        ".bmp":"image/bmp",".webp":"image/webp",".gif":"image/gif",
        ".tif":"image/tiff",".tiff":"image/tiff"
    }.get(p.suffix.lower(), "image/jpeg")
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def load_done_images(jsonl_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not jsonl_path.exists(): return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                img = obj.get("image")
                if isinstance(img, str): done.add(img)
            except Exception:
                continue
    return done

def build_system_prompt() -> str:
    return """You are a strawberry plant diagnostician.

Rules:
1) The diagnosis label is PRE-SET and MUST be used exactly as given.
2) Base EVERYTHING ONLY on what is VISIBLE in the image (no hidden parts unless visible).
3) Write ONE professional report as a single paragraph (aim ~6–9 sentences).
4) Structure (in prose): start with 'Diagnosis: <label>.' then (a) 3–6 concrete visual clues (color/texture/distribution),
   (b) likely cause tied to those clues, (c) 2–3 actionable steps with measurements (cm/inches, frequency/volume),
   (d) 2–3 visible monitoring checks for the next 7 days.
5) Prefer irrigation depth/timing, airflow, mulch, shading, drainage, sanitation; avoid chemicals unless visuals clearly warrant them.
6) Return STRICT JSON exactly: {"report": "<text>"} — no markdown, no extra keys.
"""

def build_user_payload(label: str) -> str:
    return json.dumps({"label": label, "note": "Use the fixed label; visual-only; one broad, detailed report."}, ensure_ascii=False)

def supports_temperature(model: str) -> bool:
    m = (model or "").lower()
    # o3 family in chat-completions typically ignores/400s temperature
    return not m.startswith("o3")

def safe_parse_report(txt: str) -> Optional[str]:
    txt = (txt or "").strip()
    if not txt: return None
    try:
        obj = json.loads(txt)
    except Exception:
        s, e = txt.find("{"), txt.rfind("}")
        if s >= 0 and e > s:
            try:
                obj = json.loads(txt[s:e+1])
            except Exception:
                return None
        else:
            return None
    rep = obj.get("report") if isinstance(obj, dict) else None
    return rep.strip() if isinstance(rep, str) and rep.strip() else None

def fallback_report(label: str) -> str:
    if label == "drought":
        return ("Diagnosis: drought. Leaves show dry, curled margins with a dull surface consistent with moisture stress. "
                "Clues suggest shallow or irregular irrigation during drying weather. Deep-soak to the root zone and avoid frequent shallow sips; add 5–7 cm mulch while keeping it off crowns. "
                "Next 7 days: check morning leaf turgor, surface dry-back time after watering, and evenness of new leaf expansion.")
    if label == "overwatered":
        return ("Diagnosis: overwatered. Media near the base appears persistently damp with limp foliage suggestive of low root oxygen. "
                "Reduce frequency while watering more deeply; allow the top 2–3 cm to dry between events; fork/aerate the surface and pull mulch 3–5 cm back from the crown. "
                "Next 7 days: faster surface dry-back, firmer morning leaves, no standing water after irrigation.")
    if label == "frost":
        return ("Diagnosis: frost. Tissue shows dulling and localized discoloration consistent with cold exposure. "
                "Protect plants on cold nights with row cover; avoid handling brittle tissue until temperatures moderate. "
                "Next 7 days: watch for glassy/dark petal centers and any misshapen new fruit.")
    if label == "root_rot":
        return ("Diagnosis: root_rot. Pattern of wilt under otherwise moist conditions points to impaired roots. "
                "Improve drainage, avoid saturation; in containers, confirm open drain holes. "
                "Next 7 days: improved turgor after irrigation control and removal of severely declined tissue.")
    # healthy
    return ("Diagnosis: healthy. Foliage is evenly colored with intact margins and fruit development is uniform for the stage. "
            "Maintain steady irrigation (moist, not wet) and avoid wetting fruit late in the day. "
            "Next 7 days: monitor for new spotting, afternoon wilt, or surface fuzz in humid periods.")

def call_openai_image_report(client, model: str, image: Path, label: str, temperature: Optional[float]):
    system = build_system_prompt()
    user_payload = build_user_payload(label)
    data_url = to_data_url(image)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role":"system","content": system},
                    {"role":"user","content":[
                        {"type":"text","text": user_payload},
                        {"type":"image_url","image_url":{"url": data_url}}
                    ]}
                ],
            }
            if temperature is not None and supports_temperature(model):
                kwargs["temperature"] = temperature if attempt == 1 else min(0.8, (temperature or 0.7) + 0.2)

            resp = client.chat.completions.create(**kwargs)
            txt = resp.choices[0].message.content
            rep = safe_parse_report(txt)
            if rep: return rep.strip()
            raise ValueError("Model returned no JSON 'report'")
        except Exception as e:
            msg = str(e)
            # Non-retryable API misconfig (e.g., unsupported temperature) → break fast
            if "invalid_request_error" in msg or "unsupported_value" in msg or "400" in msg:
                logging.error("Non-retryable error for %s: %s", image.name, msg)
                break
            logging.warning("OpenAI call failed (%d/%d) for %s: %s", attempt, MAX_RETRIES, image.name, msg)
            if attempt == MAX_RETRIES:
                break
            time.sleep(BACKOFF ** (attempt - 1))
    return None  # signal for fallback

def iter_images(root: Path):
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = cls_dir.name
        if label not in VALID_LABELS:
            logging.warning("Skipping non-labeled dir: %s", cls_dir)
            continue
        for img in sorted(cls_dir.rglob("*")):
            if img.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp",".gif",".tif",".tiff"):
                yield img, label

def main():
    ap = argparse.ArgumentParser("Always-write generator for MiniGPT-4 stage-2")
    ap.add_argument("--img-root", type=Path, default=Path("data/train"))
    ap.add_argument("--out", type=Path, default=Path("datasets/strawberry_stage2_train_v3.jsonl"))
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=None, help="Ignored for models that don't support it (e.g., o3).")
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    ap.add_argument("--log-file", type=Path, default=Path("logs/generate_stage2_reports.log"))
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")
    try:
        from openai import OpenAI
    except Exception:
        sys.exit("pip install openai")

    client = OpenAI()
    setup_logger(args.log_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_images(args.out)
    logging.info("Resuming: %d images already present in %s", len(done), args.out)

    processed = written = fb_count = 0
    with args.out.open("a", encoding="utf-8") as out:
        for img, label in iter_images(args.img_root):
            processed += 1
            img_str = str(img)
            if img_str in done:
                continue
            report = call_openai_image_report(client, args.model, img, label, args.temperature)
            used_fallback = False
            if not report:
                report = fallback_report(label)
                used_fallback = True
                fb_count += 1

            # Ensure label prefix; if missing, prepend it.
            prefix = f"Diagnosis: {label}"
            if not report.strip().startswith(prefix):
                report = f"{prefix}. " + report.strip().split("Diagnosis:", 1)[-1].strip()

            rec = {
                "image": img_str,
                "conversations": [
                    {"from":"human", "value":"<Img><ImageHere></Img>"},
                    {"from":"assistant", "value": report}
                ]
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            tag = "FALLBACK" if used_fallback else "OK"
            logging.info("[%s] %s -> 1 report", tag, img.relative_to(args.img_root))
            time.sleep(args.sleep)

    logging.info("DONE. Wrote %d entries across %d images -> %s (fallbacks used: %d)",
                 written, processed, args.out, fb_count)

if __name__ == "__main__":
    main()
