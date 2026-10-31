#!/usr/bin/env python3
"""stemsep - stem separation with a low-distortion discipline.

Wraps MSST (ZFTurbo/Music-Source-Separation-Training) with the things that
actually decide whether a stem sounds damaged:

  * one resample, never two          - soxr in, soxr out, model run at its native rate
  * channel math before neural nets  - 5.1 centre is already dialogue; don't guess it
  * residual subtraction             - the last stem is mix minus the rest, so the set nulls
  * a null test on every run         - prints how far the stems sum back to the original
  * ensemble in the waveform domain  - avg_wave, the only method that beat the others in
                                       ZFTurbo's own sweep

Docs: README.md
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("STEMSEP_MODELS", HERE / "models"))
REGISTRY = json.loads((HERE / "models.json").read_text())
REGISTRY.pop("_comment", None)

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
SOXR = "aresample=resampler=soxr:precision=28"


# ---------------------------------------------------------------- output


def say(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def head(msg: str) -> None:
    say(f"\033[1m{msg}\033[0m")


def die(msg: str) -> "None":
    say(f"\033[31merror:\033[0m {msg}")
    raise SystemExit(1)


# ---------------------------------------------------------------- audio io


def probe(path: Path) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels,channel_layout,codec_name,sample_fmt:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        die(f"ffprobe failed on {path}: {out.stderr.strip()}")
    data = json.loads(out.stdout)
    if not data.get("streams"):
        die(f"no audio stream in {path}")
    s = data["streams"][0]
    return {
        "sr": int(s["sample_rate"]),
        "channels": int(s["channels"]),
        "layout": s.get("channel_layout", f"{s['channels']}ch"),
        "codec": s.get("codec_name", "?"),
        "fmt": s.get("sample_fmt", "?"),
        "duration": float(data.get("format", {}).get("duration") or 0.0),
    }


def decode(path: Path, sr: int | None = None, pan: str | None = None,
           channels: int | None = None):
    """Decode to float32 (channels, samples). Resamples at most once, with soxr.

    `channels` must be given whenever `pan` is - the pan spec decides the output
    layout and there is no reliable way to infer it back from the string."""
    import numpy as np

    if channels is None:
        if pan:
            die("internal: decode(pan=...) requires an explicit channels=")
        channels = probe(path)["channels"]

    filters = []
    if pan:
        filters.append(pan)
    if sr:
        filters.append(f"{SOXR}:osr={sr}")
    cmd = [FFMPEG, "-v", "error", "-i", str(path), "-map", "0:a:0"]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-"]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        die(f"ffmpeg decode failed: {proc.stderr.decode(errors='replace').strip()}")

    raw = np.frombuffer(proc.stdout, dtype="<f4")
    if len(raw) % channels:
        raw = raw[: len(raw) - (len(raw) % channels)]
    return np.ascontiguousarray(raw.reshape(-1, channels).T.astype("float32"))


def to_stereo(path: Path, sr: int, info: dict):
    """Every separation model here takes 2 channels. Fold anything else with an
    explicit downmix rather than letting a decoder pick one."""
    ch = info["channels"]
    if ch == 2:
        return decode(path, sr=sr, channels=2)
    if ch == 1:
        return decode(path, sr=sr, pan="pan=stereo|c0=c0|c1=c0", channels=2)
    return decode(path, sr=sr, channels=2,
                  pan="pan=stereo|FL<FL+0.707*FC+0.707*BL|FR<FR+0.707*FC+0.707*BR")


def write(path: Path, audio, sr: int, out_sr: int, fmt: str = "wav") -> Path:
    """Write (channels, samples). Resamples back to out_sr once, if needed."""
    import numpy as np
    import soundfile as sf

    path = path.with_suffix("." + fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(audio.T.astype("float32"))

    if sr == out_sr:
        sf.write(str(path), data, sr, subtype="PCM_24" if fmt == "wav" else "PCM_24")
        return path

    # one resample back, through the same soxr path the input took
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(str(tmp_path), data, sr, subtype="FLOAT")
        cmd = [FFMPEG, "-v", "error", "-y", "-i", str(tmp_path),
               "-af", f"{SOXR}:osr={out_sr}", "-c:a",
               "pcm_s24le" if fmt == "wav" else "flac", str(path)]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            die(f"ffmpeg encode failed: {proc.stderr.decode(errors='replace').strip()}")
    finally:
        tmp_path.unlink(missing_ok=True)
    return path


# ---------------------------------------------------------------- models


def entry(name: str) -> dict:
    if name not in REGISTRY:
        die(f"unknown model '{name}'. Run: stemsep models")
    return REGISTRY[name]


def fetch(url: str, dest: Path, label: str) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    say(f"  downloading {label} ...")
    part = dest.with_suffix(dest.suffix + ".part")
    rc = subprocess.call(["curl", "-fL", "--progress-bar", "-C", "-", "-o", str(part), url])
    if rc != 0 or not part.exists():
        part.unlink(missing_ok=True)
        die(f"download failed: {url}")
    part.rename(dest)


def normalize_ckpt(path: Path) -> None:
    """Some published weights (Bandit v2 on Zenodo) are raw PyTorch Lightning
    checkpoints: the real tensors sit under state_dict behind a 'model.' prefix,
    alongside optimizer state and loss-handler buffers MSST does not want.
    Unwrap once, in place, so every later load is a plain state_dict."""
    marker = path.with_name(path.name + ".normalized")
    if marker.exists():
        return
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "state_dict" in ck:
        sd = ck["state_dict"]
        keys = [k for k in sd if k.startswith("model.")]
        if keys and len(keys) >= len(sd) * 0.5:
            sd = {k[len("model."):]: sd[k] for k in keys}
        torch.save(sd, str(path))
        say("  normalised Lightning checkpoint -> plain state_dict")
    marker.touch()


def ensure(name: str, lang: str = "multi") -> tuple[Path, Path, dict]:
    e = entry(name)
    slug = name if lang == "multi" or "langs" not in e else f"{name}-{lang}"
    d = MODELS_DIR / slug
    cfg, ckpt = d / "config.yaml", d / "model.ckpt"

    url = e["ckpt"]
    if "langs" in e:
        if lang not in e["langs"]:
            die(f"lang '{lang}' not available for {name}; have: {', '.join(e['langs'])}")
        url = f"https://zenodo.org/api/records/12701995/files/{e['langs'][lang]}/content"

    if not cfg.exists() or not ckpt.exists():
        head(f"model {slug} ({e.get('size', '?')})")
    fetch(e["cfg"], cfg, f"{slug} config")
    fetch(url, ckpt, f"{slug} weights")
    normalize_ckpt(ckpt)
    return cfg, ckpt, e


def separate(name: str, audio, sr: int, lang: str = "multi",
             tta: bool = False, device: int = 0) -> dict:
    """Run one model over an in-memory (channels, samples) array."""
    import msst

    cfg, ckpt, e = ensure(name, lang)
    if sr != e["sr"]:
        die(f"internal: {name} wants {e['sr']} Hz, got {sr}")

    # A target-instrument model emits only its target. extract_instrumental asks
    # MSST for the complement, which it builds by subtraction - so the pair nulls.
    say(f"  -> {name} [{e['type']}] {'+tta ' if tta else ''}on {e['sr']/1000:g}k")
    sep = msst.Separator(
        model_type=e["type"], config_path=cfg, checkpoint_path=ckpt,
        device_ids=device, use_tta=tta, detailed_progress=False,
        extract_instrumental=bool(e.get("target_only")),
    )
    try:
        return sep.separate(audio, sample_rate=sr, channels_first=True)
    finally:
        sep.close()


def ensemble(results: list[dict], algorithm: str = "avg_wave") -> dict:
    """Combine several models' outputs per stem, in the waveform domain."""
    import numpy as np
    from msst._ensemble import average_waveforms

    keys = set(results[0])
    for r in results[1:]:
        keys &= set(r)
    out = {}
    for k in sorted(keys):
        n = min(r[k].shape[-1] for r in results)
        stack = np.stack([r[k][..., :n] for r in results], axis=0)
        out[k] = average_waveforms(stack, np.ones(len(results)), algorithm)
    return out


# ---------------------------------------------------------------- quality


def null_test(mix, stems: dict) -> float:
    """dB by which the summed stems fail to reconstruct the mix. Lower is better."""
    import numpy as np

    n = min(mix.shape[-1], *(s.shape[-1] for s in stems.values()))
    total = np.zeros_like(mix[..., :n])
    for s in stems.values():
        total += s[..., :n]
    resid = mix[..., :n] - total
    rms_m = float(np.sqrt(np.mean(mix[..., :n] ** 2))) or 1e-12
    rms_r = float(np.sqrt(np.mean(resid ** 2)))
    return 20.0 * np.log10(max(rms_r, 1e-12) / rms_m)


EXACT_DB = -80.0   # below this, the stems are a true decomposition of the mix


def report_null(mix, stems: dict) -> float:
    db = null_test(mix, stems)
    if db <= EXACT_DB:
        say(f"  null test: {db:+.1f} dB  (exact - the stems sum back to the source)")
    elif db < -30:
        say(f"  null test: {db:+.1f} dB  (sums back cleanly)")
    else:
        say(f"  null test: {db:+.1f} dB  (energy unaccounted for - the model is discarding "
            f"signal; --residual STEM forces a perfect sum)")
    return db


def settle(mix, stems: dict, want: str | None, default: str) -> dict:
    """Apply residual subtraction only if the stems don't already sum back.

    An explicit --residual NAME always wins; 'auto' defers to measurement, so a
    model that already emits a complement is not needlessly rebuilt."""
    if not want:
        return stems
    if want == "auto":
        if null_test(mix, stems) <= EXACT_DB:
            return stems                       # already a true decomposition
        want = default if default in stems else sorted(stems)[-1]
    if want not in stems:
        die(f"--residual {want}: model produced {', '.join(sorted(stems))}")
    say(f"  residual: '{want}' rebuilt as source minus the rest")
    return make_residual(mix, stems, want)


def make_residual(mix, stems: dict, target: str) -> dict:
    """Rebuild `target` as mix minus the others, so the set nulls perfectly."""
    import numpy as np

    n = min(mix.shape[-1], *(s.shape[-1] for s in stems.values()))
    others = np.zeros_like(mix[..., :n])
    for k, v in stems.items():
        if k != target:
            others += v[..., :n]
    out = {k: v[..., :n] for k, v in stems.items()}
    out[target] = mix[..., :n] - others
    return out


# ---------------------------------------------------------------- recipes

# Never ensemble across models whose stems mean different things: SW carves guitar
# and piano out of 'other', a 4-stem model leaves them in it, and averaging the two
# 'other's would half-attenuate every guitar. Each row's models must agree on stems.
MUSIC = {
    "fast": {2: ["bsroformer"], 4: ["htdemucs"], 6: ["sw"]},
    "good": {2: ["leap"], 4: ["scnet-xl"], 6: ["sw"]},
    "best": {2: ["leap", "melband"], 4: ["scnet-xl", "htdemucs"], 6: ["sw"]},
}


def cmd_music(a) -> None:
    src = Path(a.input).expanduser().resolve()
    if not src.exists():
        die(f"no such file: {src}")
    info = probe(src)
    names = MUSIC[a.quality][a.stems]
    model_sr = entry(names[0])["sr"]
    out_sr = info["sr"] if a.keep_rate else model_sr

    head(f"music: {src.name}")
    say(f"  {info['layout']} {info['sr']} Hz {info['codec']} {info['duration']:.1f}s")
    if info["sr"] != model_sr:
        say(f"  resampling {info['sr']} -> {model_sr} (soxr, once) "
            f"{'and back on write' if a.keep_rate else ''}")

    if info["channels"] != 2:
        say(f"  folding {info['layout']} to stereo for the model")
    mix = to_stereo(src, model_sr, info)
    results = [separate(n, mix, model_sr, tta=a.tta, device=a.device) for n in names]
    stems = results[0] if len(results) == 1 else ensemble(results)
    if len(results) > 1:
        say(f"  ensembled {len(results)} models (avg_wave)")

    stems = settle(mix, stems, a.residual, "other")
    report_null(mix, stems)
    outdir = Path(a.output or src.parent / f"{src.stem}.stems").expanduser()
    for k, v in sorted(stems.items()):
        p = write(outdir / k, v, model_sr, out_sr, a.format)
        say(f"  wrote {p}")
    head(f"done -> {outdir}")


def cmd_cinema(a) -> None:
    import numpy as np

    src = Path(a.input).expanduser().resolve()
    if not src.exists():
        die(f"no such file: {src}")
    info = probe(src)
    name = a.model
    e = entry(name)
    model_sr = e["sr"]
    out_sr = info["sr"] if a.keep_rate else model_sr

    head(f"cinema: {src.name}")
    say(f"  {info['layout']} {info['sr']} Hz {info['codec']} {info['duration']:.1f}s")
    if info["sr"] != model_sr:
        say(f"  resampling {info['sr']} -> {model_sr} (soxr, once)")
    else:
        say(f"  input is already {model_sr} Hz - no resampling")

    outdir = Path(a.output or src.parent / f"{src.stem}.stems").expanduser()
    multichannel = info["channels"] >= 6 and not a.no_channel_math

    if multichannel:
        say(f"  {info['channels']}ch source: taking centre + LFE by channel math "
            f"(lossless), ML on the L/R/Ls/Rs fold only")
        centre = decode(src, sr=model_sr, pan="pan=mono|c0=FC", channels=1)
        lfe = decode(src, sr=model_sr, pan="pan=mono|c0=LFE", channels=1)
        bed = decode(src, sr=model_sr, channels=2,
                     pan="pan=stereo|c0=FL+0.707*BL|c1=FR+0.707*BR")
        p = write(outdir / "dialogue_centre", centre, model_sr, out_sr, a.format)
        say(f"  wrote {p}   <- untouched C channel, zero processing loss")
        p = write(outdir / "lfe", lfe, model_sr, out_sr, a.format)
        say(f"  wrote {p}")
        mix = bed
    else:
        if info["channels"] >= 6:
            say("  --no-channel-math: folding everything to stereo before the model")
        mix = to_stereo(src, model_sr, info)

    stems = separate(name, mix, model_sr, lang=a.lang, tta=a.tta, device=a.device)

    stems = settle(mix, stems, a.residual, "sfx" if "sfx" in stems else "effects")
    report_null(mix, stems)
    suffix = "_bed" if multichannel else ""
    for k, v in sorted(stems.items()):
        p = write(outdir / f"{k}{suffix}", v, model_sr, out_sr, a.format)
        say(f"  wrote {p}")

    if a.submix and "music" in stems:
        head("sub-splitting the music stem")
        mus = stems["music"]
        msr = entry("scnet-xl")["sr"]
        if msr != model_sr:
            say(f"  music model is {msr} Hz; resampling the music stem once")
            import librosa
            mus = np.ascontiguousarray(
                librosa.resample(mus, orig_sr=model_sr, target_sr=msr, res_type="soxr_vhq"))
        sub = separate("scnet-xl", mus, msr, tta=a.tta, device=a.device)
        for k, v in sorted(sub.items()):
            p = write(outdir / f"music_{k}", v, msr, out_sr, a.format)
            say(f"  wrote {p}")

    head(f"done -> {outdir}")
    if multichannel:
        say("  dialogue = dialogue_centre (+ speech_bed if dialogue leaked into the surrounds)")


def cmd_repair(a) -> None:
    src = Path(a.input).expanduser().resolve()
    if not src.exists():
        die(f"no such file: {src}")
    info = probe(src)
    e = entry(a.model)
    model_sr = e["sr"]
    out_sr = info["sr"] if a.keep_rate else model_sr

    head(f"repair [{a.model}]: {src.name}")
    say(f"  {info['layout']} {info['sr']} Hz {info['duration']:.1f}s")
    mix = to_stereo(src, model_sr, info)
    stems = separate(a.model, mix, model_sr, tta=a.tta, device=a.device)

    # A repair model emits only what it kept. Write what it took away too - that
    # residual is the only honest way to hear whether it overreached.
    if len(stems) == 1 and a.model != "apollo":
        stems = make_residual(mix, dict(stems, removed=mix), "removed")
        say("  'removed' = what the model took out (audition this before trusting the pass)")
    report_null(mix, stems)

    outdir = Path(a.output or src.parent / f"{src.stem}.{a.model}").expanduser()
    for k, v in sorted(stems.items()):
        p = write(outdir / k, v, model_sr, out_sr, a.format)
        say(f"  wrote {p}")
    head(f"done -> {outdir}")


def cmd_models(a) -> None:
    if a.pull:
        for n in a.pull:
            ensure(n, a.lang)
        head("pulled")
        return
    say(f"models cached in {MODELS_DIR}\n")
    for fam in ("music", "cinema", "repair"):
        head(fam)
        for n, e in REGISTRY.items():
            if e["family"] != fam:
                continue
            d = MODELS_DIR / n
            got = "*" if (d / "model.ckpt").exists() else " "
            sdr = f"{e['sdr']:.2f}" if e.get("sdr") else "  - "
            say(f"  {got} {n:<14} {sdr}  {e['sr']//1000}k  "
                f"{'/'.join(e['stems']):<24} {e['desc']}")
        say()
    say("  * = downloaded.  stemsep models --pull NAME   to fetch one ahead of time.")


def cmd_info(a) -> None:
    src = Path(a.input).expanduser().resolve()
    info = probe(src)
    head(src.name)
    for k, v in info.items():
        say(f"  {k:<10} {v}")
    if info["channels"] >= 6:
        say("\n  6+ channels: 'stemsep cinema' will take C and LFE by channel math and "
            "only run the model on the L/R/Ls/Rs fold.")
    if info["sr"] == 48000:
        say("  48 kHz: use 'cinema' (bandit-v2 is native 48k). 'music' models are 44.1k "
            "and will cost you one resample each way.")


# ---------------------------------------------------------------- cli


def main() -> None:
    p = argparse.ArgumentParser(
        prog="stemsep", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("input")
        sp.add_argument("-o", "--output", help="output directory")
        sp.add_argument("-f", "--format", choices=["wav", "flac"], default="wav")
        sp.add_argument("--tta", action="store_true",
                        help="test-time augmentation: ~3x slower, small quality gain")
        sp.add_argument("--keep-rate", action="store_true", default=True,
                        help="write at the source sample rate (default)")
        sp.add_argument("--model-rate", dest="keep_rate", action="store_false",
                        help="write at the model's rate instead; avoids the resample back")
        sp.add_argument("--device", type=int, default=0)

    m = sub.add_parser("music", help="split music into stems")
    common(m)
    m.add_argument("-q", "--quality", choices=["fast", "good", "best"], default="good")
    m.add_argument("-s", "--stems", type=int, choices=[2, 4, 6], default=2,
                   help="2=vocals/instrumental, 4=bass/drums/vocals/other, "
                        "6=+guitar/piano (SW)")
    m.add_argument("--residual", nargs="?", const="auto", default="auto",
                   help="rebuild this stem as mix-minus-the-rest (default: auto)")
    m.add_argument("--no-residual", dest="residual", action="store_const", const=None)
    m.set_defaults(fn=cmd_music)

    c = sub.add_parser("cinema", help="split film audio into speech / music / sfx")
    common(c)
    c.add_argument("--model", default="bandit-v2", choices=["bandit-v2", "bandit-plus"])
    c.add_argument("--lang", default="multi",
                   choices=["multi", "eng", "deu", "fra", "spa", "cmn", "fao"],
                   help="bandit-v2 language-specific weights")
    c.add_argument("--submix", action="store_true",
                   help="also split the music stem into bass/drums/vocals/other")
    c.add_argument("--no-channel-math", action="store_true",
                   help="do NOT take the centre channel directly; fold 5.1 to stereo first")
    c.add_argument("--residual", nargs="?", const="auto", default=None,
                   help="rebuild this stem as mix-minus-the-rest")
    c.set_defaults(fn=cmd_cinema)

    r = sub.add_parser("repair", help="denoise / dereverb / de-crowd / Apollo a finished stem")
    common(r)
    r.add_argument("-m", "--model", default="denoise",
                   choices=["denoise", "denoise-aggr", "dereverb", "crowd", "apollo"])
    r.set_defaults(fn=cmd_repair)

    ls = sub.add_parser("models", help="list or pre-fetch models")
    ls.add_argument("--pull", nargs="+", metavar="NAME")
    ls.add_argument("--lang", default="multi")
    ls.set_defaults(fn=cmd_models)

    i = sub.add_parser("info", help="probe a file and say what to run on it")
    i.add_argument("input")
    i.set_defaults(fn=cmd_info)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        say("\ninterrupted")
        raise SystemExit(130)
