# stemsep — stem separation for music and cinema

`stemsep` wraps **MSST** ([ZFTurbo/Music-Source-Separation-Training]) with the handling that
actually decides whether a stem sounds damaged.

```bash
stemsep info   <file>                  # probe, and say which command fits
stemsep music  <file> [-q best] [-s 4] # vocals / bass / drums / other
stemsep cinema <file> [--submix]       # speech / music / sfx
stemsep repair <file> -m denoise       # denoise, dereverb, de-crowd, Apollo
stemsep models [--pull NAME]           # what's available, what's cached
```

Outputs go to `<input>.stems/` next to the source unless `-o` says otherwise; 24-bit WAV at
the **source** sample rate by default (`-f flac`, `--model-rate` to change that).

## Install

Needs Linux (or WSL2), an NVIDIA GPU, [uv](https://docs.astral.sh/uv/), `curl`, and an
`ffmpeg` built with libsoxr (`ffmpeg -version | grep soxr`).

```bash
git clone https://github.com/badwolf-ro/stemsep.git && cd stemsep
./setup.sh                        # python 3.12 venv + torch 2.11 + MSST, pinned in requirements.lock
./stemsep models                  # list models; weights download on first use into models/
ln -s "$PWD/stemsep" ~/.local/bin/stemsep   # optional: put it on PATH
```

`setup.sh` lets uv pick the torch build that matches your driver. RTX 50-series cards
(sm_120) need CUDA 12.8: `TORCH_BACKEND=cu128 ./setup.sh`. Set `STEMSEP_MODELS=/some/dir`
to keep the weights (~4 GB for the full set) somewhere else.

## Pick the right family — this is the one mistake that matters

**Music models and cinema models are trained on different tasks and are not interchangeable.**
Running `music` on a film mix smears dialogue and effects into "other", because MUSDB-trained
models have no concept of an SFX stem. Running `cinema` on a song gives you three stems that
mean nothing musically.

| Source | Command | Why |
|---|---|---|
| Song, stereo, 44.1k | `music` | BS-RoFormer / SCNet, MUSDB+ lineage |
| Film, TV, 5.1 or 48k | `cinema` | Bandit v2, DnR v3 lineage, native 48 kHz |
| A stem you already have | `repair` | fix artifacts; do **not** re-separate |

## What the tool does that a plain MSST call does not

- **One resample, never two.** Models run at their native rate; conversion is a single soxr
  pass in (`precision=28`) and a single one back on write. Chaining 48→44.1→48 around a model
  is measurable damage — on MVSep's own DnR v3 board the same BandIt Plus checkpoint scores
  10.04 speech SDR resampled vs 9.66 not, and the music stem moves the other way.
- **Channel math before neural nets.** On a 5.1 source the centre channel is *already*
  dialogue. `cinema` writes `dialogue_centre.wav` straight off C with zero processing, takes
  LFE the same way, and runs the model only on the `FL/FR/BL/BR` fold. Throwing a 5.1 downmix
  at a stereo model discards information you were handed for free. `--no-channel-math` opts out.
- **Residual subtraction.** One stem is rebuilt as *source minus the rest*, so the set sums
  back to the original exactly. Applied automatically only when the stems don't already sum
  back (a target-instrument model's complement is a residual already, so it is left alone).
  `--residual sfx` forces a chosen stem into that role; `--no-residual` disables it.
- **A null test on every run.** Prints how far the stems miss reconstructing the source.
  `-147 dB` means an exact decomposition; `-20 dB` means the model is throwing signal away.
  This is the number to watch, not SDR.
- **Waveform-domain ensembling.** `-q best` runs two models and combines with `avg_wave` —
  the only method that was better-or-equal across the board in ZFTurbo's own sweep. It works
  because these models preserve phase; averaging magnitude spectrograms would smear it.

## Models

`stemsep models` prints the live list with a `*` against what is already downloaded.

| Name | Stems | Rate | Notes |
|---|---|---|---|
| `leap` | vocals / instrumental | 44.1k | **BS-RoFormer Leap Xe (unwa), SDR 11.76 — best open 2-stem. Default for `-s 2`** |
| `sw` | bass/drums/other/vocals/**guitar/piano** | 44.1k | **BS-RoFormer SW, 6 stems. Beats `scnet-xl` on every shared stem (bass 14.62 vs 11.94, drums 14.11 vs 11.58, other 8.72 vs 6.48). Default for `-s 6`** |
| `bsroformer` | vocals / instrumental | 44.1k | BS-RoFormer viperx, SDR 10.87 |
| `melband` | vocals / instrumental | 44.1k | Mel-Band RoFormer KJ, SDR 10.98. Different error signature — ensembles well with the above |
| `polarformer` | vocals / instrumental | 44.1k | SDR 11.00, highest open 2-stem |
| `scnet-xl` | bass/drums/vocals/other | 44.1k | Best open 4-stem (MUSDB avg 10.08). 1.1 GB |
| `htdemucs` | bass/drums/vocals/other | 44.1k | Fast fallback; two generations behind on vocals (8.33 vs 12.3) |
| `bandit-v2` | speech / music / sfx | **48k** | Best *open* cinema model. Multilingual + per-language weights |
| `bandit-plus` | speech / music / effects | 44.1k | DnR v2 predecessor, kept for A/B |
| `denoise` `denoise-aggr` | dry / removed | 44.1k | Broadband hiss off a finished stem |
| `dereverb` | noreverb / reverb | 44.1k | Pulls room off dialogue |
| `crowd` | crowd / other | 44.1k | Applause, audience |
| `apollo` | restored | 44.1k | Repairs codec + separation artifacts. Run **last** |

`--lang eng|deu|fra|spa|cmn|fao` selects Bandit v2's language-specific weights (each is a
separate 426 MB download, cached as `bandit-v2-<lang>`).

## The full cinema pipeline

Each pass does a *different* job, rather than re-doing the same one:

```bash
stemsep cinema movie.wav --submix -o stems/    # channel math -> D/M/E -> split the music stem
stemsep repair stems/speech_bed.wav -m dereverb
stemsep repair stems/music_bass.wav -m apollo
```

1. C and LFE off by channel math (lossless), model on the L/R/Ls/Rs fold
2. Bandit v2 at native 48k → speech / music / sfx
3. `--submix` runs SCNet XL on the music stem only → bass/drums/vocals/other
4. `repair` for dereverb/denoise, Apollo last

Final dialogue is `dialogue_centre` plus `speech_bed` if dialogue leaked into the surrounds —
check the bed before you assume it didn't.

## Flags worth knowing

| Flag | Effect |
|---|---|
| `-q fast\|good\|best` | `best` ensembles two models — 2–3× slower, ~+0.3–0.5 dB |
| `-s 2\|4\|6` | 2-stem, 4-stem, or 6-stem (`sw`, adds guitar + piano). `music` only |
| `--tta` | Test-time augmentation. ~3× slower, small gain |
| `--residual NAME` | Force NAME to be the mix-minus-the-rest stem |
| `--model-rate` | Write at the model's rate; skips the resample back |
| `--submix` | `cinema` only: also split the music stem |
| `--no-channel-math` | `cinema` only: fold 5.1 to stereo instead of taking C directly |

### Where the community checkpoints come from

`leap` and `sw` are community releases mirrored in
[`noblebarkrr/mvsepless_resources`](https://huggingface.co/noblebarkrr/mvsepless_resources) —
426 MSST-ready models with paired configs, indexed in its `models.json`. That catalog is the
place to look for anything not in this registry.

Two cautions. These mirrors move: the account hosting the original SW checkpoint was deleted
in mid-2026 and the models were re-hosted, so the local cache in `models/` is the copy you
should rely on. And some catalog entries are extracted from commercial products
(`bs_logic_6stem` is Logic Pro's Stem Splitter) — deliberately not in this registry.

## Gotchas

- **A 16 GB card runs all of this** — inference is light next to video work, and flash
  attention engages (sm_120 ≥ 8.0). A 15 s clip is a few seconds per model.
- **Models download on first use**, 300 MB – 1.1 GB each, resumable (`curl -C -`). Pre-fetch
  with `stemsep models --pull scnet-xl` if you don't want the wait mid-job.
- **Bandit v2's Zenodo weights are a raw Lightning checkpoint** — tensors under `state_dict`
  behind a `model.` prefix. `stemsep` unwraps them once in place on download and drops a
  `.normalized` marker. Loading the untouched file into MSST fails with a wall of missing keys.
- **Every model here takes exactly 2 channels.** Mono and 5.1 are folded explicitly before
  inference, never left to a decoder default.
- **`sw`'s six stems do not conserve energy** — they sum quieter than the source, so
  `stemsep` rebuilds `other` by subtraction to close the gap. That is automatic; just don't
  assume the six raw model outputs add back up.
- **Never ensemble models whose stems mean different things.** `sw` carves guitar and piano
  out of `other`; a 4-stem model leaves them in it. Averaging the two `other` stems would
  half-attenuate every guitar. The `MUSIC` recipe table keeps each row stem-compatible.
- **A bad null test is information, not a bug.** `-20 dB` on a denoise pass means the model
  removed a lot; audition `removed.wav` before trusting it.
- **SDR is not the target.** MVSep tracks *bleedless* and *fullness* separately and they trade
  off. For re-balancing a mix rather than isolating a hook, fullness is what you want — bleed
  you can duck, missing harmonics you cannot get back.

## Where the numbers come from

- [MVSEP Multisong leaderboard](https://mvsep.com/quality_checker/multisong_leaderboard?sort=vocals) — music, recomputed every 3 h
- [MVSEP DnR v3 leaderboard](https://mvsep.com/quality_checker/leaderboard/dnr_v3/?sort=speech) — cinema
- [pretrained_models.md](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md) — the config/checkpoint pairs in `models.json`. **Never mix a config with a checkpoint from a different row.**
- [Bandit v2 weights (Zenodo)](https://zenodo.org/records/12701995) · [bandit-v2 repo](https://github.com/kwatcharasupat/bandit-v2), Apache-2.0

[ZFTurbo/Music-Source-Separation-Training]: https://github.com/ZFTurbo/Music-Source-Separation-Training

## Licence

stemsep's own code is MIT (see `LICENSE`). It downloads third-party model weights at run
time; each checkpoint keeps the licence its authors published it under — check the source
linked in `models.json` before using a model commercially.
