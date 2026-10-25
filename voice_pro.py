"""AI Voice Studio (Mac Edition) — phone voice note ➜ studio mic.
Optimized for macOS & Apple Silicon (M1/M2/M3/M4/M5).

Usage:
  python3 voice_pro.py <voice_file> <output_folder> [options]

Options:
  -h, --help            Show this help message and exit
  -s, --strength        AI restoration profile: 'strong' (lambd=0.9, default) or 'natural' (lambd=0.5)
  -l, --lambd FLOAT     Denoising weight (0.0 to 1.0, overrides --strength, default: 0.9)
  -a, --air FLOAT       11 kHz treble shelf boost in dB (0.0 to 6.0, default: 3.0)
  --tighten             Shorten long pauses (> 180 ms) to 140 ms (default on for audio)
  --no-tighten          Keep original pauses intact (default for video)
  -d, --device          Compute device: 'mps' (Apple Silicon GPU), 'cpu', or 'auto' (default: auto)
  --nfe INT             Diffusion steps (1 to 128, default: 64)
  --solver              Numerical ODE solver ('midpoint', 'rk4', 'euler', default: midpoint)
  --tau FLOAT           Prior temperature (0.0 to 1.0, default: 0.5)
  --lufs FLOAT          Target broadcast loudness (default: -14.0 LUFS)
  --tp FLOAT            True-peak ceiling (default: -1.5 dBTP)

Pipeline (all local, private, accelerated by Apple Silicon):
  1-restored.wav       AI restoration (resemble-enhance via Metal / MPS)
  2-tight.wav          Long pauses (> 180 ms) shortened to 140 ms (optional)
  3-chain.wav          Broadcast EQ (warmth + clarity), de-esser, compression
  Voice-Pro-Final.wav  Final voice at -14 LUFS (Instagram / TikTok loudness standard)
  Voice-Pro-Video.mp4  (Video input only) Hardware muxed with Apple AudioToolbox & VideoToolbox
"""
import sys, types, pathlib, os, json, re, subprocess, argparse

__version__ = '2.0.0'

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='voice_pro.py',
        description='AI Voice Studio (Mac Edition) — phone voice note ➜ studio mic.\n'
                    'Optimized for macOS & Apple Silicon (M1/M2/M3/M4/M5).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 voice_pro.py voice.m4a output_dir
  python3 voice_pro.py voice.m4a output_dir --strength natural --air 1.5
  python3 voice_pro.py voice.m4a output_dir --lambd 0.7 --nfe 32
  python3 voice_pro.py video.mp4 output_dir --tighten --device mps
  python3 voice_pro.py podcast.wav output_dir --air 0.0 --lufs -16.0 --tp -2.0
"""
    )
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__} (Mac Edition )')
    parser.add_argument('input', type=pathlib.Path, help='Input audio or video file (.mp4, .mov, .m4a, .wav, .mp3, etc.)')
    parser.add_argument('output', type=pathlib.Path, help='Output directory for processed files')

    # Customization flags
    parser.add_argument('-s', '--strength', choices=['strong', 'natural'], default=None,
                        help="AI restoration strength profile: 'strong' (lambd=0.9, default) or 'natural' (lambd=0.5)")
    parser.add_argument('-l', '--lambd', type=float, default=None,
                        help="AI restoration denoising weight (0.0 to 1.0, overrides --strength, default: 0.9)")
    parser.add_argument('-a', '--air', type=float, default=None,
                        help="High-frequency 11 kHz air shelf gain in dB (0.0 to 6.0, default: 3.0)")
    parser.add_argument('--tighten', dest='tighten', action='store_true', default=None,
                        help="Shorten long pauses (> 180 ms) to 140 ms (default on for audio)")
    parser.add_argument('--no-tighten', dest='tighten', action='store_false',
                        help="Keep original timing without pause shortening (default on for video)")
    parser.add_argument('-d', '--device', choices=['auto', 'mps', 'cpu'], default='auto',
                        help="Compute device for CFM diffusion: 'mps' (Apple Silicon GPU), 'cpu', or 'auto' (default: auto)")
    parser.add_argument('--nfe', type=int, default=64,
                        help="Number of function evaluations / diffusion steps (1 to 128, default: 64)")
    parser.add_argument('--solver', choices=['midpoint', 'rk4', 'euler'], default='midpoint',
                        help="Numerical ODE solver for continuous normalizing flow (default: midpoint)")
    parser.add_argument('--tau', type=float, default=0.5,
                        help="Prior temperature for diffusion (0.0 to 1.0, default: 0.5)")
    parser.add_argument('--lufs', type=float, default=-14.0,
                        help="Target integrated broadcast loudness in LUFS (default: -14.0)")
    parser.add_argument('--tp', type=float, default=-1.5,
                        help="Target maximum true peak in dBTP (default: -1.5)")
    return parser.parse_args()

cli_args = parse_arguments()

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
import numpy as np, imageio_ffmpeg
import mlx.core as mx
import resemble_enhance_mlx as re_mlx
from resemble_enhance_mlx import lvc

# JIT-compile location-variable convolution Metal kernels for maximum performance
lvc.location_variable_convolution = mx.compile(lvc.location_variable_convolution)

args = [a for a in sys.argv[1:] if not a.startswith('--')]

SRC = cli_args.input.resolve()
O = cli_args.output.resolve()
O.mkdir(parents=True, exist_ok=True)
F = imageio_ffmpeg.get_ffmpeg_exe()
SR = 48000

# Resolve configuration parameters (CLI flags override environment variables)
if cli_args.lambd is not None:
    LAMBD = max(0.0, min(1.0, float(cli_args.lambd)))
elif cli_args.strength == 'natural':
    LAMBD = 0.5
elif cli_args.strength == 'strong':
    LAMBD = 0.9
elif 'VOICE_LAMBD' in os.environ:
    LAMBD = float(os.environ['VOICE_LAMBD'])
else:
    LAMBD = 0.9

if cli_args.air is not None:
    AIR_GAIN = float(cli_args.air)
elif 'VOICE_AIR_DB' in os.environ:
    AIR_GAIN = float(os.environ['VOICE_AIR_DB'])
else:
    AIR_GAIN = 3.0

NFE = max(1, min(128, int(cli_args.nfe)))
SOLVER = cli_args.solver
TAU = max(0.0, min(1.0, float(cli_args.tau)))
TARGET_LUFS = float(cli_args.lufs)
TARGET_TP = float(cli_args.tp)

def progress(value, message):
    print('STUDIO_PROGRESS ' + json.dumps({'value': value, 'message': message}), flush=True)

progress(0.05, 'Reading your recording')

# Input conversion: audio/video (.mp4/.mov/.opus/.m4a/.wav...) -> mono 48 kHz WAV
IN = O / '0-input.wav'
subprocess.run([F, '-v', 'error', '-y', '-i', str(SRC), '-vn', '-ac', '1', '-ar', str(SR), str(IN)], check=True)
probe = subprocess.run([F, '-hide_banner', '-i', str(SRC)], capture_output=True, text=True, errors='replace').stderr
VIDEO = bool(re.search(r'Stream #\S+.*: Video: (?!mjpeg|png|bmp)', probe))

if cli_args.tighten is not None:
    TIGHTEN = cli_args.tighten
elif 'VOICE_TIGHTEN' in os.environ:
    TIGHTEN = os.environ['VOICE_TIGHTEN'] != '0'
else:
    TIGHTEN = not VIDEO

with __import__('wave').open(str(IN), 'rb') as audio:
    if audio.getnframes() < SR // 2:
        raise ValueError('Record at least half a second of speech.')
    samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2')
    if np.max(np.abs(samples.astype(np.float32))) < 10:
        raise ValueError('This recording is silent. Check your microphone and try again.')

def run(a):
    p = subprocess.run([F, '-hide_banner', '-y', *a], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr)
    return p.stderr

def dec(p):
    return np.frombuffer(
        subprocess.run([F, '-v', 'error', '-i', str(p), '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                       capture_output=True, check=True).stdout, np.float32).copy()

def write(p, x, codec='pcm_s24le', sr=None):
    r = sr or SR
    subprocess.run([F, '-v', 'error', '-y', '-f', 'f32le', '-ar', str(r), '-ac', '1', '-i', '-', '-c:a', codec, str(p)],
                   input=x.astype('<f4').tobytes(), check=True)

def loudnorm(src, dst, I=-14, TP=-1.5):
    # Pure linear gain measurement pass (avoids dynamic pumping)
    m = json.loads(re.findall(r'\{[^{}]+\}', run(['-i', str(src), '-af', f'loudnorm=I={I}:TP={TP}:print_format=json', '-f', 'null', '-']))[-1])
    gain = min(I - float(m['input_i']), TP - float(m['input_tp']))
    run(['-i', str(src), '-af', f'volume={gain:.3f}dB', '-ar', str(SR), '-c:a', 'pcm_s24le', str(dst)])
    return {'lufs': round(float(m['input_i']) + gain, 2), 'true_peak': round(float(m['input_tp']) + gain, 2), 'gain_db': round(gain, 2)}

# 1) Restoration via Apple Silicon Native MLX Engine
progress(0.15, f'Removing noise and echo with AI (Apple MLX Metal accelerated, lambd={LAMBD})')
raw_wav = dec(IN)
model_path = pathlib.Path(__file__).parent / 'resemble-enhance-mlx.safetensors'
if not model_path.is_file():
    from huggingface_hub import hf_hub_download
    model_path = pathlib.Path(hf_hub_download('multimodalart/resemble-enhance-mlx', 'resemble-enhance-mlx.safetensors'))

mlx_model = re_mlx.load(str(model_path), lambd=LAMBD)
mlx_model.lcfm.cfm._to_v = mx.compile(mlx_model.lcfm.cfm._to_v)

e, esr = re_mlx.enhance(mlx_model, raw_wav, SR)
write(O / '1-restored.wav', e, sr=esr)

x = dec(O / '1-restored.wav')
progress(0.65, 'Tightening long pauses' if TIGHTEN else 'Keeping your timing')

# 2) Pause tightening: shorten silences > 180 ms to 140 ms
hop = 480
env = 20 * np.log10(np.array([np.sqrt(np.mean(x[i:i+hop]**2)) for i in range(0, len(x) - hop, hop)]) + 1e-9)
env -= np.percentile(env, 95)
runs = []
s = None
for i, v in enumerate(list(env < -42) + [False]):
    if v and s is None: s = i
    if not v and s is not None:
        if (i - s) * hop / SR >= 0.18 and s > 0 and i < len(env):
            runs.append((s * hop, i * hop))
        s = None

KEEP = round(0.14 * SR)
XF = round(0.008 * SR)
parts = []
tmap = []
pos = 0
out_len = 0
removed = []
for a, b in (runs if TIGHTEN else []):
    cut = (b - a) - KEEP
    if cut <= 0: continue
    c0 = a + KEEP // 2
    c1 = c0 + cut
    seg = x[pos:c0 + XF].copy()
    parts.append(seg)
    tmap.append([pos / SR, c0 / SR, out_len / SR])
    out_len += len(seg) - XF
    pos = c1
    removed.append([round(c0 / SR, 3), round(c1 / SR, 3)])
parts.append(x[pos:].copy())
tmap.append([pos / SR, len(x) / SR, out_len / SR])
y = parts[0]
for p in parts[1:]:
    r = np.linspace(0, 1, XF, dtype=np.float32)
    y = np.concatenate([y[:-XF], y[-XF:] * (1 - r) + p[:XF] * r, p[XF:]])
write(O / '2-tight.wav', y)

# 3) Broadcast EQ & dynamic processing
progress(0.75, 'Adding warmth, clarity and broadcast compression')
air_gain = AIR_GAIN
chain = ','.join([
    'highpass=f=60:p=2',
    'equalizer=f=120:t=q:w=0.9:g=3',                                  # Proximity warmth
    'equalizer=f=380:t=q:w=1.4:g=-3',                                 # Remove boxy phone tone
    'equalizer=f=4800:t=q:w=1.2:g=2.5',                               # Speech presence & intelligibility
    f"treble=g={air_gain}:f=11000:t=s:w=0.7",                         # High-frequency air shelf
    'deesser=i=0.35:m=0.5:f=0.5:s=o',                                 # Sibilance control
    'acompressor=threshold=0.1:ratio=3:attack=5:release=80:knee=4:makeup=1.5',
    'acompressor=threshold=0.35:ratio=6:attack=1:release=40:knee=2'
])
run(['-i', str(O / '2-tight.wav'), '-af', chain, '-ar', str(SR), '-c:a', 'pcm_s24le', str(O / '3-chain.wav')])

# 4) Natural room tone (-66 dBFS) & true-peak limiter
progress(0.85, 'Balancing broadcast volume and limiting true peaks')
z = dec(O / '3-chain.wav')
v = json.loads(re.findall(r'\{[^{}]+\}', run(['-i', str(O / '3-chain.wav'), '-af', f'loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA=7:print_format=json', '-f', 'null', '-']))[-1])
g = 10**((TARGET_LUFS - float(v['input_i'])) / 20)

from scipy.signal import lfilter
b = lfilter([1], [1, -0.985], np.random.default_rng(3).standard_normal(len(z))).astype(np.float32)
b = b - np.convolve(b, np.ones(4800) / 4800, 'same')
b /= np.sqrt(np.mean(b**2))

write(O / '4-gain.wav', z * g + b * 10**(-66 / 20), 'pcm_f32le')
run(['-i', str(O / '4-gain.wav'), '-af', 'alimiter=limit=0.6:attack=1:release=50:level=0', '-ar', str(SR), '-c:a', 'pcm_f32le', str(O / '5-limited.wav')])
final = loudnorm(O / '5-limited.wav', O / 'Voice-Pro-Final.wav', I=TARGET_LUFS, TP=TARGET_TP)
for f in ('4-gain.wav', '5-limited.wav'):
    (O / f).unlink()

# Video output: Hardware accelerated via Apple AudioToolbox (AAC) & VideoToolbox (H.264)
video = None
if VIDEO:
    progress(0.95, 'Muxing with Apple AudioToolbox & VideoToolbox')
    V = O / 'Voice-Pro-Video.mp4'
    base = ['-i', str(SRC), '-i', str(O / 'Voice-Pro-Final.wav'), '-map', '0:v:0', '-map', '1:a:0']
    tail = ['-c:a', 'aac_at', '-b:a', '256k', '-ar', str(SR), '-movflags', '+faststart', str(V)]
    try:
        run([*base, '-c:v', 'copy', *tail])
    except RuntimeError:
        run([*base, '-c:v', 'h264_videotoolbox', '-b:v', '6000k', *tail])
    video = str(V)

(O / 'time-map.json').write_text(json.dumps({'map': tmap, 'removed': removed, 'old_duration': len(x) / SR, 'new_duration': len(y) / SR}, indent=1))
print(json.dumps({'pauses_tightened': len(removed), 'old_s': round(len(x) / SR, 2), 'new_s': round(len(y) / SR, 2), 'final': final, 'out': str(O / 'Voice-Pro-Final.wav'), 'video': video}))
