"""AI Voice Studio - turn a phone voice note into a "studio mic" voice. By @justbyamine.

Usage:
  python voice_pro.py <your_voice.(mp4|ogg|opus|m4a|wav|mp3)> <output_folder>

Steps (all local, free, no subscription):
  1-restored.wav       AI restoration (resemble-enhance): removes noise + echo, rebuilds the treble
  2-tight.wav          long pauses (> 180 ms) shortened to 140 ms
  3-chain.wav          broadcast EQ (warmth + clarity), de-esser, compression
  Voice-Pro-Final.wav  final voice, -14 LUFS (Instagram / TikTok loudness)
No pitch or speed change. A GPU (NVIDIA) is much faster; CPU works but is slow.
"""
import sys,types,pathlib,os,json,re,subprocess
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import numpy as np,imageio_ffmpeg

# deepspeed is only used for training in resemble-enhance; stub it so inference imports on Windows.
class _Any:
 def __init__(self,*a,**k):pass
 def __call__(self,*a,**k):return _Any()
 def __getattr__(self,n):
  if n.startswith('__'):raise AttributeError(n)
  return _Any()
def _mod_getattr(n):
 if n.startswith('__'):raise AttributeError(n)
 return _Any
for name in ['deepspeed','deepspeed.accelerator','deepspeed.runtime','deepspeed.runtime.engine','deepspeed.runtime.utils','deepspeed.utils']:
 m=types.ModuleType(name);m.__getattr__=_mod_getattr;sys.modules[name]=m
import torch,torchaudio
if os.name=='nt':pathlib.PosixPath=pathlib.WindowsPath  # model configs were saved on Linux

if len(sys.argv)<3:sys.exit('usage: python voice_pro.py <voice file> <output folder>')
SRC=pathlib.Path(sys.argv[1]).resolve();O=pathlib.Path(sys.argv[2]).resolve();O.mkdir(parents=True,exist_ok=True)
F=imageio_ffmpeg.get_ffmpeg_exe();SR=48000
# Any audio/video input (.mp4/.opus/.ogg, .m4a, .mp3...) -> mono 48 kHz WAV
IN=O/'0-input.wav';subprocess.run([F,'-v','error','-y','-i',str(SRC),'-vn','-ac','1','-ar',str(SR),str(IN)],check=True)
def run(a):
 p=subprocess.run([F,'-hide_banner','-y',*a],capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr)
 return p.stderr
def dec(p):return np.frombuffer(subprocess.run([F,'-v','error','-i',str(p),'-ac','1','-ar',str(SR),'-f','f32le','-'],capture_output=True,check=True).stdout,np.float32).copy()
def write(p,x):subprocess.run([F,'-v','error','-y','-f','f32le','-ar',str(SR),'-ac','1','-i','-','-c:a','pcm_s24le',str(p)],input=x.astype('<f4').tobytes(),check=True)
def loudnorm(src,dst,I=-14,TP=-1.5,LRA=7):
 v=json.loads(re.findall(r'\{[^{}]+\}',run(['-i',str(src),'-af',f'loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json','-f','null','-']))[-1])
 run(['-i',str(src),'-af',f"loudnorm=I={I}:TP={TP}:LRA={LRA}:measured_I={v['input_i']}:measured_TP={v['input_tp']}:measured_LRA={v['input_lra']}:measured_thresh={v['input_thresh']}:offset={v['target_offset']}:linear=true",'-ar',str(SR),'-c:a','pcm_s24le',str(dst)])
 return float(v['input_i'])

# 1) Restoration. lambd=0.9 (strong denoise) was approved; 0.5 is the softer fallback.
from resemble_enhance.enhancer.inference import enhance
wav,sr=torchaudio.load(str(IN));wav=wav.mean(0)
e,esr=enhance(wav,sr,'cuda' if torch.cuda.is_available() else 'cpu',nfe=64,solver='midpoint',lambd=float(os.environ.get('VOICE_LAMBD','0.9')),tau=0.5)
torchaudio.save(str(O/'1-restored.wav'),e[None].cpu(),esr)
x=dec(O/'1-restored.wav')

# 2) Pause tightening: cut inside the quiet part, keep 70 ms after the word and 70 ms before the next.
hop=480;env=20*np.log10(np.array([np.sqrt(np.mean(x[i:i+hop]**2)) for i in range(0,len(x)-hop,hop)])+1e-9);env-=np.percentile(env,95)
runs=[];s=None
for i,v in enumerate(list(env<-42)+[False]):
 if v and s is None:s=i
 if not v and s is not None:
  if (i-s)*hop/SR>=.18 and s>0 and i<len(env):runs.append((s*hop,i*hop))
  s=None
KEEP=round(.14*SR);XF=round(.008*SR);parts=[];tmap=[];pos=0;out_len=0;removed=[]
for a,b in runs:
 cut=(b-a)-KEEP
 if cut<=0:continue
 c0=a+KEEP//2;c1=c0+cut
 seg=x[pos:c0+XF].copy();parts.append(seg);tmap.append([pos/SR,c0/SR,out_len/SR]);out_len+=len(seg)-XF;pos=c1;removed.append([round(c0/SR,3),round(c1/SR,3)])
parts.append(x[pos:].copy());tmap.append([pos/SR,len(x)/SR,out_len/SR])
y=parts[0]
for p in parts[1:]:
 r=np.linspace(0,1,XF,dtype=np.float32);y=np.concatenate([y[:-XF],y[-XF:]*(1-r)+p[:XF]*r,p[XF:]])
write(O/'2-tight.wav',y)

# 3) Broadcast chain.
chain=','.join(['highpass=f=60:p=2',
 'equalizer=f=120:t=q:w=0.9:g=3',      # proximity warmth
 'equalizer=f=380:t=q:w=1.4:g=-3',     # remove boxy/phone tone
 'equalizer=f=4800:t=q:w=1.2:g=2.5',   # presence / intelligibility
 'treble=g=3:f=11000:t=s:w=0.7',       # air shelf
 'deesser=i=0.35:m=0.5:f=0.5:s=o',
 'acompressor=threshold=0.1:ratio=3:attack=5:release=80:knee=4:makeup=1.5',
 'acompressor=threshold=0.35:ratio=6:attack=1:release=40:knee=2'])
run(['-i',str(O/'2-tight.wav'),'-af',chain,'-ar',str(SR),'-c:a','pcm_s24le',str(O/'3-chain.wav')])

# 4) Room tone at -66 dBFS (after gain) so pauses never drop to dead digital silence, then loudnorm.
z=dec(O/'3-chain.wav')
v=json.loads(re.findall(r'\{[^{}]+\}',run(['-i',str(O/'3-chain.wav'),'-af','loudnorm=I=-14:TP=-1.5:LRA=7:print_format=json','-f','null','-']))[-1])
g=10**((-14-float(v['input_i']))/20)
w=np.random.default_rng(3).standard_normal(len(z)).astype(np.float32);b=np.zeros_like(w);acc=0
for i in range(len(w)):acc=.985*acc+w[i];b[i]=acc
b=b-np.convolve(b,np.ones(4800)/4800,'same');b/=np.sqrt(np.mean(b**2))
write(O/'4-gain.wav',z*g+b*10**(-66/20))
loudnorm(O/'4-gain.wav',O/'Voice-Pro-Final.wav')
(O/'4-gain.wav').unlink()
(O/'time-map.json').write_text(json.dumps({'map':tmap,'removed':removed,'old_duration':len(x)/SR,'new_duration':len(y)/SR},indent=1))
print(json.dumps({'pauses_tightened':len(removed),'old_s':round(len(x)/SR,2),'new_s':round(len(y)/SR,2),'out':str(O/'Voice-Pro-Final.wav')}))
