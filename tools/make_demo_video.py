"""Build demo/before-after.mp4: raw phone take, a short pause, then the same take after voice_pro.py,
with a live waveform and a playhead so the README can play the comparison inline.
Usage: python tools/make_demo_video.py <raw.wav> <enhanced.wav> <out.mp4> [seconds_each]"""
import sys,subprocess,numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
F=imageio_ffmpeg.get_ffmpeg_exe();SR=48000;W,H,FPS=1280,720,30
raw,enh,out=sys.argv[1:4];L=float(sys.argv[4]) if len(sys.argv)>4 else 7.0;GAP=.8
def dec(p):return np.frombuffer(subprocess.run([F,'-v','error','-i',p,'-ac','1','-ar',str(SR),'-f','f32le','-'],capture_output=True,check=True).stdout,np.float32)
a=dec(raw)[:int(L*SR)];b=dec(enh)[:int(L*SR)]
a=a*(10**(-18/20)/(np.sqrt(np.mean(a**2))+1e-9))   # raw take at a fair listening level
audio=np.concatenate([a,np.zeros(int(GAP*SR),np.float32),b]);audio=np.clip(audio,-1,1);dur=len(audio)/SR
def font(n,bold=True):
 try:return ImageFont.truetype('C:/Windows/Fonts/'+('segoeuib.ttf' if bold else 'segoeui.ttf'),n)
 except OSError:return ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf',n)
def bars(x,n=180):
 k=len(x)//n;return np.array([np.sqrt(np.mean(x[i*k:(i+1)*k]**2)) for i in range(n)])
BA,BB=bars(a),bars(b);m=max(BA.max(),BB.max())
def panel(d,y,vals,label,col,prog,active):
 d.rounded_rectangle((60,y,W-60,y+230),28,fill=(24,24,30),outline=col if active else (50,50,60),width=3)
 d.text((95,y+28),label,font=font(34),fill=col if active else (150,150,160))
 n=len(vals);bw=(W-200)/n
 for i,v in enumerate(vals):
  h=max(3,150*v/m);x=100+i*bw;c=col if i/n<prog else (70,70,82)
  d.rounded_rectangle((x,y+150-h/2,x+bw*.6,y+150+h/2),2,fill=c)
import tempfile,os
frames=int(dur*FPS)+1
tmp=tempfile.NamedTemporaryFile(suffix='.f32',delete=False);tmp.write(audio.astype('<f4').tobytes());tmp.close()
p=subprocess.Popen([F,'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-f','f32le','-ar',str(SR),'-ac','1','-i',tmp.name,'-c:v','libx264','-pix_fmt','yuv420p','-crf','20','-c:a','aac','-b:a','160k','-shortest','-movflags','+faststart',out],stdin=subprocess.PIPE)
for f in range(frames):
 t=f/FPS;im=Image.new('RGB',(W,H),(12,12,16));d=ImageDraw.Draw(im)
 d.text((W/2,60),'AI Voice Studio',font=font(46),fill=(255,255,255),anchor='mm')
 d.text((W/2,108),'same take, same phone: before and after one command',font=font(26,False),fill=(160,160,170),anchor='mm')
 pa=min(1,t/L);pb=min(1,max(0,(t-L-GAP)/L))
 panel(d,160,BA,'BEFORE  ·  phone voice note',(255,90,90),pa,t<L)
 panel(d,420,BB,'AFTER  ·  voice_pro.py',(120,230,170),pb,t>=L+GAP)
 d.text((W/2,H-40),'github.com/OutOfAmine/ai-voice-studio  ·  @justbyamine',font=font(22,False),fill=(120,120,130),anchor='mm')
 p.stdin.write(im.tobytes())
p.stdin.close();p.wait();os.unlink(tmp.name);print('ok',out,round(dur,2))
