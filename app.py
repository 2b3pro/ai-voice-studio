"""AI Voice Studio - web UI. Drop a voice note, get the studio version back.

Usage:
  python app.py            # opens http://127.0.0.1:7860
  python app.py --share    # also gives a public gradio.live link (handy on Colab / Kaggle)
"""
import os,sys,subprocess,tempfile,pathlib,json
import gradio as gr

HERE=pathlib.Path(__file__).resolve().parent

def enhance(path,strength):
    if not path:raise gr.Error('Upload or record a voice note first.')
    out=pathlib.Path(tempfile.mkdtemp(prefix='voice_pro_'))
    env=dict(os.environ,VOICE_LAMBD='0.9' if strength=='Strong (default)' else '0.5')
    p=subprocess.run([sys.executable,str(HERE/'voice_pro.py'),path,str(out)],capture_output=True,text=True,env=env)
    if p.returncode:raise gr.Error('voice_pro.py failed:\n'+p.stderr[-1500:])
    info=json.loads(p.stdout.strip().splitlines()[-1])
    return str(out/'Voice-Pro-Final.wav'),f"Done: {info['old_s']} s → {info['new_s']} s, {info['pauses_tightened']} long pauses tightened."

with gr.Blocks(title='AI Voice Studio') as demo:
    gr.Markdown('# AI Voice Studio 🎧\nTurn a WhatsApp / phone voice note into a studio-mic voice. '
                'Runs locally: noise and echo removal, pause tightening, broadcast EQ, −14 LUFS. '
                '[GitHub](https://github.com/OutOfAmine/ai-voice-studio) · by [@justbyamine](https://www.instagram.com/justbyamine/)')
    with gr.Row():
        with gr.Column():
            src=gr.Audio(label='Your voice note (upload or record)',type='filepath',sources=['upload','microphone'])
            strength=gr.Radio(['Strong (default)','Soft'],value='Strong (default)',label='Restoration',
                              info='Use Soft if the result sounds over-processed or a quiet word disappears.')
            go=gr.Button('Enhance',variant='primary')
        with gr.Column():
            res=gr.Audio(label='Studio version',type='filepath')
            msg=gr.Markdown()
    go.click(enhance,[src,strength],[res,msg])

if __name__=='__main__':
    demo.queue().launch(share='--share' in sys.argv)
