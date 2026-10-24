"""Local recorder, enhancement and persistent voice library. Run: python app.py."""
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
import gradio as gr

HERE = pathlib.Path(__file__).resolve().parent
LIBRARY = pathlib.Path(os.environ.get('VOICE_STUDIO_LIBRARY', HERE / 'local-library')).resolve()
LIBRARY.mkdir(parents=True, exist_ok=True)


def history_choices():
    records = []
    for path in LIBRARY.glob('*/record.json'):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            if record['id'] == path.parent.name:
                records.append(record)
        except (OSError, ValueError, KeyError):
            continue
    return [(f"{r['name']} · {r['date']} · {r['mode']}", r['id'])
            for r in sorted(records, key=lambda r: r['created'], reverse=True)]


def open_record(key):
    if not key:
        return None, None, gr.update(value=None, interactive=False), '', gr.update(value=None, visible=False)
    if not isinstance(key, str) or not re.fullmatch(r'[a-f0-9]{32}', key):
        raise gr.Error('Choose a recording from your library.')
    folder = LIBRARY / key
    if not (folder / 'record.json').is_file():
        raise gr.Error('This recording is no longer available.')
    record = json.loads((folder / 'record.json').read_text(encoding='utf-8'))
    final = str(folder / 'Voice-Pro-Final.wav')
    return (str(folder / '0-input.wav'), final, gr.update(value=final, interactive=True), record['summary'],
            video_update(folder))


def video_update(folder):
    video = folder / 'Voice-Pro-Video.mp4'
    return gr.update(value=str(video), visible=True) if video.is_file() else gr.update(value=None, visible=False)


def enhance(path, strength, title, tighten=True, progress=gr.Progress()):
    if not path or not pathlib.Path(path).is_file():
        raise gr.Error('Record your voice or import a file first.')
    key = uuid.uuid4().hex
    folder = LIBRARY / key
    folder.mkdir()
    original = 'original' + pathlib.Path(path).suffix.lower()
    source = folder / original
    env = dict(os.environ, VOICE_LAMBD='0.9' if strength == 'Strong' else '0.5',
               PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    if not tighten:
        env['VOICE_TIGHTEN'] = '0'  # when left on, voice_pro.py still keeps the timing of a video
    progress(0, desc='Starting your local studio…')
    log_path = folder / 'processing.log'
    try:
        shutil.copy2(path, source)
        with log_path.open('w', encoding='utf-8') as log:
            process = subprocess.Popen(
                [sys.executable, str(HERE / 'voice_pro.py'), str(source), str(folder)],
                stdout=subprocess.PIPE, stderr=log, text=True, encoding='utf-8', env=env)
            info = None
            for line in process.stdout:
                if line.startswith('STUDIO_PROGRESS '):
                    update = json.loads(line.removeprefix('STUDIO_PROGRESS '))
                    progress(update['value'], desc=update['message'])
                else:
                    try:
                        candidate = json.loads(line)
                        if isinstance(candidate, dict) and 'final' in candidate:
                            info = candidate
                    except ValueError:
                        pass
            code = process.wait()
        if code or info is None or not (folder / 'Voice-Pro-Final.wav').is_file():
            detail = log_path.read_text(encoding='utf-8', errors='replace')[-1800:]
            raise RuntimeError(detail or 'The processor did not produce a result.')
        summary = (f"{info['old_s']}s → {info['new_s']}s · "
                   f"{info['pauses_tightened']} pauses shortened · {info['final']['lufs']} LUFS")
        if info.get('video'):
            summary += ' · video ready'
        now = datetime.now(timezone.utc)
        record = dict(id=key, name=(title or '').strip()[:100] or f"Recording {now.astimezone():%d %b, %H:%M}",
                      created=now.isoformat(), date=now.astimezone().strftime('%d %b %Y, %H:%M'),
                      original=original, mode=strength, summary=summary, metrics=info)
        temp = folder / 'record.tmp'
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(folder / 'record.json')
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise gr.Error(f'Processing failed. Your input is still available; try again.\n{exc}') from exc
    progress(1, desc='Ready. Saved to your local library.')
    return (str(folder / '0-input.wav'), str(folder / 'Voice-Pro-Final.wav'),
            gr.update(value=str(folder / 'Voice-Pro-Final.wav'), interactive=True), summary,
            gr.update(choices=history_choices(), value=key), video_update(folder))


CSS = """
.gradio-container {max-width: 760px !important; margin: auto !important; padding-top: 20px !important;}
#brand {font-size: 12px; letter-spacing: 2px; color: #47876f;}
#hero h1 {font-size: 36px; letter-spacing: -1px; margin-bottom: 8px;}
#steps {padding: 14px 0; color: #678476; font-size: 15px;}
.step-card {border: 1px solid var(--border-color-primary) !important; border-radius: 20px !important; padding: 22px !important;}
.primary-action {min-height: 56px !important; font-size: 17px !important; border-radius: 12px !important;}
#privacy {color: #678476; font-size: 12px; text-align: center; padding: 12px;}
footer {display: none !important;}
"""
THEME = gr.themes.Soft(primary_hue='emerald', neutral_hue='slate', font=['Arial', 'sans-serif'])


def ready_for_step_two(path):
    if not path or not pathlib.Path(path).is_file():
        raise gr.Error('Add a file or record your voice first.')
    return gr.update(visible=False), gr.update(visible=True), path, '✓ Voice added. Next: clean it up.'


def finish_voice(path, strength, title, tighten, progress=gr.Progress()):
    before, after, download, summary, history, video = enhance(path, strength, title, tighten, progress)
    return (before, after, download, f'✓ Saved automatically on this computer.\n\n{summary}', history,
            gr.update(visible=False), gr.update(visible=True), '✓ Your voice is ready.', video)


def build_app():
    with gr.Blocks(title='AI Voice Studio', analytics_enabled=False) as demo:
        selected = gr.State(None)
        gr.Markdown('AI VOICE STUDIO · BY AMINE', elem_id='brand')
        gr.Markdown('# Make your voice sound better.\nThree easy steps. Everything stays on your computer.', elem_id='hero')
        with gr.Tabs():
            with gr.Tab('New voice'):
                breadcrumb = gr.Markdown('**1. Add voice** → 2. Clean up → 3. Download', elem_id='steps')
                with gr.Column(visible=True, elem_classes='step-card') as step_one:
                    gr.Markdown('## 1. Add your voice\nHow would you like to start?')
                    with gr.Row():
                        choose_file = gr.Button('📁 I have a file', size='lg', variant='primary')
                        choose_mic = gr.Button('🎙 Record my voice', size='lg')
                    with gr.Column(visible=True) as import_box:
                        file_input = gr.File(label='Drop your audio or video here', file_types=['audio', 'video'], type='filepath', height=130)
                    with gr.Column(visible=False) as record_box:
                        gr.Markdown('Press **Record**, speak, then press **Stop**.\nAllow microphone access when your browser asks.')
                        source = gr.Audio(label='Your recording', sources=['microphone'], type='filepath', format='wav')
                    next_button = gr.Button('Next →', variant='primary', interactive=False, elem_classes='primary-action')
                with gr.Column(visible=False, elem_classes='step-card') as step_two:
                    gr.Markdown('## 2. Clean up your voice\nOne click removes noise, reduces echo and balances the volume.')
                    preview = gr.Audio(label='Your voice', interactive=False)
                    go = gr.Button('✨ Make my voice better', variant='primary', elem_classes='primary-action')
                    gr.Markdown('This may take a few minutes. Keep this page open.')
                    with gr.Accordion('Optional settings', open=False):
                        title = gr.Textbox(label='Name (optional)', placeholder='My recording', max_lines=1)
                        mode = gr.Radio(['Strong', 'Soft'], value='Strong', label='Cleanup', info='Leave Strong selected. Try Soft for a gentler result.')
                        tighten = gr.Checkbox(True, label='Shorten long pauses', info='Videos always keep their timing so the voice stays on the lips.')
                    back = gr.Button('← Change my voice', size='sm')
                with gr.Column(visible=False, elem_classes='step-card') as step_three:
                    gr.Markdown('## 3. Your voice is ready!\nPlay the original, then the improved voice, and hear the difference.')
                    with gr.Row(equal_height=True):
                        before = gr.Audio(label='Before', interactive=False)
                        after = gr.Audio(label='After ✨', interactive=False)
                    download = gr.DownloadButton('⬇ Download my voice', interactive=False, variant='primary', elem_classes='primary-action')
                    video_out = gr.Video(label='Your video with the new voice (download it from the player)', interactive=False, visible=False)
                    status = gr.Markdown()
                    again = gr.Button('＋ Start a new voice')
            with gr.Tab('Saved voices'):
                gr.Markdown('## Your saved voices\nChoose a recording to listen or download it again.')
                history = gr.Dropdown(choices=history_choices(), value=None, label='Choose a recording', interactive=True)
                refresh = gr.Button('Refresh list', size='sm')
                with gr.Row(equal_height=True):
                    saved_before = gr.Audio(label='Before', interactive=False)
                    saved_after = gr.Audio(label='After ✨', interactive=False)
                saved_download = gr.DownloadButton('⬇ Download my voice', interactive=False, variant='primary')
                saved_video = gr.Video(label='Your video with the new voice', interactive=False, visible=False)
                saved_status = gr.Markdown()
        gr.Markdown('🔒 Only on your computer · No account needed', elem_id='privacy')

        def choose(recording):
            return (gr.update(visible=not recording), gr.update(visible=recording), None,
                    None, None, gr.update(interactive=False))
        choice_outputs = [import_box, record_box, selected, source, file_input, next_button]
        choose_file.click(lambda: choose(False), outputs=choice_outputs, queue=False)
        choose_mic.click(lambda: choose(True), outputs=choice_outputs, queue=False)
        def set_voice(path):
            return path, gr.update(interactive=bool(path))
        source.input(set_voice, source, [selected, next_button], queue=False)
        file_input.upload(set_voice, file_input, [selected, next_button], queue=False)
        file_input.clear(lambda: (None, gr.update(interactive=False)), outputs=[selected, next_button], queue=False)
        next_button.click(ready_for_step_two, selected, [step_one, step_two, preview, breadcrumb])
        back.click(lambda: (gr.update(visible=True), gr.update(visible=False), '**1. Add voice** → 2. Clean up → 3. Download'),
                   outputs=[step_one, step_two, breadcrumb], queue=False)
        go.click(finish_voice, [selected, mode, title, tighten],
                 [before, after, download, status, history, step_two, step_three, breadcrumb, video_out],
                 concurrency_limit=1, concurrency_id='voice_processor')
        def reset():
            return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                    None, None, None, None, None, gr.update(value=None, interactive=False),
                    gr.update(interactive=False), '', 'Strong', True, '**1. Add voice** → 2. Clean up → 3. Download',
                    gr.update(value=None, visible=False))
        again.click(reset, outputs=[step_one, step_two, step_three, selected, source, file_input,
                                    before, after, download, next_button, title, mode, tighten, breadcrumb, video_out], queue=False)
        history.change(open_record, history, [saved_before, saved_after, saved_download, saved_status, saved_video])
        refresh.click(lambda: gr.update(choices=history_choices()), outputs=history)
    return demo


demo = build_app()

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=1).launch(
        server_name='127.0.0.1', inbrowser='--no-browser' not in sys.argv,
        share='--share' in sys.argv, allowed_paths=[str(LIBRARY)], theme=THEME, css=CSS)
