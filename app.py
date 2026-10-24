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
:root, .dark {
  --vs-bg: #07090c; --vs-panel: #0e1217; --vs-panel-2: #141a21; --vs-line: #222b35; --vs-line-2: #2e3a46;
  --vs-text: #eef3f1; --vs-muted: #8b9a96; --vs-accent: #2ee59d; --vs-accent-2: #b8f35a; --vs-accent-ink: #04140d;
  --body-background-fill: var(--vs-bg) !important; --background-fill-primary: var(--vs-panel) !important;
  --background-fill-secondary: var(--vs-panel-2) !important; --block-background-fill: var(--vs-panel-2) !important;
  --block-border-color: var(--vs-line) !important; --border-color-primary: var(--vs-line) !important;
  --border-color-accent: var(--vs-accent) !important; --body-text-color: var(--vs-text) !important;
  --body-text-color-subdued: var(--vs-muted) !important; --block-title-text-color: var(--vs-muted) !important;
  --block-label-text-color: var(--vs-muted) !important; --block-label-background-fill: transparent !important;
  --block-label-border-color: transparent !important; --block-info-text-color: var(--vs-muted) !important;
  --input-background-fill: var(--vs-panel) !important; --input-border-color: var(--vs-line) !important;
  --color-accent: var(--vs-accent) !important; --color-accent-soft: rgba(46, 229, 157, .12) !important;
  --checkbox-background-color-selected: var(--vs-accent) !important; --checkbox-border-color-selected: var(--vs-accent) !important;
  --button-primary-background-fill: var(--vs-accent) !important; --button-primary-background-fill-hover: #5cf0b5 !important;
  --button-primary-text-color: var(--vs-accent-ink) !important; --button-primary-border-color: transparent !important;
  --button-secondary-background-fill: transparent !important; --button-secondary-background-fill-hover: var(--vs-panel-2) !important;
  --button-secondary-text-color: var(--vs-text) !important; --button-secondary-border-color: var(--vs-line-2) !important;
  --block-radius: 16px !important; --button-large-radius: 14px !important; --button-small-radius: 10px !important;
  --loader-color: var(--vs-accent) !important; --color-grey-50: var(--vs-panel-2) !important;
  --link-text-color: var(--vs-accent) !important; --shadow-drop: none !important; --shadow-drop-lg: none !important;
}
body, gradio-app { background: radial-gradient(900px 420px at 50% -120px, rgba(46, 229, 157, .16), transparent 70%), var(--vs-bg) !important; }
.gradio-container { max-width: 680px !important; margin: 0 auto !important; padding: 40px 16px 24px !important; }
footer { display: none !important; }

#brand { text-align: center; }
#brand .pill { display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border: 1px solid var(--vs-line-2);
  border-radius: 999px; font-size: 12px; letter-spacing: 1.6px; color: var(--vs-muted); background: rgba(255, 255, 255, .02); }
#brand .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--vs-accent); box-shadow: 0 0 10px var(--vs-accent); }
#hero { text-align: center; margin: 18px 0 26px; }
#hero h1 { font-size: clamp(34px, 6vw, 50px); line-height: 1.05; letter-spacing: -1.5px; font-weight: 800; margin: 0 0 12px; color: var(--vs-text); }
#hero h1 span { background-image: linear-gradient(90deg, var(--vs-accent), var(--vs-accent-2)) !important; -webkit-background-clip: text; background-clip: text; color: transparent; -webkit-text-fill-color: transparent; }
#hero p { color: var(--vs-muted); font-size: 16px; margin: 0; }
#hero .bars { display: flex; justify-content: center; align-items: center; gap: 4px; height: 28px; margin-bottom: 16px; }
#hero .bars i { display: block; width: 4px; border-radius: 4px; background: linear-gradient(var(--vs-accent-2), var(--vs-accent)); animation: vs-eq 1.1s ease-in-out infinite; }
@keyframes vs-eq { 0%, 100% { transform: scaleY(.35); } 50% { transform: scaleY(1); } }
@media (prefers-reduced-motion: reduce) { #hero .bars i { animation: none; } }

.tabs > .tab-wrapper, .tab-container { justify-content: center !important; border: none !important; }
.tab-container { background: var(--vs-panel); border: 1px solid var(--vs-line) !important; border-radius: 999px; padding: 4px !important;
  width: fit-content !important; margin: 0 auto 22px !important; gap: 4px; height: auto !important; }
.tab-container button { border-radius: 999px !important; padding: 8px 20px !important; border: none !important; color: var(--vs-muted) !important; font-weight: 600; }
.tab-container button.selected { background: var(--vs-panel-2) !important; color: var(--vs-text) !important; }
.tab-container button.selected::after { display: none !important; }

#stepper { margin: 0 0 16px; }
#stepper ol { list-style: none; display: flex; align-items: center; gap: 10px; padding: 0; margin: 0; justify-content: center; }
#stepper li { margin: 0 !important; padding: 0 !important; display: flex; align-items: center; gap: 8px; color: var(--vs-muted); font-size: 14px; font-weight: 600; white-space: nowrap; }
#stepper li b { display: grid; place-items: center; width: 26px; height: 26px; border-radius: 50%; border: 1px solid var(--vs-line-2); font-size: 13px; }
#stepper li.on { color: var(--vs-text); }
#stepper li.on b { background: var(--vs-accent); border-color: var(--vs-accent); color: var(--vs-accent-ink); box-shadow: 0 0 0 4px rgba(46, 229, 157, .15); }
#stepper li.done b { background: transparent; border-color: var(--vs-accent); color: var(--vs-accent); }
#stepper .line { width: 34px; height: 1px; background: var(--vs-line-2); }
@media (max-width: 520px) { #stepper li span { display: none; } }

.step-card { background: var(--vs-panel) !important; border: 1px solid var(--vs-line) !important; border-radius: 24px !important;
  padding: 26px !important; gap: 16px !important; box-shadow: 0 30px 60px -30px rgba(0, 0, 0, .7) !important; }
.step-card h2 { font-size: 22px !important; margin: 0 0 4px !important; letter-spacing: -.3px; }
.step-card p { color: var(--vs-muted) !important; margin: 0 !important; }

.choice-row { gap: 12px !important; }
.choice { min-height: 64px !important; border-radius: 16px !important; font-size: 16px !important; font-weight: 700 !important;
  border: 1px solid var(--vs-line-2) !important; background: var(--vs-panel-2) !important; color: var(--vs-muted) !important;
  justify-content: center; gap: 10px; transition: border-color .15s, color .15s, background .15s; }
.choice:hover { color: var(--vs-text) !important; border-color: #3d4b59 !important; }
.choice.picked { border-color: var(--vs-accent) !important; color: var(--vs-text) !important; background: rgba(46, 229, 157, .08) !important; }
.choice::before { content: ''; width: 20px; height: 20px; background: currentColor; flex: none;
  -webkit-mask: var(--icon) center / contain no-repeat; mask: var(--icon) center / contain no-repeat; }
#choose-file { --icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E"); }
#choose-mic { --icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='9' y='2' width='6' height='12' rx='3'/%3E%3Cpath d='M19 10v1a7 7 0 0 1-14 0v-1'/%3E%3Cline x1='12' y1='18' x2='12' y2='22'/%3E%3C/svg%3E"); }

#dropzone { border: 1.5px dashed var(--vs-line-2) !important; border-radius: 18px !important; background: rgba(255, 255, 255, .015) !important; transition: border-color .15s, background .15s; }
#dropzone:hover { border-color: var(--vs-accent) !important; background: rgba(46, 229, 157, .05) !important; }
#dropzone button, #dropzone .wrap { color: var(--vs-muted) !important; font-size: 15px; }
#dropzone .icon-wrap svg, #dropzone svg { color: var(--vs-accent) !important; }
.hint { font-size: 13px; text-align: center; color: var(--vs-muted); }

.primary-action { min-height: 58px !important; font-size: 17px !important; font-weight: 800 !important; border-radius: 16px !important; letter-spacing: .2px; }
button.primary-action:not(:disabled) { box-shadow: 0 10px 30px -10px rgba(46, 229, 157, .55); }
button.primary-action:disabled { background: var(--vs-panel-2) !important; color: #55635f !important; border: 1px solid var(--vs-line) !important; opacity: 1 !important; cursor: not-allowed; }
.ghost { color: var(--vs-muted) !important; border: none !important; background: transparent !important; box-shadow: none !important; }
.ghost:hover { color: var(--vs-text) !important; }
.compare > div { border-radius: 16px !important; }
#after-audio { border-color: rgba(46, 229, 157, .45) !important; }
.summary { text-align: center; font-size: 13px; }
.summary p { color: var(--vs-muted) !important; }

#history-pick { background: transparent !important; border: none !important; padding: 0 !important; }
.refresh { border: 1px solid var(--vs-line-2) !important; border-radius: 12px !important; }
[data-testid=status-tracker]:not(.hide) { background: rgba(14, 18, 23, .94) !important; border-radius: 16px; z-index: 5; }
[data-testid=status-tracker], [data-testid=status-tracker] * { font-family: inherit !important; color: var(--vs-text) !important; font-size: 14px !important; }
.progress-bar { background: var(--vs-accent) !important; }
#privacy { text-align: center; margin-top: 20px; }
#privacy span { display: inline-block; line-height: 1.4; color: var(--vs-muted); font-size: 13px; padding: 8px 14px; border-radius: 999px; border: 1px solid var(--vs-line); }
@media (max-width: 520px) { .step-card { padding: 20px !important; } .primary-action { font-size: 16px !important; } }
"""
THEME = gr.themes.Base(primary_hue='emerald', neutral_hue='slate',
                       font=[gr.themes.GoogleFont('Inter'), 'system-ui', 'sans-serif'])
WAVE = gr.WaveformOptions(waveform_color='#3a4652', waveform_progress_color='#2ee59d')
STEPS = ('Add voice', 'Clean up', 'Download')
FORCE_DARK = "() => { document.body.classList.add('dark'); }"  # the studio has one look, whatever the OS theme


def stepper(current):
    items = []
    for n, name in enumerate(STEPS, 1):
        state = 'done' if n < current else 'on' if n == current else ''
        items.append(f"<li class='{state}'><b>{'✓' if n < current else n}</b><span>{name}</span></li>")
    return "<ol>" + "<li class='line' aria-hidden='true'></li>".join(items) + "</ol>"


def ready_for_step_two(path):
    if not path or not pathlib.Path(path).is_file():
        raise gr.Error('Add a file or record your voice first.')
    return gr.update(visible=False), gr.update(visible=True), path, stepper(2)


def finish_voice(path, strength, title, tighten, progress=gr.Progress()):
    before, after, download, summary, history, video = enhance(path, strength, title, tighten, progress)
    return (before, after, download, f'Saved on this computer · {summary}', history,
            gr.update(visible=False), gr.update(visible=True), stepper(4), video)


def build_app():
    with gr.Blocks(title='AI Voice Studio', analytics_enabled=False) as demo:
        selected = gr.State(None)
        gr.HTML("<span class='pill'><span class='dot'></span>AI VOICE STUDIO · BY AMINE</span>", elem_id='brand')
        gr.HTML("<div class='bars' aria-hidden='true'>"
                + ''.join(f"<i style='height:{h}px;animation-delay:{d}s'></i>" for h, d in
                          ((10, 0), (18, .3), (26, .15), (14, .45), (28, .05), (20, .35), (12, .2), (22, .5), (9, .1)))
                + "</div><h1>Make your voice<br><span>sound better.</span></h1>"
                "<p>Three easy steps. Everything stays on your computer.</p>", elem_id='hero')
        with gr.Tabs():
            with gr.Tab('New voice'):
                steps = gr.HTML(stepper(1), elem_id='stepper')
                with gr.Column(visible=True, elem_classes='step-card') as step_one:
                    gr.Markdown('## Add your voice\nUpload a voice note or video, or record yourself now.')
                    with gr.Row(elem_classes='choice-row'):
                        choose_file = gr.Button('Upload a file', elem_id='choose-file', elem_classes=['choice', 'picked'])
                        choose_mic = gr.Button('Record now', elem_id='choose-mic', elem_classes='choice')
                    with gr.Column(visible=True) as import_box:
                        file_input = gr.File(show_label=False, file_types=['audio', 'video'], type='filepath', height=170, elem_id='dropzone')
                        gr.Markdown('MP3, WAV, M4A, OGG, MP4, MOV · a video gets its new voice too', elem_classes='hint')
                    with gr.Column(visible=False) as record_box:
                        source = gr.Audio(show_label=False, sources=['microphone'], type='filepath', format='wav', waveform_options=WAVE)
                        gr.Markdown('Press **Record**, speak close to the mic, then **Stop**. Allow microphone access when asked.', elem_classes='hint')
                    next_button = gr.Button('Continue', variant='primary', interactive=False, elem_classes='primary-action')
                with gr.Column(visible=False, elem_classes='step-card') as step_two:
                    gr.Markdown('## Clean up your voice\nOne click removes noise and echo, then balances the volume.')
                    preview = gr.Audio(label='Your voice', interactive=False, waveform_options=WAVE)
                    go = gr.Button('✨ Make my voice better', variant='primary', elem_classes='primary-action')
                    gr.Markdown('Takes a minute or two on a GPU. Keep this page open.', elem_classes='hint')
                    with gr.Accordion('Optional settings', open=False):
                        title = gr.Textbox(label='Name', placeholder='My recording', max_lines=1)
                        mode = gr.Radio(['Strong', 'Soft'], value='Strong', label='Cleanup', info='Strong is recommended. Soft keeps more of the room.')
                        tighten = gr.Checkbox(True, label='Shorten long pauses', info='Videos always keep their timing so the voice stays on the lips.')
                    back = gr.Button('← Use another recording', size='sm', elem_classes='ghost')
                with gr.Column(visible=False, elem_classes='step-card') as step_three:
                    gr.Markdown('## Your voice is ready\nPlay Before, then After, and hear the difference.')
                    with gr.Row(equal_height=True, elem_classes='compare'):
                        before = gr.Audio(label='Before', interactive=False, waveform_options=WAVE)
                        after = gr.Audio(label='After ✨', interactive=False, elem_id='after-audio', waveform_options=WAVE)
                    download = gr.DownloadButton('Download my voice', interactive=False, variant='primary', elem_classes='primary-action')
                    video_out = gr.Video(label='Your video with the new voice', interactive=False, visible=False, height=380)
                    status = gr.Markdown(elem_classes='summary')
                    again = gr.Button('＋ Start a new voice', size='sm', elem_classes='ghost')
            with gr.Tab('Saved voices'):
                with gr.Column(elem_classes='step-card'):
                    gr.Markdown('## Saved voices\nEvery voice you clean up is kept here, on this computer.')
                    with gr.Row(equal_height=True):
                        history = gr.Dropdown(choices=history_choices(), value=None, label='Choose a recording', show_label=False,
                                              interactive=True, scale=5, elem_id='history-pick')
                        refresh = gr.Button('Refresh', size='sm', scale=1, min_width=96, elem_classes='refresh')
                    with gr.Row(equal_height=True, elem_classes='compare'):
                        saved_before = gr.Audio(label='Before', interactive=False, waveform_options=WAVE)
                        saved_after = gr.Audio(label='After ✨', interactive=False, waveform_options=WAVE)
                    saved_download = gr.DownloadButton('Download my voice', interactive=False, variant='primary', elem_classes='primary-action')
                    saved_video = gr.Video(label='Your video with the new voice', interactive=False, visible=False, height=380)
                    saved_status = gr.Markdown(elem_classes='summary')
        gr.HTML('<span>🔒 Only on your computer · No account needed</span>', elem_id='privacy')

        def choose(recording):
            return (gr.update(visible=not recording), gr.update(visible=recording), None,
                    None, None, gr.update(interactive=False),
                    gr.update(elem_classes=['choice'] + ([] if recording else ['picked'])),
                    gr.update(elem_classes=['choice'] + (['picked'] if recording else [])))
        choice_outputs = [import_box, record_box, selected, source, file_input, next_button, choose_file, choose_mic]
        choose_file.click(lambda: choose(False), outputs=choice_outputs, queue=False)
        choose_mic.click(lambda: choose(True), outputs=choice_outputs, queue=False)
        def set_voice(path):
            return path, gr.update(interactive=bool(path))
        source.input(set_voice, source, [selected, next_button], queue=False)
        file_input.upload(set_voice, file_input, [selected, next_button], queue=False)
        file_input.clear(lambda: (None, gr.update(interactive=False)), outputs=[selected, next_button], queue=False)
        next_button.click(ready_for_step_two, selected, [step_one, step_two, preview, steps])
        back.click(lambda: (gr.update(visible=True), gr.update(visible=False), stepper(1)),
                   outputs=[step_one, step_two, steps], queue=False)
        go.click(finish_voice, [selected, mode, title, tighten],
                 [before, after, download, status, history, step_two, step_three, steps, video_out],
                 concurrency_limit=1, concurrency_id='voice_processor', show_progress_on=[preview])
        def reset():
            return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                    None, None, None, None, None, gr.update(value=None, interactive=False),
                    gr.update(interactive=False), '', 'Strong', True, stepper(1),
                    gr.update(value=None, visible=False))
        again.click(reset, outputs=[step_one, step_two, step_three, selected, source, file_input,
                                    before, after, download, next_button, title, mode, tighten, steps, video_out], queue=False)
        history.change(open_record, history, [saved_before, saved_after, saved_download, saved_status, saved_video])
        refresh.click(lambda: gr.update(choices=history_choices()), outputs=history)
    return demo


demo = build_app()

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=1).launch(
        server_name='127.0.0.1', inbrowser='--no-browser' not in sys.argv,
        share='--share' in sys.argv, allowed_paths=[str(LIBRARY)], theme=THEME, css=CSS, js=FORCE_DARK)
