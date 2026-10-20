"""Passive live split-screen viewer: scene RGB and frame-matched DINO/SAM.

The local browser server and diagnostic models run in a spawned process. Its
single-slot input queue drops old frames; it never supplies observations or
commands to the grasp controller. All endpoints bind to loopback only.
"""
from collections import deque
import io
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>SO101 实时视觉调试</title><style>
body{background:#121922;color:#edf4ff;font:16px system-ui;margin:24px}h1{font-size:24px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}section{background:#202c3b;padding:16px;border-radius:12px}
img{width:100%;height:58vh;object-fit:contain;background:#0c1119}select,button{font:inherit;padding:7px;background:#edf4ff;border:0;border-radius:5px}
p{color:#b9cadb;line-height:1.6}#status{white-space:pre-wrap}.error{color:#ff9c9c}
@media(max-width:800px){.grid{grid-template-columns:1fr}img{height:40vh}}
</style><h1>SO101 实时视觉调试</h1>
<p>左侧是当前原始画面；右侧是最近完成处理的那一帧。绿色为 SAM 掩码，黄色为 Grounding 检测框。<br>
检测异步更新；右侧始终使用对应的原图，不把旧掩码叠在新帧上。<br>
这里显示独立的实时检测；抓取规划还会做深度和质量筛选，最终采用的掩码可能不同。此窗口不参与机械臂控制。</p>
<div class="grid"><section><h2>原始视频 <select id="left"><option value="overview">机械臂全景</option><option value="wrist">腕部原图</option></select></h2><img id="raw"></section>
<section><h2>处理后相机 <select id="right"><option value="overlay">Grounding + SAM</option><option value="grounding">仅检测框</option><option value="mask">二值掩码</option><option value="processed_raw">对应原图</option><option value="candidates">候选点与朝向</option></select></h2>
<p id="pose-controls" hidden><select id="candidate"><option value="">等待候选…</option></select><br>
紫点：三维中心；白线：平行夹爪开度；红箭头：接近方向；绿/蓝：夹爪 Y/Z 轴。<br>
候选图固定在推理原图；这是网络姿态，SO101 执行还会做工具适配和可达性筛选。评分不是成功概率。</p>
<img id="processed"></section></div>
<p id="status">等待仿真帧和检测模型…</p><p>运行结束后，最后的分屏图、检测图和掩码保存在本次输出目录的 live_debug/ 中。</p>
<script>
const byId=x=>document.getElementById(x);let stopped=false;
function images(){if(stopped)return;const t=Date.now();const mode=byId('right').value;byId('pose-controls').hidden=mode!=='candidates';byId('raw').src='/frame/'+byId('left').value+'?t='+t;byId('processed').src='/frame/'+mode+'?t='+t+'&candidate='+byId('candidate').value;}
setInterval(images,200);images();
let candidateVersion='';
async function status(){try{const d=await(await fetch('/status',{cache:'no-store'})).json();
const poses=await(await fetch('/candidate-data',{cache:'no-store'})).json();
const version=JSON.stringify([poses.trial,poses.source,poses.chosen_execution?.candidate_id,poses.execution_finished,poses.candidates?.length]);
if(version!==candidateVersion){candidateVersion=version;const select=byId('candidate');select.replaceChildren();
(poses.candidates||[]).forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent='#'+(p.rank+1)+' '+p.candidate_id+' | score '+p.score.toFixed(3)+' | '+(p.width_m*1000).toFixed(1)+'mm | '+p.status;select.append(o);});
if(!select.options.length){const o=document.createElement('option');o.value='';o.textContent='等待候选…';select.append(o);}else{select.value=poses.default_index;}}
byId('status').className=(d.error||d.candidate_error)?'error':'';
byId('status').textContent='回合 '+(d.trial??'—')+' | 当前帧 '+(d.current_frame??'—')+' | 当前仿真 '+(d.current_sim_s??0).toFixed(2)+'s\n'+
'处理帧 '+(d.processed_frame??'—')+' | 处理帧仿真 '+(d.processed_sim_s??0).toFixed(2)+'s | 结果墙钟年龄 '+(d.result_age_s??0).toFixed(1)+'s | '+
(d.error||d.candidate_error||d.message||'')+' | 丢弃旧帧 '+(d.dropped??0);
}catch(e){byId('status').textContent='连接已结束；保留最后画面。查看本地 live_debug/ 目录中的图片与状态。';stopped=true;}}
setInterval(status,1000);status();
</script></html>'''


def encode(image, format='JPEG'):
    stream = io.BytesIO()
    image.save(stream, format=format)
    return stream.getvalue()


def overlays(rgb, detection, mask):
    """Draw both stages on the very same RGB that produced their predictions."""
    original = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert('RGB')
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (original.height, original.width):
        raise ValueError('Live mask must match its source RGB frame')
    pixels = np.asarray(original).copy()
    pixels[mask] = (.55 * pixels[mask] + .45 * np.array([35, 235, 125])).astype(np.uint8)
    binary = Image.fromarray(mask.astype(np.uint8) * 255)
    edge = np.asarray(binary.filter(ImageFilter.MaxFilter(3))) != np.asarray(binary.filter(ImageFilter.MinFilter(3)))
    pixels[edge] = [175, 255, 210]
    images = {'overlay': Image.fromarray(pixels), 'grounding': original.copy(),
              'mask': binary.convert('RGB'), 'processed_raw': original}
    if detection.get('detected') and detection.get('box_xyxy'):
        box = np.asarray(detection['box_xyxy'], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
            raise ValueError('Invalid live detection box')
        for key in ('overlay', 'grounding'):
            draw = ImageDraw.Draw(images[key])
            draw.rectangle(tuple(box), outline='#ffe26b', width=2)
            score = detection.get('score')
            caption = str(detection.get('label') or 'target').encode('ascii', 'replace').decode()
            if isinstance(score, (float, int)):
                caption += f' {score:.3f}'
            draw.text((max(0., box[0]), max(0., box[1] - 12)), caption,
                      fill='#ffe26b', stroke_width=1, stroke_fill='black')
    return images


def split_image(overview, processed, caption):
    canvas = Image.new('RGB', (1280, 560), '#101824')
    for index, rgb in enumerate((overview, processed)):
        image = rgb if isinstance(rgb, Image.Image) else Image.fromarray(rgb)
        image = ImageOps.contain(image.convert('RGB'), (640, 480))
        canvas.paste(image, (640 * index + (640 - image.width) // 2, 34 + (480 - image.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), 'Raw scene RGB', fill='white')
    draw.text((654, 10), 'Wrist Grounding + SAM (same source frame)', fill='white')
    draw.text((14, 529), caption, fill='white')
    return canvas


def atomic_bytes(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(value)
    temporary.replace(path)


class State:
    def __init__(self):
        self.lock = threading.Lock()
        placeholder = Image.new('RGB', (640, 480), '#1e2c3c')
        ImageDraw.Draw(placeholder).text((24, 30), 'Waiting for camera / DINO + SAM...', fill='white')
        self.images = {k: encode(placeholder) for k in ('overview', 'wrist', 'overlay', 'grounding', 'mask', 'processed_raw', 'candidates')}
        self.candidate_snapshot = None
        self.info = {'message': 'loading models on first frame', 'dropped': 0}
        self.result_wall = None


def handler_for(state):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path == '/':
                content, mime = PAGE.encode(), 'text/html; charset=utf-8'
            elif path == '/status':
                with state.lock:
                    status = dict(state.info)
                    status['result_age_s'] = time.time() - state.result_wall if state.result_wall else None
                content, mime = json.dumps(status).encode(), 'application/json'
            elif path == '/candidate-data':
                with state.lock:
                    snapshot = state.candidate_snapshot
                content = json.dumps(snapshot.data() if snapshot else {'candidates': []}).encode()
                mime = 'application/json'
            elif path.startswith('/frame/'):
                key = path[len('/frame/'):]
                with state.lock:
                    content = state.images.get(key)
                    snapshot = state.candidate_snapshot
                if key == 'candidates' and snapshot is not None:
                    value = parse_qs(urlsplit(self.path).query).get('candidate', [''])[0]
                    try:
                        index = int(value) if value else None
                        if index is not None and not 0 <= index < len(snapshot.candidates):
                            raise ValueError('candidate index out of range')
                        content = snapshot.jpeg(index)
                    except ValueError:
                        self.send_error(400, 'Invalid candidate index')
                        return
                if content is None:
                    self.send_error(404)
                    return
                mime = 'image/jpeg'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass
    return Handler


def refresh_candidates(state, directory, previous_stamp):
    path = directory / 'candidate_snapshot.json'
    if not path.exists() or path.stat().st_mtime_ns == previous_stamp:
        return previous_stamp
    stamp = path.stat().st_mtime_ns
    from .candidate_debug import CandidateSnapshot
    snapshot = CandidateSnapshot(json.loads(path.read_text()))
    with state.lock:
        if snapshot.spec['trial'] != state.info.get('trial'):
            return previous_stamp
        state.candidate_snapshot = snapshot
        state.info.update(candidate_count=len(snapshot.candidates), candidate_source=snapshot.spec['proposals_dir'], candidate_error=None)
    atomic_bytes(directory / 'last_candidates.png', encode(snapshot.render(), 'PNG'))
    atomic_bytes(directory / 'candidate_data.json', (json.dumps(snapshot.data(), indent=2) + '\n').encode())
    return stamp


def diagnostic_perception(rgb, instruction, device, model_path):
    from .grounding import detect, default_model
    from .segmentation import segment
    image = Image.fromarray(rgb)
    box, detection = detect(image, instruction, default_model(), device)
    if box is None:
        return detection, np.zeros(rgb.shape[:2], dtype=bool), {'accepted': False, 'reason': 'not_detected', 'backend': 'sam'}
    mask, metadata = segment(image, box['box_xyxy'], backend='sam', model_path=model_path, device=device)
    return detection, mask, metadata


def process_frames(state, pending, condition, stopping, directory, instruction, device, model_path, interval):
    while not stopping.is_set():
        with condition:
            condition.wait_for(lambda: pending or stopping.is_set(), timeout=.5)
            if not pending or stopping.is_set():
                continue
            packet = pending.pop()
            pending.clear()
        start = time.time()
        try:
            detection, mask, segmentation = diagnostic_perception(packet['wrist'], instruction, device, model_path)
            images = overlays(packet['wrist'], detection, mask)
            meta = {'trial': packet['trial'], 'seed': packet['seed'], 'frame': packet['frame'],
                    'sim_time_s': packet['sim_time_s'], 'source_wall_time': packet['wall_time'],
                    'processing_s': time.time() - start, 'detection': detection,
                    'segmentation': segmentation, 'role': 'passive_debug_only_not_used_by_controller'}
            encoded = {key: encode(image) for key, image in images.items()}
            with state.lock:
                # Never display previous-episode masks over a reset/new episode.
                same_trial = state.info.get('trial') == packet['trial']
                if same_trial:
                    state.images.update(encoded)
                    state.result_wall = time.time()
                    state.info.update(processed_frame=packet['frame'], processed_sim_s=packet['sim_time_s'],
                                      message='DINO + SAM updated' if detection['detected'] else 'target not detected', error=None)
            for key, image in images.items():
                atomic_bytes(directory / ('last_' + key + '.png'), encode(image, 'PNG'))
            board = split_image(packet['overview'], images['overlay'],
                                f'Trial {packet["trial"]} | frame {packet["frame"]} | source sim {packet["sim_time_s"]:.2f}s | processing {meta["processing_s"]:.2f}s')
            atomic_bytes(directory / 'last_split.png', encode(board, 'PNG'))
            atomic_bytes(directory / 'last_perception.json', (json.dumps(meta, indent=2) + '\n').encode())
            with (directory / 'perception.jsonl').open('a') as stream:
                stream.write(json.dumps(meta) + '\n')
        except Exception as error:
            with state.lock:
                if state.info.get('trial') == packet['trial']:
                    state.info.update(error=f'{type(error).__name__}: {error}', message='diagnostic detection failed')
            with (directory / 'errors.log').open('a') as stream:
                import traceback
                traceback.print_exc(file=stream)
        stopping.wait(max(0., interval - (time.time() - start)))


def viewer_main(incoming, ready, directory, port, instruction, device, model_path, interval):
    """Child lifetime equals the simulation session; no unattended server remains."""
    import os
    from contextlib import redirect_stdout, redirect_stderr
    os.environ['OMP_NUM_THREADS'] = '1'
    directory = Path(directory)
    try:
        state = State()
        server = ThreadingHTTPServer(('127.0.0.1', port), handler_for(state))
        server.daemon_threads = True
    except Exception as error:
        ready.send({'error': str(error)})
        return
    ready.send({'url': f'http://127.0.0.1:{server.server_port}'})
    ready.close()
    pending, condition, stopping = deque(maxlen=1), threading.Condition(), threading.Event()
    serve = threading.Thread(target=server.serve_forever, daemon=True)
    serve.start()
    with (directory / 'worker.log').open('a', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        thread = threading.Thread(target=process_frames, args=(state, pending, condition, stopping,
                                  directory, instruction, device, model_path, interval), daemon=True)
        thread.start()
        previous_trial = None
        candidate_stamp = None
        try:
            while True:
                try:
                    candidate_stamp = refresh_candidates(state, directory, candidate_stamp)
                except Exception as error:
                    with state.lock:
                        state.info['candidate_error'] = f'{type(error).__name__}: {error}'
                try:
                    packet = incoming.get(timeout=.2)
                except queue.Empty:
                    continue
                if packet is None:
                    break
                if packet['trial'] != previous_trial:
                    blank = State()
                    with state.lock:
                        for key in ('overlay', 'grounding', 'mask', 'processed_raw', 'candidates'):
                            state.images[key] = blank.images[key]
                        state.candidate_snapshot = None
                        state.result_wall = None
                        state.info.update(trial=packet['trial'], processed_frame=None,
                                          processed_sim_s=None, error=None,
                                          candidate_count=0, candidate_source=None, candidate_error=None,
                                          message='waiting for this trial DINO + SAM')
                    previous_trial = packet['trial']
                raw = {'overview': encode(Image.fromarray(packet['overview'])),
                       'wrist': encode(Image.fromarray(packet['wrist']))}
                with state.lock:
                    state.images.update(raw)
                    state.info.update(trial=packet['trial'], seed=packet['seed'], current_frame=packet['frame'],
                                      current_sim_s=packet['sim_time_s'], dropped=packet['dropped'])
                with condition:
                    pending.append(packet)
                    condition.notify()
        finally:
            stopping.set()
            with condition:
                condition.notify_all()
            thread.join(timeout=3.)
            server.shutdown()
            server.server_close()
            with state.lock:
                atomic_bytes(directory / 'status.json', (json.dumps(dict(state.info, stopped=True), indent=2) + '\n').encode())


class LiveDebug:
    def __init__(self, directory, instruction, port=8765, device='cpu', model_path=None, interval=1., fps=8.):
        if not math.isfinite(interval) or interval <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Live interval/FPS must be positive and finite')
        if not 0 <= port <= 65535:
            raise ValueError('Invalid live viewer port')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        context = mp.get_context('spawn')
        self.incoming = context.Queue(maxsize=1)
        receiver, sender = context.Pipe(duplex=False)
        self.process = context.Process(target=viewer_main, args=(self.incoming, sender, str(directory),
             port, instruction, device, str(model_path) if model_path else None, interval), daemon=True)
        self.process.start()
        sender.close()
        if not receiver.poll(15):
            self.process.terminate()
            self.process.join()
            raise RuntimeError('Live viewer failed to start within 15 seconds')
        try:
            status = receiver.recv()
        except EOFError as error:
            raise RuntimeError('Live viewer exited during startup') from error
        finally:
            receiver.close()
        if 'error' in status:
            self.process.join(timeout=2.)
            raise RuntimeError('Live viewer: ' + status['error'])
        self.url, self.period, self.last = status['url'], 1 / fps, -math.inf
        self.frames, self.dropped, self.closed = 0, 0, False
        self.directory = directory
        (directory / 'viewer.json').write_text(json.dumps({'url': self.url, 'device': device,
            'interval_s': interval, 'display_max_fps': fps, 'role': 'passive_debug_only'}, indent=2) + '\n')

    def publish_candidates(self, proposals_dir, intrinsics_path, trial, candidates, execution=None):
        from .candidate_debug import execution_summary
        spec = {'trial': trial, 'proposals_dir': str(Path(proposals_dir).resolve()),
                'intrinsics_path': str(Path(intrinsics_path).resolve()),
                'execution': execution_summary(candidates, execution)}
        atomic_bytes(self.directory / 'candidate_snapshot.json', (json.dumps(spec) + '\n').encode())

    def offer(self, supplier, trial, seed, sim_time_s, force=False):
        now = time.monotonic()
        if self.closed or not force and now - self.last < self.period:
            return
        if not self.process.is_alive():
            raise RuntimeError(f'Live viewer exited unexpectedly; inspect {self.directory}')
        self.last = now
        overview, wrist = supplier()
        self.frames += 1
        packet = {'overview': overview, 'wrist': wrist, 'frame': self.frames,
                  'trial': trial, 'seed': seed, 'sim_time_s': sim_time_s,
                  'wall_time': time.time(), 'dropped': self.dropped}
        try:
            self.incoming.put_nowait(packet)
        except queue.Full:
            self.dropped += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.incoming.put(None, timeout=2.)
        except queue.Full:
            pass
        self.process.join(timeout=8.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=3.)
        self.incoming.cancel_join_thread()
        self.incoming.close()
