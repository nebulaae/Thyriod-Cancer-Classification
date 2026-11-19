"""
app.py — MedAI MVP web demo (Gradio).

Upload one thyroid ultrasound image and 2–7 cytology patches of the same
patient; the app returns the calibrated probability of papillary thyroid
carcinoma (PTC), the decision at the deployed Youden threshold, and the
cytology patches ranked by their relevance to the fused [CLS] token.

Weights: MEDAI_CKPT (local file) if set and present, otherwise the file
MEDAI_CKPT_FILE from the Hugging Face model repo MEDAI_MODEL_REPO.
Research use only — not a medical device.

Local run:  python app.py   (then open http://127.0.0.1:7860)
"""

import os, json, time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr

from model import MultimodalTransformer, get_eval_transform

MODEL_REPO = os.environ.get('MEDAI_MODEL_REPO', 'nebulaae/medai-thyroid')
CKPT_FILE = os.environ.get('MEDAI_CKPT_FILE', 'medai_final.pth')
LOCAL_CKPT = os.environ.get('MEDAI_CKPT', os.path.join('checkpoints', 'medai_final.pth'))
MAX_PATCHES = 7
HERE = os.path.dirname(os.path.abspath(__file__))

torch.set_num_threads(max(1, os.cpu_count() or 1))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _resolve_ckpt():
    if os.path.exists(LOCAL_CKPT):
        return LOCAL_CKPT
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=MODEL_REPO, filename=CKPT_FILE)


def load():
    path = _resolve_ckpt()
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = MultimodalTransformer(ckpt['cfg'], pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    deploy = ckpt.get('deploy', {})
    return model, ckpt['cfg'], deploy


MODEL, CFG, DEPLOY = load()
TEMPERATURE = float(DEPLOY.get('temperature', 1.0))
THRESHOLD = float(DEPLOY.get('threshold', 0.5))
TF = get_eval_transform(CFG.get('img_size', 224))

_card_path = os.path.join(HERE, 'metrics_summary.json')
CARD = json.load(open(_card_path)) if os.path.exists(_card_path) else {}


def _to_rgb(x):
    if isinstance(x, Image.Image):
        return x.convert('RGB')
    if isinstance(x, np.ndarray):
        return Image.fromarray(x).convert('RGB')
    return Image.open(x).convert('RGB')


def analyse(us_image, patch_files):
    if us_image is None:
        raise gr.Error('Please upload the ultrasound image.')
    if not patch_files:
        raise gr.Error('Please upload 2–7 cytology patches.')
    files = list(patch_files)[:MAX_PATCHES]
    notes = []
    if len(patch_files) > MAX_PATCHES:
        notes.append(f'Only the first {MAX_PATCHES} patches were used.')
    if len(files) < 2:
        notes.append('The model was trained with 2–7 patches; a single patch is outside the validated range.')

    t0 = time.perf_counter()
    us_img = _to_rgb(us_image)
    patches = [_to_rgb(f) for f in files]
    us = TF(us_img).unsqueeze(0).to(DEVICE)
    path = torch.stack([TF(p) for p in patches]).unsqueeze(0).to(DEVICE)
    mask = torch.ones(1, len(patches), dtype=torch.bool, device=DEVICE)
    with torch.inference_mode():
        logit, feats = MODEL(us, path, mask, return_feats=True)
        cls_f = feats['fused'][0, 0]
        sims = F.cosine_similarity(cls_f.unsqueeze(0), feats['fused'][0, 2:2 + len(patches)]).cpu().numpy()
    p = float(torch.sigmoid(logit.float() / TEMPERATURE).item())
    ms = (time.perf_counter() - t0) * 1000

    decision = 'PTC (suspicious for papillary carcinoma)' if p >= THRESHOLD else 'Benign'
    label = {'PTC': p, 'Benign': 1 - p}
    order = np.argsort(-sims)
    gallery = [(patches[j], f"#{rank + 1} · patch {j + 1} · relevance {sims[j]:.3f}")
               for rank, j in enumerate(order)]
    md = [f"### Decision: **{decision}**",
          f"- Calibrated P(PTC) = **{p:.3f}** (threshold {THRESHOLD:.3f}, temperature {TEMPERATURE:.2f})",
          f"- Most relevant cytology patch: **patch {int(order[0]) + 1}**",
          f"- Inference time: {ms:.0f} ms on {DEVICE.type.upper()}"]
    md += [f"- ⚠ {n}" for n in notes]
    md.append("\n_Research prototype — not for clinical use._")
    return label, '\n'.join(md), gallery


def _about():
    lines = ["**MedAI** — interpretable multimodal deep learning for thyroid lesion classification "
             "(benign vs papillary thyroid carcinoma). Dual ConvNeXt-Tiny encoders, a transformer over the "
             "cytology patch bag and a [CLS]-token fusion transformer."]
    if CARD:
        lines.append(f"**Validation ({CARD.get('protocol', '5-fold patient-level CV')}):** "
                     + ', '.join(f"{k} {v}" for k, v in CARD.get('metrics', {}).items()))
    if DEPLOY:
        lines.append(f"Deployed model trained {DEPLOY.get('trained_at', '?')} on "
                     f"{DEPLOY.get('n_train_patients', '?')} patients; calibration and threshold fitted on "
                     f"{DEPLOY.get('n_selection_patients', '?')} held-out patients.")
    lines.append("⚠ **Research use only.** Not a medical device; do not use for diagnosis or treatment.")
    return '\n\n'.join(lines)


def build_ui():
    with gr.Blocks(title='MedAI — Thyroid lesion classifier', theme=gr.themes.Soft()) as demo:
        gr.Markdown('# MedAI · Thyroid lesion classifier (ultrasound + cytology)')
        gr.Markdown(_about())
        with gr.Row():
            with gr.Column():
                us_in = gr.Image(type='pil', label='Ultrasound image', height=280)
                patch_in = gr.File(file_count='multiple', type='filepath',
                                   file_types=['image', '.tif', '.tiff'], label='Cytology patches (2–7)')
                btn = gr.Button('Analyse', variant='primary')
            with gr.Column():
                out_label = gr.Label(num_top_classes=2, label='Prediction')
                out_md = gr.Markdown()
        out_gallery = gr.Gallery(label='Cytology patches ranked by relevance', columns=4, height='auto')
        btn.click(analyse, [us_in, patch_in], [out_label, out_md, out_gallery])

        ex_dir = os.path.join(HERE, 'examples')
        if os.path.isdir(ex_dir):
            cases = []
            for case in sorted(os.listdir(ex_dir)):
                d = os.path.join(ex_dir, case)
                us = [f for f in os.listdir(d) if f.startswith('us')]
                pats = sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith('patch'))
                if us and pats:
                    cases.append([os.path.join(d, us[0]), pats])
            if cases:
                gr.Examples(cases, [us_in, patch_in], label='Example cases from the public dataset (see Space README)')
    return demo


if __name__ == '__main__':
    build_ui().launch(server_name=os.environ.get('GRADIO_SERVER_NAME', '127.0.0.1'))
