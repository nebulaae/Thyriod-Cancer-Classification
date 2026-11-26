"""
deploy/publish.py — package and publish the MedAI MVP to Hugging Face.

  1. Uploads checkpoints/medai_final.pth to the model repo <user>/medai-thyroid.
  2. Builds dist/hf_space (app.py, model.py, requirements, README with Space
     metadata, metrics_summary.json, optional example cases) and uploads it to
     the Gradio Space <user>/medai-thyroid-demo.

Prerequisite: `hf auth login` (or HF_TOKEN) with a write token, done by the user.

Usage:
  python deploy/publish.py [--login] [--user <hf-username>] [--private] [--examples deploy/examples] [--dry_run]
"""

import os, sys, json, shutil, argparse

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE_REQUIREMENTS = """torch==2.8.0
torchvision==0.23.0
numpy<2.2
pillow
huggingface_hub
"""

SPACE_README = """---
title: MedAI Thyroid Lesion Classifier
emoji: 🩺
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
short_description: US + cytology fusion for benign vs PTC (research demo)
---

# MedAI — thyroid lesion classifier (research demo)

Upload one thyroid **ultrasound image** and **2–7 cytology patches** of the same patient.
The model (dual ConvNeXt-Tiny encoders, transformer over the patch bag, [CLS]-token fusion)
returns the calibrated probability of papillary thyroid carcinoma (PTC), the decision at the
deployed Youden threshold, and the cytology patches ranked by relevance.

**Validation:** {validation}

Weights: [{model_repo}](https://huggingface.co/{model_repo}) · Code: [GitHub](https://github.com/nebulaae/image-proccessing-project)

Team MedAI (Usmanov Lazizbek), Central Asian University — course 4-SRC-10-804.

> ⚠ Research use only. Not a medical device; not for diagnosis or treatment decisions.
{examples_note}"""


def validation_card():
    """Summary of the 5-fold CV run (graphs/metrics_cv_summary.csv, results/pooled_oof.json)."""
    card = {'protocol': '5-fold patient-level CV, inner-validation selection', 'metrics': {}}
    s_path = os.path.join(ROOT, 'graphs', 'metrics_cv_summary.csv')
    if os.path.exists(s_path):
        s = pd.read_csv(s_path).set_index('Metric')
        for k in ('AUROC', 'Sensitivity', 'Specificity', 'Brier (cal)', 'ECE (cal)'):
            if k in s.index:
                card['metrics'][k] = f"{s.loc[k, 'Mean']:.3f} ± {s.loc[k, 'SD']:.3f}"
    p_path = os.path.join(ROOT, 'results', 'pooled_oof.json')
    if os.path.exists(p_path):
        pooled = json.load(open(p_path))
        card['metrics']['pooled OOF AUROC'] = f"{pooled['Pooled OOF AUROC']:.3f} {pooled['Pooled OOF AUROC CI']}"
    return card


def build_space(out_dir, model_repo, examples):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    for f in ('app.py', 'model.py'):
        shutil.copy(os.path.join(ROOT, f), out_dir)
    with open(os.path.join(out_dir, 'requirements.txt'), 'w') as fh:
        fh.write(SPACE_REQUIREMENTS)
    card = validation_card()
    with open(os.path.join(out_dir, 'metrics_summary.json'), 'w') as fh:
        json.dump(card, fh, indent=2)
    validation = ', '.join(f"{k} {v}" for k, v in card['metrics'].items()) or 'see repository'
    note = ''
    if examples:
        shutil.copytree(examples, os.path.join(out_dir, 'examples'))
        lic = os.path.join(examples, 'LICENSE_NOTE.md')
        note = '\n' + open(lic, encoding='utf-8').read() if os.path.exists(lic) else ''
    with open(os.path.join(out_dir, 'README.md'), 'w', encoding='utf-8') as fh:
        fh.write(SPACE_README.format(validation=validation, model_repo=model_repo, examples_note=note))
    # The Space downloads weights from the model repo at start-up
    with open(os.path.join(out_dir, 'app.py'), encoding='utf-8') as fh:
        app = fh.read()
    app = app.replace("os.environ.get('MEDAI_MODEL_REPO', 'nebulaae/medai-thyroid')",
                      f"os.environ.get('MEDAI_MODEL_REPO', '{model_repo}')")
    with open(os.path.join(out_dir, 'app.py'), 'w', encoding='utf-8') as fh:
        fh.write(app)
    print(f"Space bundle -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', default=None, help='Hugging Face username (default: the logged-in account)')
    ap.add_argument('--login', action='store_true', help='Prompt for a token in this terminal first')
    ap.add_argument('--model_name', default='medai-thyroid')
    ap.add_argument('--space_name', default='medai-thyroid-demo')
    ap.add_argument('--ckpt', default=os.path.join(ROOT, 'checkpoints', 'medai_final.pth'))
    ap.add_argument('--examples', default=None, help='Folder of example cases (only with a licence that allows it)')
    ap.add_argument('--private', action='store_true')
    ap.add_argument('--dry_run', action='store_true', help='Build dist/hf_space without uploading')
    args = ap.parse_args()

    if args.login:
        from huggingface_hub import login
        login()   # interactive prompt; the token is typed by the user and cached locally
    if args.user is None:
        from huggingface_hub import HfApi
        args.user = HfApi().whoami()['name']
    model_repo = f"{args.user}/{args.model_name}"
    space_repo = f"{args.user}/{args.space_name}"
    out_dir = build_space(os.path.join(ROOT, 'dist', 'hf_space'), model_repo, args.examples)
    if args.dry_run:
        return

    if not os.path.exists(args.ckpt):
        sys.exit(f"Checkpoint not found: {args.ckpt} (run: python main.py --final_only --data_root ...)")
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Logged in as {api.whoami()['name']}")
    api.create_repo(model_repo, repo_type='model', private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=args.ckpt, path_in_repo='medai_final.pth', repo_id=model_repo,
                    commit_message='Upload MedAI deployment checkpoint')
    api.create_repo(space_repo, repo_type='space', space_sdk='gradio', private=args.private, exist_ok=True)
    api.upload_folder(folder_path=out_dir, repo_id=space_repo, repo_type='space',
                      commit_message='Deploy MedAI demo')
    print(f"Model: https://huggingface.co/{model_repo}\nSpace: https://huggingface.co/spaces/{space_repo}")


if __name__ == '__main__':
    main()
