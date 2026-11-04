"""官方Windows headless渲染：每文档独立临时profile，不使用桌面输入。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import subprocess
import sys
from pdf2image import pdfinfo_from_path

HERE = Path(__file__).resolve().parent
RENDERER = Path('C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/documents/26.1026.10807/skills/documents/render_docx.py')
SOFFICE = Path('C:/Program Files/LibreOffice/program/soffice.exe')
POPPLER = Path('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(pair):
    index, row = pair
    source = HERE / 'documents' / row['output']
    assert sha(source) == row['output_sha256']
    out = HERE / 'docx_qa' / ['D1_v13', 'D2_v4', 'D3_v2'][index]
    out.mkdir(exist_ok=True)
    env = os.environ.copy()
    env['PATH'] = str(SOFFICE.parent) + os.pathsep + str(POPPLER) + os.pathsep + env.get('PATH', '')
    env['PYTHONIOENCODING'] = 'utf-8'
    command = [sys.executable, str(RENDERER), str(source), '--output_dir', str(out), '--emit_pdf', '--verbose']
    result = subprocess.run(command, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
    (out / 'render_log.txt').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'Official renderer failed for {source.name}; see {out / "render_log.txt"}')
    pdf = out / (source.stem + '.pdf')
    page_count = int(pdfinfo_from_path(str(pdf), poppler_path=str(POPPLER))['Pages'])
    pages = sorted(out.glob('page-*.png'), key=lambda p: int(p.stem.split('-')[-1]))
    assert len(pages) == page_count and page_count > 0
    assert sha(source) == row['output_sha256']
    summary = {
        'source': str(source), 'source_sha256': sha(source),
        'renderer': str(RENDERER), 'renderer_sha256': sha(RENDERER),
        'native_soffice': str(SOFFICE),
        'platform_constraint': 'Windows runtime has no bundled LibreOffice. Official renderer Windows native-executable branch uses a separate temporary UserInstallation profile with headless flags.',
        'desktop_Word_or_input_used': False,
        'pdf': str(pdf), 'pdf_sha256': sha(pdf), 'pages': page_count,
        'page_images': [{'page': i + 1, 'path': str(p), 'sha256': sha(p)} for i, p in enumerate(pages)],
        'visual_review': 'PENDING',
    }
    (out / 'render_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'document': source.name, 'pages': page_count, 'render_dir': str(out)}, ensure_ascii=False), flush=True)
    return summary


def main():
    rows = json.loads((HERE / 'documents' / 'document_build_manifest.json').read_text(encoding='utf-8'))
    assert SOFFICE.is_file() and RENDERER.is_file()
    with ThreadPoolExecutor(max_workers=3) as pool:
        summaries = list(pool.map(render, enumerate(rows)))
    (HERE / 'docx_qa' / 'current_render_manifest.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
