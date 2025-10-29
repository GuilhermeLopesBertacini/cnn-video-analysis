#!/usr/bin/env python3
"""
Script para selecionar vídeos cujos resultados contém deteções com class_id == 3.
Gera um arquivo JSON e TXT com a lista de vídeos encontrados e pode opcionalmente copiar
os vídeos para `selected_videos/class3/`.
"""
from pathlib import Path
import json
import shutil
import argparse

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
# these will be set by init_dirs
OUT_DIR = None
OUT_JSON = None
OUT_TXT = None


def init_dirs(class_id):
    """Inicializa variáveis de caminho usadas pelo script e cria estruturas mínimas.

    Define os nomes de saída (globais): OUT_DIR, OUT_JSON, OUT_TXT.
    Se class_id for None, usa 3 como padrão.
    """
    global OUT_DIR, OUT_JSON, OUT_TXT
    if class_id is None:
        class_id = 3
    OUT_DIR = ROOT / "selected_videos" / f"{class_id}"
    OUT_JSON = ROOT / f"selected_videos_{class_id}.json"
    OUT_TXT = ROOT / f"selected_videos_{class_id}.txt"

def find_selected(class_id_target=3):
    selected = []
    if not RESULTS_DIR.exists():
        print(f"Resultados não encontrados em {RESULTS_DIR}")
        return selected

    for sub in RESULTS_DIR.iterdir():
        if not sub.is_dir():
            continue
        if not sub.name.startswith("bb_centers_videos_camera_"):
            continue
        for json_file in sub.glob("*.json"):
            try:
                data = json.loads(json_file.read_text())
            except Exception as e:
                print(f"Falha ao ler {json_file}: {e}")
                continue
            centers = data.get("centers", [])
            found = any((c.get("class_id") == class_id_target) for c in centers)
            if found:
                video_path = data.get("video_path")
                # if video_path is relative to repo, resolve it against ROOT
                if video_path:
                    vp = (ROOT / video_path).resolve()
                else:
                    # fallback to constructing from video_name
                    name = data.get("video_name")
                    vp = None
                    if name:
                        # try possible occurrence directories (project root and assets)
                        candidate_dirs = [
                            ROOT / "occurrence_entering",
                            ROOT / "assets" / "occurrence_entering",
                        ]
                        for candidate in candidate_dirs:
                            if not candidate.exists():
                                continue
                            for camera_dir in candidate.glob("videos_camera_*"):
                                # look recursively for a file that matches by name
                                for f in camera_dir.rglob("*"):
                                    try:
                                        if f.is_file() and (f.name == name or f.stem == Path(name).stem):
                                            vp = f.resolve()
                                            break
                                    except Exception:
                                        continue
                                if vp:
                                    break
                            if vp:
                                break
                selected.append({
                    "video_name": data.get("video_name"),
                    "video_path": str(vp) if vp else None,
                    "source_json": str(json_file)
                })
    return selected


def save_selected(selected, copy=False):
    if OUT_JSON is None or OUT_DIR is None or OUT_TXT is None:
        raise RuntimeError("init_dirs must be called before save_selected()")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    with OUT_TXT.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write((item.get("video_path") or item.get("video_name")) + "\n")

    if copy:
        for item in selected:
            src = item.get("video_path")
            if src and Path(src).exists():
                dst = OUT_DIR / Path(src).name
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"Falha ao copiar {src} -> {dst}: {e}")
            else:
                print(f"Arquivo de vídeo não encontrado, pulando: {src}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seleciona vídeos com class_id nos resultados JSON.")
    parser.add_argument("--class-id", type=int, default=3, help="ID da classe a selecionar (default: 3)")
    parser.add_argument("--copy", action="store_true", help="Copiar vídeos selecionados para selected_videos/class_id/")
    args = parser.parse_args()
    init_dirs(args.class_id)
    sel = find_selected(args.class_id)
    print(f"Encontrados {len(sel)} vídeos com class_id == {args.class_id}")
    if len(sel) > 0:
        save_selected(sel, copy=args.copy)
        print(f"Lista salva em: {OUT_JSON} e {OUT_TXT}")
        if args.copy:
            print(f"Vídeos copiados para: {OUT_DIR}")
    else:
        print("Nenhum vídeo encontrado.")
