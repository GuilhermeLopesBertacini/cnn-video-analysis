#!/usr/bin/env python3
"""Visualizador simples de inferência com YOLO.

Este script percorre os vídeos em um diretório (por padrão `selected_videos/3`),
executa inferência com o modelo YOLO e exibe os frames com as bounding boxes desenhadas.

Controles na janela:
- 'q' = sair do script
- 'n' = pular para o próximo vídeo

Uso:
    python teste.py --videos_dir selected_videos/3 --model assets/models/yolo8_nano_640_43750_otimizado.pt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import torch
from ultralytics import YOLO

class_map = {
    0: "LOW",
    1: "HIGH",
    2: "MULTI-OBJECT",
    3: "OBSTRUCTED"
}

def draw_boxes(image, boxes) -> None:
    """Desenha retângulos e rótulos na imagem (mutates image).

    image: imagem BGR pronta para exibição pelo OpenCV
    boxes: lista/iterável de Boxes (objeto retornado pelo ultralytics)
    """
    for box in (boxes or []):
        try:
            coords = box.xyxy[0].tolist()
        except Exception:
            # fallback
            coords = list(box.xyxy[0])

        x1, y1, x2, y2 = map(int, coords[:4])
        conf = float(box.conf[0]) if getattr(box, 'conf', None) is not None else None
        cls = int(box.cls[0]) if getattr(box, 'cls', None) is not None else None

        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        label = None
        if cls is not None and conf is not None:
            label = f"{class_map[cls]}:{conf:.2f}"
        elif conf is not None:
            label = f"{conf:.2f}"

        if label:
            # desenhar fundo para texto
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - h - 6), (x1 + w, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def process_video(model: YOLO, video_path: Path, device: str = 'cuda', window_name: Optional[str] = None):
    """Executa inferência em streaming e mostra frames com boxes.

    O usuário pode pressionar 'n' para avançar ao próximo vídeo, ou 'q' para sair.
    """
    window_name = window_name or video_path.name
    print(f"▶ Processando: {video_path}  on {device}")

    try:
        results = model.predict(source=str(video_path), stream=True, device=device, show=False, save=False)
    except Exception as e:
        print(f"❌ Erro ao executar predict em {video_path}: {e}")
        return

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    for result in results:
        img = result.orig_img
        if img is None:
            continue

        # Alguns retornos são RGB; converter para BGR para exibir corretamente com OpenCV
        try:
            if img.shape[2] != 3:
                img_disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_disp = img
        except Exception:
            img_disp = img

        draw_boxes(img_disp, result.boxes)

        cv2.imshow(window_name, img_disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Saindo por solicitação do usuário (q)")
            cv2.destroyAllWindows()
            sys.exit(0)
        if key == ord('n'):
            print("Próximo vídeo (n) solicitado pelo usuário")
            break

    cv2.destroyWindow(window_name)


def find_videos(videos_dir: Path, exts=None):
    exts = exts or ['.mp4', '.avi', '.mov', '.mkv']
    videos = []
    for ext in exts:
        videos.extend(sorted(videos_dir.glob(f"*{ext}")))
    return videos


def parse_args():
    p = argparse.ArgumentParser(description="Visualizador de inferência YOLO (desenhar bounding boxes)")
    p.add_argument('--class-id', type=str, help='ID da classe para filtrar vídeos')
    p.add_argument('--model', type=str, default='assets/models/yolo8_nano_640_43750_otimizado.pt', help='Caminho para o modelo YOLO')
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help="Dispositivo a usar: 'auto' detecta CUDA se disponível")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Usando dispositivo: {device}")
    videos_dir = Path(os.path.join('selected_videos', args.class_id))
    if not videos_dir.exists():
        print(f"Diretório não encontrado: {videos_dir}")
        return

    video_files = find_videos(videos_dir)
    if not video_files:
        print(f"Nenhum vídeo encontrado em: {videos_dir}")
        return

    print(f"Encontrados {len(video_files)} vídeos. Modelo: {args.model}")

    # Carregar modelo (uma vez)
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    # Processar vídeos sequencialmente para uso interativo com GPU
    for video_path in video_files:
        process_video(model, video_path, device=device)

    print("✔️  Todos os vídeos processados")


if __name__ == '__main__':
    main()