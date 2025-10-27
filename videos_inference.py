import json
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import torch
from ultralytics import YOLO


def process_video(model: YOLO, video_path: Path, device: str = 'cuda') -> Dict:
    """
    Processa um vídeo e extrai os centros de massa das bounding boxes.
    
    Args:
        model: Modelo YOLO carregado
        video_path: Caminho para o arquivo de vídeo
        device: Dispositivo para inferência ('cuda' ou 'cpu')
    
    Returns:
        Dicionário com informações do vídeo e centros de massa
    """
    bb_centers_of_mass = []
    frame_shapes = None
    total_detections = 0
    
    results = model.predict(
        source=str(video_path), 
        show=False, 
        save=False,  # Não salvar vídeos automaticamente para economizar espaço
        stream=True, 
        device=device,
        verbose=False  # Reduzir output verboso
    )
    
    for result in results:
        boxes = result.boxes
        
        # Capturar shape do frame (apenas uma vez)
        if frame_shapes is None and len(boxes) > 0:
            frame_shapes = result.orig_shape
        
        for box in boxes:
            cords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = cords[0], cords[1], cords[2], cords[3]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Adicionar também a confiança e classe se disponível
            confidence = float(box.conf[0]) if box.conf is not None else None
            class_id = int(box.cls[0]) if box.cls is not None else None
            
            bb_centers_of_mass.append({
                'center_x': center_x,
                'center_y': center_y,
                'confidence': confidence,
                'class_id': class_id
            })
            total_detections += 1
    
    return {
        'video_name': video_path.stem,
        'video_path': str(video_path),
        'frame_shape': frame_shapes,
        'total_detections': total_detections,
        'centers': bb_centers_of_mass
    }


def save_results(video_data: Dict, output_dir: Path, video_name: str):
    """
    Salva os resultados em diferentes formatos.
    
    Args:
        video_data: Dados processados do vídeo
        output_dir: Diretório de saída
        video_name: Nome do vídeo (sem extensão)
    """
    # Criar diretório de saída se não existir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar JSON completo com todas as informações
    json_path = output_dir / f"{video_name}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(video_data, f, indent=2)
    
    # Salvar TXT no formato original (apenas coordenadas)
    txt_path = output_dir / f"{video_name}.txt"
    with open(txt_path, 'w') as f:
        for center in video_data['centers']:
            f.write(f"({center['center_x']}), ({center['center_y']})\n")


def main():
    """Função principal para processar todos os vídeos."""
    
    # Configurações
    MODEL_PATH = "assets/models/yolo8_nano_640_43750_otimizado.pt"
    VIDEOS_DIR = Path("assets/occurrence_entering/videos_camera_0/videos")
    OUTPUT_DIR = Path("results/bb_centers")
    
    # Verificar disponibilidade de CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Usando dispositivo: {device.upper()}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    # Carregar modelo
    print(f"\n📦 Carregando modelo: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Buscar todos os vídeos
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(VIDEOS_DIR.glob(f"*{ext}")))
    
    if not video_files:
        print(f"❌ Nenhum vídeo encontrado em: {VIDEOS_DIR}")
        return
    
    print(f"\n📹 Encontrados {len(video_files)} vídeos para processar\n")
    
    # Processar cada vídeo
    all_results = []
    
    for video_path in tqdm(video_files, desc="Processando vídeos", unit="vídeo"):
        try:
            # Processar vídeo
            video_data = process_video(model, video_path, device=device)
            all_results.append(video_data)
            
            # Salvar resultados individuais
            save_results(video_data, OUTPUT_DIR, video_path.stem)
            
        except Exception as e:
            print(f"\n❌ Erro ao processar {video_path.name}: {str(e)}")
            continue
    
    # Salvar sumário geral
    summary = {
        'total_videos': len(all_results),
        'total_detections': sum(v['total_detections'] for v in all_results),
        'device_used': device,
        'model_path': MODEL_PATH,
        'videos': [
            {
                'name': v['video_name'],
                'detections': v['total_detections'],
                'frame_shape': v['frame_shape']
            }
            for v in all_results
        ]
    }
    
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Salvar todos os centros em um único arquivo consolidado
    consolidated_path = OUTPUT_DIR / "all_centers_consolidated.txt"
    with open(consolidated_path, 'w') as f:
        for video_data in all_results:
            f.write(f"# Video: {video_data['video_name']}\n")
            for center in video_data['centers']:
                f.write(f"({center['center_x']}), ({center['center_y']})\n")
            f.write("\n")
    
    print(f"\n✅ Processamento concluído!")
    print(f"   📊 Total de vídeos: {summary['total_videos']}")
    print(f"   🎯 Total de detecções: {summary['total_detections']}")
    print(f"   💾 Resultados salvos em: {OUTPUT_DIR}")
    print(f"   📄 Sumário: {summary_path}")


if __name__ == "__main__":
    main()