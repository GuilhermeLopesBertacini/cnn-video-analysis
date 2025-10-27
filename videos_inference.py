import json
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import torch
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count
import multiprocessing

# Globals used by worker processes
global_model = None
worker_device = 'cpu'
# A module-level model path used by workers if necessary. Set in main().
MODEL_PATH_GLOBAL: str | None = None


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
        if frame_shapes is None and boxes is not None and len(boxes) > 0:
            frame_shapes = result.orig_shape

        for box in (boxes or []):
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


def init_worker(model_path: str, device_arg: str):
    """
    Initializer for worker processes: load the YOLO model once per worker.
    """
    global global_model, worker_device
    worker_device = device_arg
    # Reduce verbosity when loading in workers
    try:
        global_model = YOLO(model_path)
    except Exception as e:
        # If a worker cannot load the model, print to stderr so user can see it
        print(f"Worker failed to load model: {e}")


def process_video_worker(video_path_str: str) -> Dict:
    """
    Worker wrapper that calls process_video using the worker's global_model.
    Returns a result dict or a dict with an 'error' field on failure.
    """
    global global_model, worker_device
    video_path = Path(video_path_str)
    try:
        # If for some reason the global_model wasn't initialized, fall back to loading locally
        if global_model is None:
            # Fall back to a model path stored in module-level var (set in main)
            if MODEL_PATH_GLOBAL:
                model = YOLO(MODEL_PATH_GLOBAL)
                return process_video(model, video_path, device=worker_device)
            else:
                raise RuntimeError("Model not initialized in worker and no MODEL_PATH_GLOBAL set")
        else:
            return process_video(global_model, video_path, device=worker_device)
    except Exception as e:
        return {
            'video_name': video_path.stem,
            'video_path': str(video_path),
            'frame_shape': None,
            'total_detections': 0,
            'centers': [],
            'error': str(e)
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
    cameras_dir = Path("assets/occurrence_entering/")
    # Verificar disponibilidade de CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Usando dispositivo: {device.upper()}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    # Carregar modelo
    print(f"\nCarregando modelo: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    for cam_dir in cameras_dir.iterdir():
        cam_name = cam_dir.name
        print(f"\nProcessando vídeos da câmera: {cam_name}\n")
        VIDEOS_DIR = Path(f"assets/occurrence_entering/{cam_name}/videos")
        OUTPUT_DIR = Path(f"results/bb_centers_{cam_name}")
        
        # Buscar todos os vídeos
        video_files = list(VIDEOS_DIR.glob(f"*.mp4"))
        
        if not video_files:
            print(f"❌ Nenhum vídeo encontrado em: {VIDEOS_DIR}")
            return
        
        print(f"\n📹 Encontrados {len(video_files)} vídeos para processar\n")
        
        # Processar cada vídeo
        all_results = []

        # If using GPU, process sequentially to avoid contention on the same device.
        if device == 'cuda':
            for video_path in tqdm(video_files, desc="Processando vídeos", unit="vídeo"):
                try:
                    video_data = process_video(model, video_path, device=device)
                    all_results.append(video_data)
                    # Salvar resultados individuais
                    save_results(video_data, OUTPUT_DIR, video_path.stem)
                except Exception as e:
                    print(f"\n❌ Erro ao processar {video_path.name}: {str(e)}")
                    continue
        else:
            # CPU parallel processing: initialize one model per worker to avoid repeated loads
            global MODEL_PATH_GLOBAL
            MODEL_PATH_GLOBAL = MODEL_PATH
            n_workers = min(cpu_count(), len(video_files)) or 1
            print(f"Usando {n_workers} workers para processamento paralelo (CPU).")

            with Pool(processes=n_workers, initializer=init_worker, initargs=(MODEL_PATH, device)) as pool:
                # imap_unordered returns results as they complete; iterate with tqdm
                results_iter = pool.imap_unordered(process_video_worker, [str(p) for p in video_files])
                for res in tqdm(results_iter, total=len(video_files), desc="Processando vídeos", unit="vídeo"):
                    if res is None:
                        continue
                    if 'error' in res:
                        print(f"\n❌ Erro ao processar {res.get('video_path')}: {res['error']}")
                        continue
                    all_results.append(res)
                    # Salvar resultados individuais
                    save_results(res, OUTPUT_DIR, res['video_name'])
        
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