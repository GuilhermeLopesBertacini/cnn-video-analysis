# Video Analysis - Detecção de Objetos e Análise de Heatmap

Projeto para processamento de vídeos com detecção de objetos usando YOLO e geração de mapas de calor.

## 📋 Requisitos

- Python 3.8+
- CUDA (opcional, mas recomendado para GPU)
- GPU compatível com CUDA (para melhor performance)

## 🚀 Instalação

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 📹 Processamento de Vídeos

### Executar inferência em todos os vídeos:

```bash
python videos_inference.py
```

Ou use o script auxiliar:
```bash
chmod +x run_inference.sh
./run_inference.sh
```

### O que o script faz:

- ✅ Detecta automaticamente se há GPU CUDA disponível
- ✅ Processa todos os vídeos em `assets/occurrence_entering/videos_camera_0/videos/`
- ✅ Extrai centros de massa das bounding boxes
- ✅ Salva resultados em múltiplos formatos (JSON e TXT)
- ✅ Gera sumário consolidado de todas as detecções
- ✅ Mostra barra de progresso durante processamento

### Estrutura de saída:

```
results/bb_centers/
├── summary.json                    # Sumário geral de todos os vídeos
├── all_centers_consolidated.txt    # Todos os centros em um único arquivo
├── video_001.json                  # Dados completos do vídeo (com confiança, classe, etc)
├── video_001.txt                   # Apenas coordenadas dos centros
├── video_002.json
├── video_002.txt
└── ...
```

## 📊 Análise e Visualização (Notebook)

Abra o notebook `analysis.ipynb` para:

- 🔥 Gerar heatmaps de densidade de detecções
- 📈 Visualizar distribuição espacial dos objetos
- 📊 Criar gráficos de dispersão
- 🎨 Customizar visualizações

Para usar o notebook:
```bash
jupyter notebook analysis.ipynb
```

## 🎯 Funcionalidades

### videos_inference.py

- Processamento em lote de múltiplos vídeos
- Detecção automática de GPU/CUDA
- Salvamento estruturado dos resultados
- Informações de confiança e classe dos objetos
- Barra de progresso com `tqdm`
- Tratamento de erros robusto

### analysis.ipynb

- Carregamento e parsing de resultados
- Geração de heatmaps suavizados
- Gráficos de dispersão com densidade
- Análise estatística das detecções

## 📁 Estrutura do Projeto

```
video_analysis/
├── assets/
│   ├── models/
│   │   └── yolo8_nano_640_43750_otimizado.pt
│   └── occurrence_entering/
│       └── videos_camera_0/
│           └── videos/
├── results/
│   └── bb_centers/
├── videos_inference.py    # Script principal de inferência
├── analysis.ipynb         # Notebook de análise
├── requirements.txt       # Dependências
└── README.md             # Este arquivo
```

## 🔧 Configurações Avançadas

Para personalizar o processamento, edite as variáveis em `videos_inference.py`:

```python
MODEL_PATH = "assets/models/yolo8_nano_640_43750_otimizado.pt"
VIDEOS_DIR = Path("assets/occurrence_entering/videos_camera_0/videos")
OUTPUT_DIR = Path("results/bb_centers")
```

## 💡 Dicas de Performance

- Use GPU CUDA para processamento ~10-50x mais rápido
- Ajuste o batch size do modelo se tiver muita VRAM
- Para vídeos muito grandes, considere processar em chunks
- Desative `save=True` no predict para economizar espaço

## 📝 Formatos de Saída

### JSON (completo):
```json
{
  "video_name": "video_001",
  "total_detections": 150,
  "frame_shape": [1080, 1920],
  "centers": [
    {
      "center_x": 284.43,
      "center_y": 241.93,
      "confidence": 0.95,
      "class_id": 0
    }
  ]
}
```

### TXT (simples):
```
(284.43), (241.93)
(585.98), (123.45)
...
```

## 🐛 Troubleshooting

**Erro de CUDA:**
```bash
# Verifique se CUDA está instalado
nvidia-smi

# Verifique se PyTorch detecta CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**Erro de memória GPU:**
- Reduza o tamanho do batch
- Use um modelo menor (nano é o menor)
- Processe vídeos em lotes menores

