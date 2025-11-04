"""
Video Processor Module

This module handles YOLO inference on video files, extracting bounding box centers,
confidence scores, and class IDs for each detection.

Main classes:
  - VideoProcessor: Core class for processing individual videos
  - DetectionResult: Data class for storing detection results
  - ProcessingConfig: Configuration for inference parameters
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import cv2
import numpy as np
from ultralytics import YOLO
import json
from datetime import datetime
from src.config.logger import LoggerClass

@dataclass
class DetectionResult:
    """Stores detection results for a single frame or video"""

    frame_id: int
    center_x: float
    center_y: float
    confidence: float
    class_id: int
    bbox: Tuple[float, float, float, float] # (x1, y1, x2, y2)

    def to_dict(self) -> Dict:
        """Convert detection to dictionary format"""
        return {
          "frame_id": self.frame_id,
          "center_x": self.center_x,
          "center_y": self.center_y,
          "confidence": self.confidence,
          "class_id": self.class_id,
          "bbox": self.bbox
        }

@dataclass
class VideoResult:  
    """Stores detection results for a video"""

    video_path: str
    video_name: str
    frame_shape: Tuple[int, int, int] # (height, width, channels)
    total_frames: int
    total_detections: int
    detections: List[DetectionResult] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert video result to dictionary format"""
        return {
          "video_path": self.video_path,
          "video_name": self.video_name,
          "frame_shape": self.frame_shape,
          "total_frames": self.total_frames,
          "total_detections": self.total_detections,
          "detections": [det.to_dict() for det in self.detections],
          "processing_time": self.processing_time,
          "metadata": self.metadata
        }

    def get_centers_only(self) -> List[Tuple[float, float]]:
        """Extract only (x, y) centers from detections."""
        return [(det.center_x, det.center_y) for det in self.detections]
    
    def filter_by_class(self, class_id: int) -> 'VideoResult':
        """Return a new VideoResult with only detections of specified class"""
        filtered_detections = [
            det for det in self.detections if det.class_id == class_id
        ]
        result = VideoResult(
            video_path=self.video_path,
            video_name=self.video_name,
            frame_shape=self.frame_shape,
            total_frames=self.total_frames,
            total_detections= len(filtered_detections),
            detections=filtered_detections,
            processing_time=self.processing_time,
            metadata={**self.metadata, "filtered_by_class": class_id}
        )
        return result
    
    def filter_by_confidence(self, min_confidence: float) -> 'VideoResult':
        """Return a new VideoResult with only high-confidence detections"""

        filtered_detections = [
            det for det in self.detections if det.confidence >= min_confidence
        ]
        result = VideoResult(
            video_path=self.video_path,
            video_name=self.video_name,
            frame_shape=self.frame_shape,
            total_frames=self.total_frames,
            total_detections=len(filtered_detections),
            detections=filtered_detections,
            processing_time=self.processing_time,
            metadata={**self.metadata, 'min_confidence': min_confidence}
        )
        return result


@dataclass
class ProcessingConfig:
    """Configuration for video processing"""

    # Model parameters
    model_path: str
    device: str = 'cuda'
    conf_treshold: float = 0.25
    iou_treshold: float = 0.45

    # Processing parameters
    frame_skip: int = 0
    max_frames: Optional[int] = None
    target_class: Optional[List[int]] = None

    # Output parameters
    save_visualizations: bool = False
    visualization_output_dir: Optional[str] = None

    # Performance parameters
    batch_size: int = 1
    verbose: bool = True

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.conf_treshold < 0 or self.conf_treshold > 1:
            raise ValueError(f"conf_treshold must me between 0 and 1, got {self.conf_treshold}")
        
        if self.iou_treshold < 0 or self.iou_treshold > 1:
            raise ValueError(f"iou_treshold must be between 0 or 1, got {self.iou_treshold}")

        if self.frame_skip < 0:
            raise ValueError(f"frame_skip must be >= 0, got {self.frame_skip}")


class VideoProcessor:
    """
    Main class for processing videos with YOLO detection.

    Features:
      - Load and process video files
      - Extract bounding box centers and metadata
      - Support for frame skipping and class filtering
      - Optional visualization output
      - Multiple export formats (JSON, TXT)
    
    Example:
      >>> config = ProcessingConfig(model_path="model.pt", device="cuda")
      >>> processor = VideoProcessor(config)
      >>> result = processor.process_video("video.mp4")
      >>> processor.save_results(result, "output.json")
    """

    def __init__(self, config: ProcessingConfig) -> None:
        """
        Initialize videoProcessor with configuration

        Args:
            config (ProcessingConfig): object with model and processing settings
        """
        self.config = config
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self):
        """Load YOLO model from disk"""
        try:
            LoggerClass.debug(f"Loading YOLO model from {self.config.model_path}")
            self.model = YOLO(self.config.model_path)
            LoggerClass.debug(f"Model loaded successfully on device {self.config.device}")
        except Exception as e:
            LoggerClass.error(f"Failed to load model from {self.config.model_path}: {e}")

    def process_video(
            self,
            video_path: Union[str, Path],
            frame_callback: Optional[callable] = None
    ) -> VideoResult:
        """
        Process a single video file with YOLO detection

        Args:
            video_path (Union[str, Path]): Path to the video file
            frame_callback (Optional[callable], optional): Callback function called for each frame.
                Defaults to None.

        Returns:
            VideoResult: Object containing all detections
        
        Raises:
            FileNotFoundError: If the video file doesn't exists
            RuntimeError: If video processing fails
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        LoggerClass.debug(f"Processing video: {video_path.name}")

        start_time = datetime.now()

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        LoggerClass.debug(f"Video properties:")
        LoggerClass.debug(f"    Resolution: {frame_width}x{frame_height}")
        LoggerClass.debug(f"    Total Frames: {total_frames}")
        LoggerClass.debug(f"    FPS: {fps:.2f}")
        LoggerClass.debug(f"    Duration: {total_frames/fps:.2f}s")

        detections = []
        frame_id = 0
        frames_processed = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Frame skipping logic
                if self.config.frame_skip > 0 and frame_id % (self.config.frame_skip + 1) != 0:
                    frame_id += 1
                    continue

                # Max frames limit
                if self.config.max_frames and frames_processed >= self.config.max_frames:
                    break

                # Run YOLO inference
                results = self.model.predict(
                    frame,
                    conf=self.config.conf_treshold,
                    iou=self.config.iou_treshold,
                    verbose=False
                )

                # Extract detections
                frame_detections = self._extract_detections(results[0], frame_id)

                # Filter by target classes if specified
                if self.config.target_class is not None:
                    frame_detections = [
                        det for det in frame_detections
                        if det.class_id in self.config.target_class
                    ]
                
                detections.extend(frame_detections)

                if frame_callback:
                    frame_callback(frame_id, frame, frame_detections)
                
                frames_processed += 1
                frame_id += 1

                # Progress update
                if frames_processed % 100 == 0:
                    LoggerClass.debug(f"    Processed {frames_processed}/{total_frames} frames.."
                                      f"({len(detections)} detections so far)")
                    
        finally:
            cap.release()
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        result = VideoResult(
            video_path=str(video_path),
            video_name=video_path.name,
            frame_shape=(frame_height, frame_width, 3),
            total_frames=total_frames,
            total_detections=len(detections),
            detections=detections,
            processing_time=processing_time,
            metadata={
                "fps": fps,
                "frames_processed": frames_processed,
                "frame_skip": self.config.frame_skip,
                "conf_threshold": self.config.conf_treshold,
                "iou_threshold": self.config.iou_treshold,
                "target_classes": self.config.target_class,
                "processed_at": start_time.isoformat()
            }
        )

        LoggerClass.info(f"Processing complete: {frames_processed} frames, "
                         f"{len(detections)} detections in {processing_time:.2f}")
        return result
    
    def _extract_detections(
            self,
            result,
            frame_id: int
    ) -> List[DetectionResult]:
        """
        Extract detection results from YOLO output

        Args:
            result:  YOLO result object
            frame_id (int): Current frame ID

        Returns:
            List[DetectionResult]: List of detections for this frame
        """
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.bozes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            detection = DetectionResult(
                frame_id=frame_id,
                center_x=float(center_x),
                center_y=float(center_y),
                confidence=float(conf),
                class_id=int(cls_id),
                bbox=(float(x1), float(y1), float(x2), float(y2))
            )
            detections.append(detection)

        return detections

    def save_results(
            self,
            results: VideoResult,
            output_path: Union[str, Path],
            format: str = 'json'
    ) -> None:
        """
        Save detection results to file

        Args:
            results (VideoResult): Detection results to save
            output_path (Union[str, Path]): Output file path
            format (str, optional): Output format. Defaults to 'json'.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            self.save_json(results, output_path)
        elif format == 'txt':
            self.save_txt(results, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'txt'.")
        
    
    def save_json(
            self,
            results: VideoResult,
            output_path: Union[str, Path]
    ) -> None:
        """
        Save detection results to JSON file

        Args:
            results (VideoResult): Detection results to save
            output_path (Union[str, Path]): Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=4)
        LoggerClass.info(f"Results saved to JSON: {output_path}")
    
    def save_txt(
            self,
            results: VideoResult,
            output_path: Union[str, Path]
    ) -> None:
        """
        Save detection results to TXT file

        Args:
            results (VideoResult): Detection results to save
            output_path (Union[str, Path]): Output file path
        """
        with open(output_path, 'w') as f:
            for det in results.detections:
                line = (f"{det.frame_id} "
                        f"{det.class_id} "
                        f"{det.confidence:.4f} "
                        f"{det.bbox[0]:.2f} {det.bbox[1]:.2f} "
                        f"{det.bbox[2]:.2f} {det.bbox[3]:.2f}\n")
                f.write(line)
        LoggerClass.info(f"Results saved to TXT: {output_path}")

    
    def process_multiple_videos(
            self,
            video_paths: List[Union[str, Path]],
            output_dir: Optional[Union[str, Path]] = None,
            progress_callback: Optional[callable] = None
    ) -> List[VideoResult]:
        """
        Process multiple videos sequentially

        Args:
            video_paths (List[Union[str, Path]]): List of video file paths
            frame_callback (Optional[callable], optional): Callback function for each frame.
                Defaults to None.

        Returns:
            List[VideoResult]: List of detection results for each video
        """
        results = []
        total_videos = len(video_paths)

        LoggerClass.info(f"Processing {total_videos} videos...")

        for idx, video_path in enumerate(video_paths, 1):
            LoggerClass.info(f"[{idx}/{total_videos}] Processing: {Path(video_path).name}")

            try:
                result = self.process_video(video_path)
                results.append(result)

                # Save results if output directory is specified
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    video_name = Path(video_path).stem
                    json_path = output_dir / f"{video_name}.json"
                    self.save_results(result, json_path, format="json")

                if progress_callback:
                    progress_callback(idx, total_videos, result)

            except Exception as e:
                LoggerClass.error(f"Failed to process {video_path}: {e}")
                continue

        LoggerClass.info(f"✓ Batch processing complete: {len(results)}/{total_videos} successful")
        return results

    def generate_summary(self, results: List[VideoResult]) -> Dict:
        """
        Generate summary statistics from multiple video results

        Args:
            results (List[VideoResult]): List of video results

        Returns:
            Dict: Summary statistics
        """
        if not results:
            return {}

        total_detections = sum(r.total_detections for r in results)
        total_frames = sum(r.total_frames for r in results)
        total_time = sum(r.processing_time for r in results)

        # Class distribution
        class_counts = {}
        for result in results:
            for det in result.detections:
                class_counts[det.class_id] = class_counts.get(det.class_id, 0) + 1

        # Confidence statistics
        all_confidences = [det.confidence for result in results for det in result.detections]
        
        summary = {
            "total_videos": len(results),
            "total_frames": total_frames,
            "total_detections": total_detections,
            "total_processing_time": total_time,
            "avg_detections_per_video": total_detections / len(results) if results else 0,
            "avg_processing_time": total_time / len(results) if results else 0,
            "class_distribution": class_counts,
            "confidence_stats": {
                "mean": float(np.mean(all_confidences)) if all_confidences else 0,
                "min": float(np.min(all_confidences)) if all_confidences else 0,
                "max": float(np.max(all_confidences)) if all_confidences else 0,
                "std": float(np.std(all_confidences)) if all_confidences else 0
            }
        }

        return summary