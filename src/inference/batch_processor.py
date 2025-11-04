"""
Batch Processor Module

This module handles batch processing of multiple videos with support for
parallel processing, progress tracking, and error handling.

Main classes:
  - BatchProcessor: Process multiple videos in parallel or sequentially
  - BatchResult: Data class for storing batch processing results
  - BatchConfig: Configuration for batch processing
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from tqdm import tqdm

from src.inference.video_processor import VideoProcessor, VideoResult, ProcessingConfig
from src.config.logger import LoggerClass


@dataclass
class BatchResult:
    """Stores results from batch processing"""
    
    total_videos: int
    successful: int
    failed: int
    results: List[VideoResult] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)  # {video_path: error_message}
    processing_time: float = 0.0
    start_time: str = ""
    end_time: str = ""
    
    def to_dict(self) -> Dict:
        """Convert batch result to dictionary"""
        return {
            "total_videos": self.total_videos,
            "successful": self.successful,
            "failed": self.failed,
            "processing_time": self.processing_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "results": [r.to_dict() for r in self.results],
            "errors": self.errors,
            "summary": {
                "total_detections": sum(r.total_detections for r in self.results),
                "total_frames": sum(r.total_frames for r in self.results),
                "avg_detections_per_video": sum(r.total_detections for r in self.results) / len(self.results) if self.results else 0,
                "avg_processing_time": sum(r.processing_time for r in self.results) / len(self.results) if self.results else 0,
            }
        }
    
    def get_successful_results(self) -> List[VideoResult]:
        """Get only successful results"""
        return self.results
    
    def get_failed_videos(self) -> List[str]:
        """Get list of failed video paths"""
        return list(self.errors.keys())
    
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_videos == 0:
            return 0.0
        return (self.successful / self.total_videos) * 100


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    
    # Processing config
    processing_config: ProcessingConfig
    
    # Parallel processing
    use_parallel: bool = False
    max_workers: Optional[int] = None  # None = auto-detect
    use_processes: bool = False  # True = ProcessPoolExecutor, False = ThreadPoolExecutor
    
    # Output settings
    output_dir: Optional[Union[str, Path]] = None
    save_individual_results: bool = True
    save_batch_summary: bool = True
    
    # Progress tracking
    show_progress: bool = True
    verbose: bool = True
    
    # Error handling
    stop_on_error: bool = False
    retry_failed: int = 0  # Number of retries for failed videos
    
    def __post_init__(self):
        """Validate configuration"""
        if self.output_dir:
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)


class BatchProcessor:
    """
    Process multiple videos in batch with parallel processing support.
    
    Features:
      - Sequential or parallel processing
      - Progress tracking with tqdm
      - Error handling and retry logic
      - Automatic result saving
      - Batch statistics generation
    
    Example:
      >>> processing_config = ProcessingConfig(model_path="model.pt")
      >>> batch_config = BatchConfig(
      ...     processing_config=processing_config,
      ...     use_parallel=True,
      ...     max_workers=4,
      ...     output_dir="outputs/batch_results"
      ... )
      >>> processor = BatchProcessor(batch_config)
      >>> batch_result = processor.process_batch(video_paths)
    """
    
    def __init__(self, config: BatchConfig):
        """
        Initialize batch processor
        
        Args:
            config (BatchConfig): Batch processing configuration
        """
        self.config = config
        self.processor = VideoProcessor(config.processing_config)
        LoggerClass.info("BatchProcessor initialized")
        LoggerClass.debug(f"  Parallel: {config.use_parallel}")
        LoggerClass.debug(f"  Max workers: {config.max_workers}")
    
    def process_batch(
        self,
        video_paths: List[Union[str, Path]],
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """
        Process multiple videos in batch
        
        Args:
            video_paths (List[Union[str, Path]]): List of video file paths
            progress_callback (Optional[Callable], optional): Callback function 
                called after each video. Signature: callback(current, total, result)
        
        Returns:
            BatchResult: Batch processing results
        """
        video_paths = [Path(p) for p in video_paths]
        total_videos = len(video_paths)
        
        LoggerClass.info(f"Starting batch processing of {total_videos} videos")
        
        start_time = datetime.now()
        
        if self.config.use_parallel:
            batch_result = self._process_parallel(video_paths, progress_callback)
        else:
            batch_result = self._process_sequential(video_paths, progress_callback)
        
        end_time = datetime.now()
        
        # Update batch result timing
        batch_result.start_time = start_time.isoformat()
        batch_result.end_time = end_time.isoformat()
        batch_result.processing_time = (end_time - start_time).total_seconds()
        
        # Save batch summary
        if self.config.save_batch_summary and self.config.output_dir:
            self._save_batch_summary(batch_result)
        
        # Log summary
        self._log_summary(batch_result)
        
        return batch_result
    
    def _process_sequential(
        self,
        video_paths: List[Path],
        progress_callback: Optional[Callable]
    ) -> BatchResult:
        """
        Process videos sequentially
        
        Args:
            video_paths (List[Path]): List of video paths
            progress_callback (Optional[Callable]): Progress callback
        
        Returns:
            BatchResult: Processing results
        """
        results = []
        errors = {}
        
        iterator = tqdm(video_paths, desc="Processing videos") if self.config.show_progress else video_paths
        
        for idx, video_path in enumerate(iterator, 1):
            try:
                result = self._process_single_video(video_path)
                results.append(result)
                
                # Save individual result
                if self.config.save_individual_results and self.config.output_dir:
                    self._save_individual_result(result)
                
                # Progress callback
                if progress_callback:
                    progress_callback(idx, len(video_paths), result)
                
            except Exception as e:
                error_msg = str(e)
                errors[str(video_path)] = error_msg
                LoggerClass.error(f"Failed to process {video_path.name}: {error_msg}")
                
                if self.config.stop_on_error:
                    LoggerClass.error("Stopping batch processing due to error")
                    break
        
        return BatchResult(
            total_videos=len(video_paths),
            successful=len(results),
            failed=len(errors),
            results=results,
            errors=errors
        )
    
    def _process_parallel(
        self,
        video_paths: List[Path],
        progress_callback: Optional[Callable]
    ) -> BatchResult:
        """
        Process videos in parallel
        
        Args:
            video_paths (List[Path]): List of video paths
            progress_callback (Optional[Callable]): Progress callback
        
        Returns:
            BatchResult: Processing results
        """
        results = []
        errors = {}
        
        # Choose executor type
        if self.config.use_processes:
            ExecutorClass = ProcessPoolExecutor
            LoggerClass.debug("Using ProcessPoolExecutor")
        else:
            ExecutorClass = ThreadPoolExecutor
            LoggerClass.debug("Using ThreadPoolExecutor")
        
        max_workers = self.config.max_workers or min(4, len(video_paths))
        
        with ExecutorClass(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_video = {
                executor.submit(self._process_single_video, video_path): video_path
                for video_path in video_paths
            }
            
            # Process completed tasks with progress bar
            iterator = as_completed(future_to_video)
            if self.config.show_progress:
                iterator = tqdm(iterator, total=len(video_paths), desc="Processing videos")
            
            for idx, future in enumerate(iterator, 1):
                video_path = future_to_video[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Save individual result
                    if self.config.save_individual_results and self.config.output_dir:
                        self._save_individual_result(result)
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(idx, len(video_paths), result)
                    
                except Exception as e:
                    error_msg = str(e)
                    errors[str(video_path)] = error_msg
                    LoggerClass.error(f"Failed to process {video_path.name}: {error_msg}")
                    
                    if self.config.stop_on_error:
                        LoggerClass.error("Stopping batch processing due to error")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
        
        return BatchResult(
            total_videos=len(video_paths),
            successful=len(results),
            failed=len(errors),
            results=results,
            errors=errors
        )
    
    def _process_single_video(self, video_path: Path) -> VideoResult:
        """
        Process a single video with retry logic
        
        Args:
            video_path (Path): Path to video file
        
        Returns:
            VideoResult: Processing result
        
        Raises:
            Exception: If processing fails after all retries
        """
        attempts = self.config.retry_failed + 1
        last_error = None
        
        for attempt in range(1, attempts + 1):
            try:
                if self.config.verbose and attempt > 1:
                    LoggerClass.info(f"Retry {attempt-1}/{self.config.retry_failed} for {video_path.name}")
                
                result = self.processor.process_video(video_path)
                return result
                
            except Exception as e:
                last_error = e
                if attempt < attempts:
                    LoggerClass.warning(f"Attempt {attempt} failed: {e}")
                    continue
        
        # All attempts failed
        raise last_error
    
    def _save_individual_result(self, result: VideoResult) -> None:
        """
        Save individual video result
        
        Args:
            result (VideoResult): Video result to save
        """
        if not self.config.output_dir:
            return
        
        output_path = self.config.output_dir / f"{Path(result.video_name).stem}.json"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            LoggerClass.debug(f"Saved result: {output_path.name}")
        except Exception as e:
            LoggerClass.error(f"Failed to save result for {result.video_name}: {e}")
    
    def _save_batch_summary(self, batch_result: BatchResult) -> None:
        """
        Save batch processing summary
        
        Args:
            batch_result (BatchResult): Batch result to save
        """
        if not self.config.output_dir:
            return
        
        summary_path = self.config.output_dir / "batch_summary.json"
        
        try:
            with open(summary_path, 'w') as f:
                json.dump(batch_result.to_dict(), f, indent=2)
            LoggerClass.info(f"Batch summary saved: {summary_path}")
        except Exception as e:
            LoggerClass.error(f"Failed to save batch summary: {e}")
    
    def _log_summary(self, batch_result: BatchResult) -> None:
        """
        Log batch processing summary
        
        Args:
            batch_result (BatchResult): Batch result to log
        """
        LoggerClass.info("=" * 70)
        LoggerClass.info("BATCH PROCESSING SUMMARY")
        LoggerClass.info("=" * 70)
        LoggerClass.info(f"Total videos: {batch_result.total_videos}")
        LoggerClass.info(f"Successful: {batch_result.successful}")
        LoggerClass.info(f"Failed: {batch_result.failed}")
        LoggerClass.info(f"Success rate: {batch_result.success_rate():.1f}%")
        LoggerClass.info(f"Total processing time: {batch_result.processing_time:.2f}s")
        
        if batch_result.results:
            total_detections = sum(r.total_detections for r in batch_result.results)
            total_frames = sum(r.total_frames for r in batch_result.results)
            avg_time = sum(r.processing_time for r in batch_result.results) / len(batch_result.results)
            
            LoggerClass.info(f"Total detections: {total_detections}")
            LoggerClass.info(f"Total frames: {total_frames}")
            LoggerClass.info(f"Avg processing time: {avg_time:.2f}s/video")
        
        if batch_result.errors:
            LoggerClass.warning(f"\nFailed videos ({len(batch_result.errors)}):")
            for video_path, error in batch_result.errors.items():
                LoggerClass.warning(f"  - {Path(video_path).name}: {error}")
        
        LoggerClass.info("=" * 70)
    
    def process_directory(
        self,
        directory: Union[str, Path],
        extensions: List[str] = None,
        recursive: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """
        Process all videos in a directory
        
        Args:
            directory (Union[str, Path]): Directory containing videos
            extensions (List[str], optional): Video file extensions to process. 
                Defaults to ['.mp4', '.avi', '.mov', '.mkv'].
            recursive (bool, optional): Search recursively. Defaults to False.
            progress_callback (Optional[Callable], optional): Progress callback.
        
        Returns:
            BatchResult: Batch processing results
        """
        if extensions is None:
            extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
        
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find video files
        if recursive:
            video_paths = []
            for ext in extensions:
                video_paths.extend(directory.rglob(f"*{ext}"))
        else:
            video_paths = []
            for ext in extensions:
                video_paths.extend(directory.glob(f"*{ext}"))
        
        LoggerClass.info(f"Found {len(video_paths)} videos in {directory}")
        
        if not video_paths:
            LoggerClass.warning("No video files found")
            return BatchResult(
                total_videos=0,
                successful=0,
                failed=0,
                results=[],
                errors={}
            )
        
        return self.process_batch(video_paths, progress_callback)


# Convenience function
def process_videos_batch(
    video_paths: List[Union[str, Path]],
    model_path: str,
    output_dir: Union[str, Path],
    use_parallel: bool = False,
    device: str = 'cuda',
    **kwargs
) -> BatchResult:
    """
    Convenience function to process videos in batch
    
    Args:
        video_paths (List[Union[str, Path]]): List of video paths
        model_path (str): Path to YOLO model
        output_dir (Union[str, Path]): Output directory
        use_parallel (bool, optional): Use parallel processing. Defaults to False.
        device (str, optional): Device to use. Defaults to 'cuda'.
        **kwargs: Additional arguments for ProcessingConfig
    
    Returns:
        BatchResult: Batch processing results
    """
    processing_config = ProcessingConfig(
        model_path=model_path,
        device=device,
        **kwargs
    )
    
    batch_config = BatchConfig(
        processing_config=processing_config,
        use_parallel=use_parallel,
        output_dir=output_dir
    )
    
    processor = BatchProcessor(batch_config)
    return processor.process_batch(video_paths)