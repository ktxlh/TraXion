import torch
import wandb

class WandbLogger:
    def __init__(self, job_type, args, project="traxion"):
        self.args = args
        self.use_wandb = args.use_wandb
        self.epoch = 0.0
        self.iteration = 0
        self.device = args.device
        self.gpu_index = None
        self.run_name = f"{args.dataset}-clique{args.clique_size}-dim{args.model_dim}-heads{args.n_heads}-layers{args.num_layers}"

        if self.device and isinstance(self.device, str) and self.device.startswith("cuda"):
            try:
                if ":" in self.device:
                    self.gpu_index = int(self.device.split(":", 1)[1])
                else:
                    self.gpu_index = 0
            except (ValueError, IndexError):
                self.gpu_index = None

        self._prev_best_artifact = None

        if self.use_wandb:
            import os
            wandb_run_id = getattr(args, "wandb_run_id", None)
            entity = getattr(args, "wandb_entity", None) or os.environ.get("WANDB_ENTITY") or None
            self.run = wandb.init(
                entity=entity,
                project=getattr(args, "wandb_project", None) or project,
                dir=args.training_log_dir,
                config=args,
                job_type=job_type,
                name=self.run_name,
                id=wandb_run_id,
                resume="must" if wandb_run_id else None,
                settings=wandb.Settings(disable_git=False, console="wrap"),
            )
        
    def log(self, metrics: dict, split=None):
        if self.use_wandb:
            if split is not None:
                # Replace metric_name with split/metric_name for better organization in Wandb
                metrics = {f"{split}/{k}": v for k, v in metrics.items()}
                
            # Add GPU memory metrics for the specific device
            if self.gpu_index is not None and torch.cuda.is_available():
                try:
                    allocated = torch.cuda.memory_allocated(self.gpu_index) / 1024**3  # Convert to GB
                    reserved = torch.cuda.memory_reserved(self.gpu_index) / 1024**3
                    metrics[f"gpu/memory_allocated_gb"] = allocated
                    metrics[f"gpu/memory_reserved_gb"] = reserved
                except RuntimeError:
                    pass  # Device not available
            metrics['epoch'] = self.epoch
            metrics['iteration'] = self.iteration
            self.run.log(metrics)

    def set_epoch_progress(self, epoch_index: int, batch_index: int, total_batches: int):
        if total_batches <= 0:
            self.epoch = float(epoch_index)
            return

        self.epoch = float(epoch_index) + (float(batch_index) / float(total_batches))
        self._calibrate_epoch()

    def complete_epoch(self):
        self.epoch = float(round(self.epoch))

    def _calibrate_epoch(self, tolerance: float = 1e-9):
        nearest_integer = round(self.epoch)
        if abs(self.epoch - nearest_integer) < tolerance:
            self.epoch = float(nearest_integer)
        
    def log_artifact(self, epoch, model_path):
        if self.use_wandb:
            # Use the run ID in the name so each run has its own artifact
            # lineage, avoiding "base artifact is deleted" errors from
            # cross-run version chains.
            artifact_name = f"model.state_dict-{self.run.id}"
            checkpoint_artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"trained for {epoch} epochs"
            )
            checkpoint_artifact.add_file(model_path)
            try:
                new_artifact = self.run.log_artifact(checkpoint_artifact)
                new_artifact.wait()  # block until upload is committed
            except Exception as e:
                print(f"Warning: failed to log artifact to W&B ({e}); checkpoint is still saved locally.")
                return

            # Delete the previous best version so only one version is kept
            if self._prev_best_artifact is not None:
                try:
                    self._prev_best_artifact.delete(delete_aliases=True)
                except Exception:
                    pass
            self._prev_best_artifact = new_artifact

    def finish(self):
        if self.use_wandb:
            self.run.finish()