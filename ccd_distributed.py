"""Single-node DDP launching and the shared CCD optimization loop."""

from __future__ import annotations

import gc
import math
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


def configure_training_world(config: dict, devices: list[str], *, resume: bool) -> None:
    world_size = len(devices)
    if resume and config.get("world_size", 1) != world_size:
        raise ValueError(
            f"续训必须保持 world_size：checkpoint={config.get('world_size', 1)}，"
            f"当前={world_size}。更改卡数请重新训练。"
        )
    global_batch = config["batch_size"] * world_size
    config.update(devices=devices, device=devices[0], world_size=world_size,
                  global_batch_size=global_batch)
    if not resume and config.get("max_images_requested") is not None:
        config["max_steps"] = math.ceil(config["max_images_requested"] / global_batch)
    config["max_images"] = config["max_steps"] * global_batch
    if not resume and world_size > 1:
        config["hard_loss_mode"] = "per_image"


def distributed_loader(loader: DataLoader, *, rank: int, world_size: int, seed: int) -> DataLoader:
    """Shard without padding duplicates; each rank emits the same full batches."""
    required = loader.batch_size * world_size
    if len(loader.dataset) < required:
        raise ValueError(f"数据集图片数 {len(loader.dataset)} 小于全局 batch {required}。")
    sampler = DistributedSampler(loader.dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=seed, drop_last=True)
    return DataLoader(loader.dataset, batch_size=loader.batch_size, sampler=sampler,
                      drop_last=True, num_workers=loader.num_workers,
                      collate_fn=loader.collate_fn, pin_memory=loader.pin_memory,
                      persistent_workers=loader.persistent_workers)


def optimize(model, config: dict, manifest: dict, output: Path, previous: dict | None,
             save_every: int, *, rank: int = 0, world_size: int = 1) -> int:
    import efficientad_ccd as cli

    cli.load_runtime()
    distributed = world_size > 1
    primary = rank == 0
    loader = cli.make_loader(manifest["train"], config, shuffle=True, training=True)
    auxiliary_loader = model.imagenet_loader
    if distributed:
        del model.imagenet_iterator
        loader = distributed_loader(loader, rank=rank, world_size=world_size, seed=config["seed"])
        auxiliary_loader = distributed_loader(auxiliary_loader, rank=rank, world_size=world_size,
                                               seed=config["seed"] + 1)
    optimizer = torch.optim.Adam(
        list(model.model.student.parameters()) + list(model.model.ae.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, int(0.95 * config["max_steps"])), gamma=0.1,
    )
    step = previous["step"] if previous else 0
    if previous and "optimizer_state" in previous:
        optimizer.load_state_dict(previous["optimizer_state"])
        scheduler.load_state_dict(previous["scheduler_state"])
    model.train()
    network = model.model
    if distributed:
        # Statistics are checkpoint parameters, but never optimized. DDP must not
        # wait for gradients from these or from the frozen teacher.
        network.mean_std.requires_grad_(False)
        network.quantiles.requires_grad_(False)
        device = torch.device(config["device"])
        network = DistributedDataParallel(
            network, device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=True,
        )
    epoch = auxiliary_epoch = 0
    if distributed:
        # Each loader has its own epoch length and resume position.
        epoch = step // len(loader)
        auxiliary_epoch = step // len(auxiliary_loader)
        loader.sampler.set_epoch(epoch)
        auxiliary_loader.sampler.set_epoch(auxiliary_epoch)
    iterator = iter(loader)
    auxiliary_iterator = iter(auxiliary_loader) if distributed else model.imagenet_iterator
    if distributed:
        # Restoring only the epoch would replay its first batches after every
        # interruption. Advance both streams to the next unprocessed batch.
        for _ in range(step % len(loader)):
            next(iterator)
        for _ in range(step % len(auxiliary_loader)):
            next(auxiliary_iterator)
    progress = cli.tqdm(total=config["max_steps"], initial=step,
                        desc=f"训练 {manifest['category']}", disable=not primary)
    log_context = (output / "loss.csv").open("w", encoding="utf-8", newline="") if primary else nullcontext(None)
    with log_context as log:
        if primary:
            log.write("step,loss,loss_st,loss_ae,loss_stae,lr\n")
        try:
            while step < config["max_steps"]:
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    if distributed:
                        loader.sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)
                try:
                    auxiliary = next(auxiliary_iterator)[0]
                except StopIteration:
                    auxiliary_epoch += 1
                    if distributed:
                        auxiliary_loader.sampler.set_epoch(auxiliary_epoch)
                    auxiliary_iterator = iter(auxiliary_loader)
                    auxiliary = next(auxiliary_iterator)[0]
                optimizer.zero_grad(set_to_none=True)
                losses = network(batch=batch.image.to(config["device"]),
                                 batch_imagenet=auxiliary.to(config["device"]))
                loss = sum(losses)
                finite = torch.isfinite(loss).to(torch.int32)
                if distributed:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise RuntimeError(f"第 {step + 1} 步损失不是有限数值，所有进程停止训练。")
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                numbers = torch.stack([loss.detach(), *(x.detach() for x in losses)])
                if distributed:
                    dist.all_reduce(numbers)
                    numbers /= world_size
                if primary:
                    values = numbers.tolist()
                    log.write(f"{step}," + ",".join(str(x) for x in values) + f",{scheduler.get_last_lr()[0]}\n")
                    progress.update(1)
                    if step % 20 == 0 or step == 1:
                        progress.set_postfix(loss=f"{values[0]:.5f}")
                        log.flush()
                    if step % save_every == 0:
                        cli.save_checkpoint(output / "checkpoints/last.pt", model, config, manifest,
                                            step, optimizer, scheduler)
        except KeyboardInterrupt:
            if primary and not distributed:
                cli.save_checkpoint(output / "checkpoints/last.pt", model, config, manifest,
                                    step, optimizer, scheduler)
            raise
        finally:
            progress.close()
    if primary:
        cli.save_checkpoint(output / "checkpoints/last.pt", model, config, manifest,
                            step, optimizer, scheduler)
    return step


def _worker(rank: int, devices: list[str], bootstrap: str, rendezvous: str,
            output: str, save_every: int, backend: str) -> None:
    import efficientad_ccd as cli

    cli.load_runtime()
    device = devices[rank]
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    dist.init_process_group(backend, init_method=rendezvous, rank=rank,
                            world_size=len(devices), timeout=timedelta(minutes=10))
    try:
        initial = cli.read_checkpoint(Path(bootstrap))
        config = initial["config"].copy()
        config["device"] = device
        cli.seed_everything(config["seed"] + rank)
        model = cli.new_model(config)
        model.model.load_state_dict(initial["model_state"])
        cli.prepare_assets(model, config, load_teacher=False)
        optimize(model, config, initial["manifest"], Path(output), initial,
                 save_every, rank=rank, world_size=len(devices))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def train_distributed(model, config: dict, manifest: dict, output: Path,
                      previous: dict | None, save_every: int) -> int:
    """Prepare once, spawn workers, then restore rank-zero weights for calibration."""
    import efficientad_ccd as cli

    # API callers can omit the version while constructing a new default model.
    # Persist its actual architecture before workers interpret missing versions
    # as legacy checkpoints.
    config = dict(config)
    if hasattr(model.model, "resnet_architecture_version"):
        config["resnet_architecture_version"] = model.model.resnet_architecture_version
    devices = config["devices"]
    for name, count in (("训练集", len(manifest["train"])),
                        ("辅助集", len(model.imagenet_loader.dataset))):
        if count < config["global_batch_size"]:
            raise ValueError(f"{name}图片数 {count} 小于全局 batch {config['global_batch_size']}。")
    backend = "nccl" if devices[0].startswith("cuda") and dist.is_nccl_available() else "gloo"
    if not dist.is_available() or (backend == "gloo" and not dist.is_gloo_available()):
        raise RuntimeError(f"当前 PyTorch 不支持 {backend} 分布式后端。")
    print(f"DDP: devices={devices}, backend={backend}, 每卡 batch={config['batch_size']}, "
          f"全局 batch={config['global_batch_size']}")
    # Free the parent's GPU allocations while workers train.
    del model.imagenet_iterator
    model.cpu()
    gc.collect()
    if config["device"].startswith("cuda"):
        with torch.cuda.device(config["device"]):
            torch.cuda.empty_cache()
    with TemporaryDirectory(prefix="ddp_", dir=output) as temporary:
        root = Path(temporary)
        bootstrap = root / "initial.pt"
        payload = dict(previous) if previous else {"step": 0, "format_version": 1}
        payload.update(model_state=model.model.state_dict(), config=config, manifest=manifest)
        torch.save(payload, bootstrap)
        del payload
        workers = None
        try:
            workers = torch.multiprocessing.spawn(
                _worker, args=(devices, str(bootstrap), (root / "rendezvous").resolve().as_uri(),
                               str(output), save_every, backend), nprocs=len(devices), join=False,
            )
            while not workers.join(timeout=1):
                pass
        except KeyboardInterrupt:
            print(f"多卡训练已中断；可使用最后完整保存的 {output / 'checkpoints/last.pt'} 续训。")
            raise
        finally:
            # KeyboardInterrupt does not guarantee ProcessContext kills children
            # on Windows. Stop them before deleting rendezvous/bootstrap files.
            if workers is not None:
                for process in workers.processes:
                    if process.is_alive():
                        process.terminate()
                for process in workers.processes:
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join()
    trained = cli.read_checkpoint(output / "checkpoints/last.pt")
    model.model.load_state_dict(trained["model_state"])
    model.to(config["device"])
    return trained["step"]
