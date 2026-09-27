"""Guards on ``build_checkpoint_callbacks``: the latest epoch is always on disk.

These run a real ``Trainer`` over a 20-parameter stand-in rather than asserting on the
callbacks' attributes, because the behaviour at issue is not configuration — it is which
of Lightning's internal save paths fires.  ``ModelCheckpoint(save_last=True,
save_top_k=0)``, the obvious spelling of "always save the newest", writes NOTHING in
Lightning 2.x: ``_save_last_checkpoint`` runs only when the top-k path already saved at
this global step, and ``save_top_k=0`` returns before it can.  That failure is silent, and
on the cluster it would surface as an empty checkpoints/ directory after a 12 h slot.

The module logs nothing at all, so a checkpoint callback that still wanted a monitored
metric raises here rather than there.
"""
from pathlib import Path

import pytest
import torch
import lightning as L
from torch.utils.data import DataLoader, TensorDataset

from vic.checkpoints import resolve_resume
from vic.training.callbacks import build_checkpoint_callbacks

EVERY_N, N_EPOCHS = 3, 7


class _Tiny(L.LightningModule):
    """Manual optimisation and two networks, as the GAN modules have."""

    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        self.g = torch.nn.Linear(2, 2)
        self.d = torch.nn.Linear(2, 2)

    def training_step(self, batch, _):
        x, = batch
        for opt, net in zip(self.optimizers(), (self.g, self.d)):
            self.toggle_optimizer(opt)
            opt.zero_grad()
            self.manual_backward(net(x).pow(2).mean())
            opt.step()
            self.untoggle_optimizer(opt)

    def validation_step(self, batch, _):
        x, = batch
        return self.g(x).pow(2).mean()

    def configure_optimizers(self):
        return (
            torch.optim.Adam(self.g.parameters(), lr=1e-3),
            torch.optim.Adam(self.d.parameters(), lr=1e-3),
        )


def _loader():
    return DataLoader(TensorDataset(torch.randn(8, 2)), batch_size=4)


def _fit(ckpt_dir: Path, max_epochs: int, resume: str | None = None) -> L.Trainer:
    trainer = L.Trainer(
        default_root_dir=ckpt_dir.parent,
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=build_checkpoint_callbacks(ckpt_dir, every_n_epochs=EVERY_N),
    )
    trainer.fit(_Tiny(), _loader(), _loader(), ckpt_path=resume)
    return trainer


def _epoch_of(path: Path) -> int:
    """The 0-indexed epoch a checkpoint was written at the end of.

    Lightning stores ``trainer.current_epoch``, still N inside epoch N's
    ``on_train_epoch_end`` — so an n-epoch run leaves n-1, and resuming starts at n.
    """
    return torch.load(path, map_location="cpu", weights_only=False)["epoch"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> Path:
    ckpt_dir = tmp_path_factory.mktemp("run") / "checkpoints"
    _fit(ckpt_dir, N_EPOCHS)
    return ckpt_dir


def test_last_checkpoint_holds_the_final_epoch(trained):
    """The whole point: no metric selects this, so it is the newest weights, always."""
    assert (trained / "last.ckpt").exists(), "nothing saved -- the save_last trap is back"
    assert _epoch_of(trained / "last.ckpt") == N_EPOCHS - 1


def test_last_checkpoint_is_overwritten_not_versioned(trained):
    """Without ``enable_version_counter=False`` the second epoch writes last-v1.ckpt, and
    the run then leaves one file per epoch and a stale last.ckpt."""
    assert not list(trained.glob("last-v*.ckpt"))


def test_periodic_snapshots_are_kept_and_never_pruned(trained):
    """The insurance against an interrupt inside the in-place last.ckpt write."""
    snaps = sorted(p.name for p in trained.glob("converter-epoch=*.ckpt"))
    assert snaps == [f"converter-epoch={e:04d}.ckpt"
                     for e in range(EVERY_N - 1, N_EPOCHS, EVERY_N)]


def test_nothing_is_named_after_a_metric(trained):
    assert not list(trained.glob("*val*")), "a monitored checkpoint is back"


def test_last_checkpoint_advances_every_epoch(tmp_path):
    """Not merely every ``every_n_epochs`` — a periodic-only last.ckpt would lose up to
    99 epochs of a 12 h run on requeue."""
    ckpt_dir = tmp_path / "checkpoints"
    last = ckpt_dir / "last.ckpt"
    seen = []
    for n in range(1, EVERY_N + 2):
        _fit(ckpt_dir, n, resume=str(last) if last.exists() else None)
        seen.append(_epoch_of(last))
    assert seen == list(range(EVERY_N + 1))


def test_a_snapshot_can_be_resumed_from_when_last_is_unreadable(tmp_path):
    """The scenario the snapshots exist for: walltime or a crash truncates last.ckpt."""
    ckpt_dir = tmp_path / "checkpoints"
    _fit(ckpt_dir, N_EPOCHS)

    (ckpt_dir / "last.ckpt").write_bytes((ckpt_dir / "last.ckpt").read_bytes()[:1024])
    with pytest.raises(Exception):
        torch.load(ckpt_dir / "last.ckpt", map_location="cpu", weights_only=False)

    snap = sorted(ckpt_dir.glob("converter-epoch=*.ckpt"))[-1]
    trainer = _fit(ckpt_dir, _epoch_of(snap) + 2, resume=str(snap))
    assert trainer.current_epoch == _epoch_of(snap) + 2
    assert len(trainer.optimizers[0].state) > 0, "optimiser state did not come back"


# ---------------------------------------------------------------------------
# resolve_resume: which checkpoint a run continues from.
# ---------------------------------------------------------------------------

def test_fresh_run_resumes_from_nothing(tmp_path):
    assert resolve_resume(tmp_path / "checkpoints", None) is None


def test_seed_is_used_on_the_first_start(tmp_path):
    seed = tmp_path / "old" / "checkpoints" / "last.ckpt"
    seed.parent.mkdir(parents=True)
    seed.touch()
    assert resolve_resume(tmp_path / "new" / "checkpoints", seed) == seed


def test_a_directory_seed_resolves_to_its_last_ckpt(tmp_path):
    """`training.resume_from` naming a run's checkpoints/ dir is the obvious typo-free
    spelling, so accept it."""
    old = tmp_path / "old" / "checkpoints"
    old.mkdir(parents=True)
    (old / "last.ckpt").touch()
    assert resolve_resume(tmp_path / "new" / "checkpoints", old) == old / "last.ckpt"


def test_own_last_ckpt_beats_the_seed(tmp_path):
    """The precedence that matters.  Backwards, a requeued job would discard every
    epoch it had run since the seed and start over from the earlier run's weights --
    silently, and repeatedly, for as long as the job kept requeueing."""
    seed = tmp_path / "old" / "checkpoints" / "last.ckpt"
    seed.parent.mkdir(parents=True)
    seed.touch()

    own_dir = tmp_path / "new" / "checkpoints"
    own_dir.mkdir(parents=True)
    (own_dir / "last.ckpt").touch()

    assert resolve_resume(own_dir, seed) == own_dir / "last.ckpt"


def test_a_missing_seed_raises_rather_than_training_from_scratch(tmp_path):
    """A mistyped cluster path must not cost a 12 h slot and read as a fresh run."""
    with pytest.raises(SystemExit, match="does not exist"):
        resolve_resume(tmp_path / "checkpoints", tmp_path / "nope" / "last.ckpt")


def test_a_real_run_continues_from_a_seed_in_another_directory(tmp_path):
    """End to end: train, then seed a *different* output_dir from that run."""
    first = tmp_path / "run_a" / "checkpoints"
    _fit(first, N_EPOCHS)

    second = tmp_path / "run_b" / "checkpoints"
    resume = resolve_resume(second, first / "last.ckpt")
    trainer = _fit(second, N_EPOCHS + 3, resume=str(resume))

    assert trainer.current_epoch == N_EPOCHS + 3
    assert len(trainer.optimizers[0].state) > 0
    assert _epoch_of(second / "last.ckpt") == N_EPOCHS + 2
