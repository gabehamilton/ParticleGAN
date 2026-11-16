"""Energy-distance transport against detached seen targets (a declared training objective).

This is a generator/prior training objective, not an evaluator, sampler or
post-optimizer repair. It reads only targets the caller has already consumed
for its generator phase: no labels, mode centers, extra data draws or
altered sampling weights enter.

Objective, over flattened samples ``x`` (generated, ``n`` rows) and detached
``y`` (targets, ``m`` rows)::

    E(x, y) = (2 * mean_ij rho(x_i - y_j) - mean_ii' rho(x_i - x_i')) / s

* ``rho(v) = sqrt(||v||^2 + delta^2)`` is a smoothed Euclidean norm with
  ``delta = SMOOTHING * s``. Its gradient is ``v / rho(v)``: continuous, and
  exactly zero for coincident points (including every ``i == i'`` pair).
  Distances use ``torch.cdist(..., compute_mode="donot_use_mm_for_euclid_dist")``
  (exact differences, no Gram-matrix expansion).
* ``mean_ii'`` runs over all ``n * n`` ordered pairs, including the diagonal
  (the V-statistic; each diagonal term is the constant ``delta``).
* ``s = sqrt(mean_j ||y_j - mean(y)||^2)`` is the square root of the target
  set's total (trace) coordinate variance, detached. Because ``delta`` is
  proportional to ``s``, ``E(c x, c y) = E(x, y)`` for every ``c > 0``.
  A target set with fewer than two rows, or with ``s <= eps``
  (``eps = torch.finfo(dtype).eps``, e.g. identical rows), has no usable
  scale and raises ``ValueError``; the objective is never divided by a
  clamped near-zero scale.
* The target-target term ``mean_jj' rho(y_j - y_j')`` of the energy distance
  is constant with respect to every trained parameter and is neither computed
  nor part of the objective. ``E`` is therefore the energy distance minus that
  constant, divided by ``s``.

Gradients reach generator and prior parameters only through ``x``. All
derivatives are PyTorch autograd of the expression above; there is no custom
backward.
"""
from dataclasses import dataclass

import torch

SMOOTHING = 1e-6
DISTANCE_MODE = "donot_use_mm_for_euclid_dist"


@dataclass(frozen=True)
class EnergyTransportTerms:
    """Unweighted objective and its detached-scale components."""
    loss: torch.Tensor
    cross: torch.Tensor
    fake_self: torch.Tensor
    scale: torch.Tensor


def _validate(fake, target):
    if (not isinstance(fake, torch.Tensor) or not isinstance(target, torch.Tensor)
            or fake.ndim < 2 or target.ndim < 2 or fake.shape[1:] != target.shape[1:]
            or len(fake) < 2 or len(target) < 2
            or not fake.is_floating_point() or fake.dtype != target.dtype
            or fake.device != target.device):
        raise ValueError("energy transport needs at least two generated samples and at least two "
                         "target rows with the same sample shape, floating dtype and device")


def energy_transport_terms(fake, target):
    """Return :class:`EnergyTransportTerms` for generated ``fake`` against detached ``target``."""
    _validate(fake, target)
    x, y = fake.flatten(1), target.detach().flatten(1)
    scale = (y - y.mean(0)).square().sum(1).mean().sqrt()
    if not bool(scale > torch.finfo(y.dtype).eps):  # also rejects NaN
        raise ValueError("energy transport target set has no spread (trace standard deviation "
                         f"{float(scale):.3g} <= eps); the scale-normalized objective is undefined")
    delta_squared = (SMOOTHING * scale).square()
    cross = (torch.cdist(x, y, compute_mode=DISTANCE_MODE).square() + delta_squared).sqrt().mean()
    fake_self = (torch.cdist(x, x, compute_mode=DISTANCE_MODE).square() + delta_squared).sqrt().mean()
    loss = (2 * cross - fake_self) / scale
    return EnergyTransportTerms(loss=loss, cross=cross / scale, fake_self=fake_self / scale, scale=scale)


def energy_transport_loss(fake, target):
    """Unweighted scale-normalized energy objective (see the module documentation)."""
    return energy_transport_terms(fake, target).loss


class SeenTargetMemory:
    """FIFO of the most recent ``capacity`` detached target rows (a ring buffer).

    ``push`` appends a batch, oldest rows leave first; a batch at least as large
    as the capacity keeps only its last ``capacity`` rows. ``targets()`` returns
    the stored rows in storage order (a set for the permutation-invariant
    objective, no copy); ``contents()`` returns them oldest first. The buffer is
    allocated on the first push with that batch's sample shape, dtype and
    device; later batches must match. ``state_dict`` stores the complete buffer,
    fill count and write position for exact continuation.
    """

    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("memory capacity must be a positive integer")
        self.capacity = capacity
        self.buffer = None
        self.count = 0
        self.next = 0

    def __len__(self):
        return self.count

    @torch.no_grad()
    def push(self, batch):
        if not isinstance(batch, torch.Tensor) or batch.ndim < 2 or not len(batch) or not batch.is_floating_point():
            raise ValueError("memory batches must be nonempty floating tensors with a sample dimension")
        rows = batch.detach()
        if self.buffer is None:
            self.buffer = rows.new_zeros((self.capacity, *rows.shape[1:]))
        elif (rows.shape[1:] != self.buffer.shape[1:] or rows.dtype != self.buffer.dtype
              or rows.device != self.buffer.device):
            raise ValueError("memory batch sample shape, dtype or device changed")
        if len(rows) >= self.capacity:
            self.buffer.copy_(rows[len(rows) - self.capacity:])
            self.count, self.next = self.capacity, 0
            return
        first = min(len(rows), self.capacity - self.next)
        self.buffer[self.next:self.next + first].copy_(rows[:first])
        if first < len(rows):
            self.buffer[:len(rows) - first].copy_(rows[first:])
        self.next = (self.next + len(rows)) % self.capacity
        self.count = min(self.count + len(rows), self.capacity)

    def targets(self):
        """Stored rows in storage order; empty memory raises."""
        if not self.count:
            raise ValueError("seen-target memory is empty")
        return self.buffer[:self.count]

    def contents(self):
        """Independent copy of the stored rows, oldest first (``None`` before the first push)."""
        if self.buffer is None:
            return None
        if self.count < self.capacity:
            return self.buffer[:self.count].clone()
        return torch.cat((self.buffer[self.next:], self.buffer[:self.next]))

    def state_dict(self):
        return {"capacity": self.capacity, "count": self.count, "next": self.next,
                "rows": None if self.buffer is None else self.buffer.detach().clone()}

    def check_state(self, state, *, dtype=None):
        """Validate a saved memory without changing this one."""
        if not isinstance(state, dict) or set(state) != {"capacity", "count", "next", "rows"}:
            raise ValueError("invalid seen-target memory state")
        capacity, count, position, rows = state["capacity"], state["count"], state["next"], state["rows"]
        if capacity != self.capacity or type(capacity) is not int:
            raise ValueError("seen-target memory capacity differs from the recipe")
        if (type(count) is not int or type(position) is not int or not 0 <= count <= capacity
                or not 0 <= position < capacity or (count < capacity and position != count % capacity)):
            raise ValueError("invalid seen-target memory fill count or write position")
        if rows is None:
            if count:
                raise ValueError("seen-target memory rows are missing")
            return
        if (not isinstance(rows, torch.Tensor) or rows.ndim < 2 or len(rows) != capacity
                or not rows.is_floating_point() or (dtype is not None and rows.dtype != dtype)
                or not count or not torch.isfinite(rows[:count]).all()):
            raise ValueError("invalid seen-target memory rows")
        if self.buffer is not None and rows.shape[1:] != self.buffer.shape[1:]:
            raise ValueError("seen-target memory sample shape differs")

    def load_state_dict(self, state, *, device=None, dtype=None):
        self.check_state(state, dtype=dtype)
        rows = state["rows"]
        self.buffer = None if rows is None else rows.detach().to(device=device or rows.device).clone()
        self.count, self.next = state["count"], state["next"]
