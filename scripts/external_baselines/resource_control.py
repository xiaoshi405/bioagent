"""Resource-only amendment, bounded execution and immutable freeze lineage."""
from __future__ import annotations

import asyncio
import copy
import ctypes
import os
from pathlib import Path
from ctypes import wintypes

from .common import file_digest
from .experiment import ROOT, read, confined

RESOURCE_VERSION = 'external-resource-repair-v1.1'
POLICY = {
    'version': RESOURCE_VERSION,
    'sample_interval_seconds': 1.0,
    'max_root_private_bytes': 8 * 1024**3,
    'max_process_tree_private_bytes': 16 * 1024**3,
    'minimum_system_commit_headroom_bytes': 2 * 1024**3,
    'maximum_system_commit_fraction': 0.85,
    'action': 'pause batch, cancel active units, preserve all started attempts',
}
CHANGED_SOURCES = {
    'scripts/external_baselines/scitoolagent.py',
    'scripts/run_external_experiment.py',
    'scripts/score_external_experiment.py',
}
EXPECTED_OLD_ADAPTER = 'scitoolagent-official-protein-v1'
EXPECTED_NEW_ADAPTER = 'scitoolagent-official-protein-v1.1-resource'


# ctypes caches pointer types globally. Defining this type per sample would
# retain a new type on every watchdog tick; create it once for the process.
class _PerformanceInfo(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('CommitTotal', ctypes.c_size_t),
                ('CommitLimit', ctypes.c_size_t), ('CommitPeak', ctypes.c_size_t),
                ('PhysicalTotal', ctypes.c_size_t), ('PhysicalAvailable', ctypes.c_size_t),
                ('SystemCache', ctypes.c_size_t), ('KernelTotal', ctypes.c_size_t),
                ('KernelPaged', ctypes.c_size_t), ('KernelNonpaged', ctypes.c_size_t),
                ('PageSize', ctypes.c_size_t), ('HandleCount', wintypes.DWORD),
                ('ProcessCount', wintypes.DWORD), ('ThreadCount', wintypes.DWORD)]


def normalized_map(mapping):
    return {k.replace('\\', '/'): v for k, v in mapping.items()}


def validate_resource_amendment(base, amended):
    """Allow a named lifecycle repair, never planner/data/budget changes."""
    expected = copy.deepcopy(base)
    actual = copy.deepcopy(amended)
    for key in ('created_at', 'status', 'code_sha256', 'packages', 'methods', 'resource_amendment'):
        expected.pop(key, None)
        actual.pop(key, None)
    if actual != expected:
        raise ValueError('Resource amendment changes experimental design/settings')
    if amended.get('status') != 'FROZEN_RESOURCE_AMENDMENT_AFTER_INFRASTRUCTURE_INCIDENT':
        raise ValueError('Resource amendment must identify its post-incident timing')
    old_methods = copy.deepcopy(base['methods'])
    new_methods = copy.deepcopy(amended['methods'])
    if old_methods['SciToolAgent']['adapter'] != EXPECTED_OLD_ADAPTER:
        raise ValueError('Unexpected original resource adapter')
    if new_methods['SciToolAgent']['adapter'] != EXPECTED_NEW_ADAPTER:
        raise ValueError('Unexpected repaired resource adapter')
    new_methods['SciToolAgent']['adapter'] = old_methods['SciToolAgent']['adapter']
    if new_methods != old_methods:
        raise ValueError('Method/upstream identity changed beyond resource adapter')
    old_code, new_code = normalized_map(base['code_sha256']), normalized_map(amended['code_sha256'])
    if not set(old_code).issubset(new_code):
        raise ValueError('Resource amendment drops frozen sources')
    if any(new_code[k] != v for k, v in old_code.items() if k not in CHANGED_SOURCES):
        raise ValueError('Unrelated method/tool/scorer source changed')
    allowed_additions = {'scripts/external_baselines/resource_control.py',
                         'scripts/prepare_external_resource_repair.py'}
    if set(new_code) - set(old_code) != allowed_additions:
        raise ValueError('Unexpected resource sources')
    old_packages, new_packages = base['packages'], dict(amended['packages'])
    if 'psutil' not in new_packages:
        raise ValueError('Resource monitor dependency missing')
    new_packages.pop('psutil')
    if new_packages != old_packages:
        raise ValueError('Existing dependency versions changed')
    amendment = amended['resource_amendment']
    if amendment['version'] != RESOURCE_VERSION or amendment['memory_policy'] != POLICY:
        raise ValueError('Unexpected resource policy')
    if amendment['automatic_retry'] is not False:
        raise ValueError('Resource repair cannot authorize retries')


def verify_resource_lineage(frozen, root=ROOT):
    amendment = frozen.get('resource_amendment')
    if not amendment:
        return
    base_path = confined(root, amendment['base_freeze_path'])
    if file_digest(base_path) != amendment['base_freeze_sha256']:
        raise ValueError('Original freeze was changed')
    base = read(base_path)
    validate_resource_amendment(base, frozen)
    snapshot = confined(root, amendment['base_source_snapshot'])
    for relative, sha in base['code_sha256'].items():
        if file_digest(confined(snapshot, relative)) != sha:
            raise ValueError('Original source snapshot changed')
    for relative, sha in amendment['acceptance_files_sha256'].items():
        if file_digest(confined(root, relative)) != sha:
            raise ValueError('Resource acceptance evidence changed')


def verify_batch_freeze(frozen, freeze_path, start, root=ROOT):
    if start['freeze_sha256'] == file_digest(Path(freeze_path)):
        return
    verify_resource_lineage(frozen, root)
    amendment = frozen.get('resource_amendment')
    if not amendment or start['freeze_sha256'] != amendment['base_freeze_sha256']:
        raise ValueError('Batch has a different freeze')
    if start['freeze_path'].replace('\\', '/') != amendment['base_freeze_path'].replace('\\', '/'):
        raise ValueError('Unexpected original batch freeze path')


class ResourcePressure(BaseException):
    """Infrastructure stop; never converted into an agent failure outcome."""
    def __init__(self, reasons, sample):
        super().__init__('; '.join(reasons))
        self.reasons = reasons
        self.sample = sample


def memory_sample():
    import psutil
    root = psutil.Process()
    own = root.memory_info()
    private = getattr(own, 'private', own.vms)
    tree_private = private
    for child in root.children(recursive=True):
        try:
            m = child.memory_info()
            tree_private += getattr(m, 'private', m.vms)
        except psutil.NoSuchProcess:
            pass
    if os.name == 'nt':
        info = _PerformanceInfo()
        info.cb = ctypes.sizeof(info)
        api = ctypes.WinDLL('psapi', use_last_error=True)
        api.GetPerformanceInfo.argtypes = [ctypes.POINTER(_PerformanceInfo), wintypes.DWORD]
        api.GetPerformanceInfo.restype = wintypes.BOOL
        if not api.GetPerformanceInfo(ctypes.byref(info), info.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        used, limit = info.CommitTotal * info.PageSize, info.CommitLimit * info.PageSize
    else:
        vm, swap = psutil.virtual_memory(), psutil.swap_memory()
        limit = vm.total + swap.total
        used = limit - vm.available - swap.free
    return {'root_pid': root.pid, 'root_private_bytes': private,
            'process_tree_private_bytes': tree_private,
            'system_commit_bytes': used, 'system_commit_limit_bytes': limit,
            'system_commit_headroom_bytes': limit - used}


def check_pressure(sample, policy=POLICY):
    reasons = []
    if sample['root_private_bytes'] >= policy['max_root_private_bytes']:
        reasons.append('root private memory threshold')
    if sample['process_tree_private_bytes'] >= policy['max_process_tree_private_bytes']:
        reasons.append('process tree private memory threshold')
    if sample['system_commit_headroom_bytes'] <= policy['minimum_system_commit_headroom_bytes']:
        reasons.append('system commit headroom threshold')
    if sample['system_commit_bytes'] / sample['system_commit_limit_bytes'] >= policy['maximum_system_commit_fraction']:
        reasons.append('system commit fraction threshold')
    if reasons:
        raise ResourcePressure(reasons, sample)


async def run_guarded(work_factory, policy, record_sample, sampler=memory_sample):
    # Check before dispatch can claim even one new task.
    try:
        initial = sampler()
        record_sample(initial)
        check_pressure(initial, policy)
    except ResourcePressure:
        raise
    except Exception as exc:
        raise ResourcePressure(['resource monitor unavailable: ' + type(exc).__name__], {}) from exc

    async def monitor():
        while True:
            await asyncio.sleep(policy['sample_interval_seconds'])
            try:
                sample = sampler()
                record_sample(sample)
                check_pressure(sample, policy)
            except ResourcePressure:
                raise
            except Exception as exc:
                raise ResourcePressure(['resource monitor unavailable: ' + type(exc).__name__], {}) from exc

    work = asyncio.create_task(work_factory())
    watcher = asyncio.create_task(monitor())
    try:
        done, _ = await asyncio.wait((work, watcher), return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            # Consume the exception directly: a BaseException in a Task must not
            # be raised outside this awaited boundary or silently ignored.
            await watcher
        return await work
    finally:
        for task in (work, watcher):
            if not task.done():
                task.cancel()
        await asyncio.gather(work, watcher, return_exceptions=True)
