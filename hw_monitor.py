#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yijun Zhou
"""hw_monitor.py - live terminal dashboard of a Linux machine's hardware.

Panels (refreshed every --interval seconds, default 1 s):
  Alerts     only shown when needed: serious kernel messages of this boot (NVIDIA
             Xid errors, kernel BUG/Oops, machine checks, hung tasks, OOM kills,
             disk I/O errors) and kernel taint flags
  CPU        total + per-core usage / frequency / temperature, load, package
             power*, thermal limit (TjMax - TCC offset), throttling
  GPU        NVIDIA via NVML (util, VRAM, temp, power, clocks, PCIe, throttle
             reasons, processes, "GPU lost / reset required" errors); Intel iGPU
             frequency and busy % (from RC6)
  Memory     RAM, swap, dirty / writeback page cache
  Disks      read/write throughput, IOPS, busy %, temperature*, filesystem usage
  Network    per-interface rx/tx rate and totals, link speed, Wi-Fi SSID/signal
  Sensors    every hwmon / thermal-zone temperature, fan speeds
  Power      AC, battery charge / draw / health, CPU+GPU power, peripherals
  Processes  top CPU consumers, process-state counts

  * CPU package power (RAPL) and SMART temperatures of SATA / USB hard drives
    need root (plus smartmontools for the drives). Everything else works as a
    normal user; the kernel log needs membership of the 'adm' group.

Slow or hang-prone sources (NVML, smartctl, statfs, the kernel log) are read on
background threads, so a GPU that stops answering cannot freeze the dashboard
or the CSV log. The log is fsync'ed every few seconds, so after a hard crash it
holds the readings up to the last moments.

Usage:
  ./hw_monitor.py                      live dashboard, 1 s refresh (Ctrl+C quits)
  ./hw_monitor.py -i 2                 refresh every 2 s
  ./hw_monitor.py --log hw.csv         dashboard + append one CSV row per tick
  nohup ./hw_monitor.py --no-ui --log hw.csv -i 2 &
                                       headless logging that survives closing
                                       the terminal (or run it inside tmux)
  ./hw_monitor.py --no-ui              one compact text line per tick
  ./hw_monitor.py --once               print a single snapshot and exit
  ./hw_monitor.py --sections cpu,gpu   only some panels (small terminals)
  ./hw_monitor.py --ping               also time a TCP handshake to 1.1.1.1:443
  sudo "$(command -v python3)" hw_monitor.py
                                       adds CPU power and HDD temperatures (naming
                                       the interpreter keeps your venv/conda env,
                                       which plain sudo would drop)

Dependencies: Python 3.9+, psutil, rich (pip install psutil rich). NVIDIA data
is read straight from libnvidia-ml via ctypes, so no nvidia-smi subprocess is
spawned per tick.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import glob
import grp
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque

try:
    import psutil
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError as exc:
    sys.exit(f"hw_monitor: missing Python package '{exc.name}'. Install the dependencies with\n"
             f"  pip install psutil rich\n"
             f"or run the script with a Python interpreter that already has them.")

__version__ = '1.0.0'

HISTORY = 240           # samples kept for sparklines
SECTIONS = ('cpu', 'gpu', 'mem', 'disk', 'net', 'sensors', 'power', 'procs')


# ─────────────────────────────── helpers ────────────────────────────────

def read(path):
    """Stripped contents of a sysfs/procfs file, or None if it can't be read."""
    try:
        with open(path) as f:
            return f.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


def read_int(path):
    try:
        return int(read(path))
    except (TypeError, ValueError):
        return None


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def parse_cpulist(s):
    """'0-7,9' -> {0, 1, ..., 7, 9}"""
    cpus = set()
    for part in (x.strip() for x in (s or '').split(',')):
        if '-' in part:
            a, b = part.split('-', 1)
            cpus.update(range(int(a), int(b) + 1))
        elif part:
            cpus.add(int(part))
    return cpus


def fmt_bytes(n, rate=False):
    """Binary-scaled compact size: 1536 -> '1.5K', rate=True appends '/s'."""
    if n is None:
        return '–'
    suffix = '/s' if rate else ''
    unit = 'B'
    for unit in ('B', 'K', 'M', 'G', 'T', 'P'):
        if abs(n) < 1000 or unit == 'P':
            break
        n /= 1024
    if unit == 'B':
        return f'{n:.0f}B{suffix}'
    return f'{n:.1f}{unit}{suffix}' if abs(n) < 100 else f'{n:.0f}{unit}{suffix}'


def fmt_bits(nbytes):
    """Decimal bit rate: 125000 B/s -> '1.0Mb/s'."""
    if nbytes is None:
        return '–'
    n = nbytes * 8
    unit = 'b'
    for unit in ('b', 'Kb', 'Mb', 'Gb', 'Tb'):
        if n < 1000 or unit == 'Tb':
            break
        n /= 1000
    return f'{n:.0f}{unit}/s' if unit == 'b' or n >= 100 else f'{n:.1f}{unit}/s'


def fmt_duration(sec):
    sec = int(sec)
    d, sec = divmod(sec, 86400)
    h, sec = divmod(sec, 3600)
    m, s = divmod(sec, 60)
    if d:
        return f'{d}d {h:02d}h{m:02d}m'
    return f'{h}h{m:02d}m' if h else f'{m}m{s:02d}s'


def fmt_num(v, spec='.0f', unit='', none='–'):
    return none if v is None else f'{v:{spec}}{unit}'


def slug(s):
    return re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_')


def cmdline(pid, fallback=''):
    raw = read(f'/proc/{pid}/cmdline')
    # Collapse NUL separators and any newlines/tabs (python -c '...' scripts) to spaces.
    cmd = ' '.join(raw.replace('\x00', ' ').split()) if raw else ''
    return cmd or (f'[{fallback}]' if fallback else '?')


# ─────────────────────────── colours and glyphs ─────────────────────────

TEMP_LIMITS = {          # (warm, hot) °C for yellow / red
    'cpu': (70, 90),
    'gpu': (70, 85),
    'nvme': (55, 70),
    'hdd': (45, 55),
    'other': (60, 80),
}
EIGHTHS = ' ▏▎▍▌▋▊▉'
SPARK = '▁▂▃▄▅▆▇█'
TRACK = 'on grey23'      # background of the empty part of a bar


def level_style(value, warn, crit):
    if value is None:
        return 'dim'
    return 'green' if value < warn else 'yellow' if value < crit else 'bold red'


def temp_style(t, kind='other', crit=None):
    warn, hot = TEMP_LIMITS.get(kind, TEMP_LIMITS['other'])
    if crit and crit - 5 > warn:
        hot = min(hot, crit - 5)
    return level_style(t, warn, hot)


def temp_text(t, kind='other', crit=None, short=False):
    if t is None:
        return Text('–', style='dim')
    return Text(f'{t:.0f}°' if short else f'{t:.0f}°C', style=temp_style(t, kind, crit))


def bar(pct, width=12, warn=60, crit=85):
    """Horizontal bar with 1/8-cell resolution, coloured by level."""
    if pct is None:
        return Text(' ' * width, style=TRACK)
    p = max(0.0, min(100.0, pct))
    cells = p / 100 * width
    full = int(cells)
    frac = int((cells - full) * 8)
    filled = '█' * full + (EIGHTHS[frac] if full < width and frac else '')
    t = Text(filled, style=f'{level_style(p, warn, crit)} {TRACK}')
    t.append(' ' * (width - len(filled)), style=TRACK)
    return t


def sparkline(values, width, top=None, style='cyan', lo=0.0):
    """Last `width` values as ▁..█, scaled to [lo, top] (top defaults to the max)."""
    if width <= 0:
        return Text('')
    vals = [v or 0 for v in list(values)[-width:]]
    hi = top if top else max(vals, default=0)
    span = hi - lo
    if span <= 0:
        s = '▁' * len(vals)
    else:
        s = ''.join(SPARK[max(0, min(7, int((v - lo) / span * 7.999)))] for v in vals)
    return Text(s.rjust(width), style=style)


def kv(label, value, style=''):
    """'label value' fragment for one-line summaries."""
    t = Text(f'{label} ', style='dim')
    t.append(value if isinstance(value, Text) else Text(str(value), style=style))
    return t


def join(parts, sep='  '):
    """One summary line; it is cut with '…' rather than wrapped when too long."""
    t = Text(no_wrap=True, overflow='ellipsis')
    for i, p in enumerate(x for x in parts if x is not None):
        if i:
            t.append(sep)
        t.append(p if isinstance(p, Text) else Text(str(p)))
    return t


# ─────────────────────────── background polling ─────────────────────────

class Poller(threading.Thread):
    """Calls fn() every `period` s on a daemon thread; .value is the latest result.

    Used for anything slow or hang-prone (NVMe temperature queries take ~10 ms
    each, NVML blocks for seconds per call when the GPU stops answering, statfs
    never returns on a hung mount, scanning /proc can take 100+ ms under load),
    so a slow or hung device never stalls the dashboard.
    """

    def __init__(self, fn, period, name='poller'):
        super().__init__(daemon=True, name=name)
        self.fn, self.period = fn, period
        self.value = None
        self.error = None
        self.busy_since = None                # monotonic start of the call in progress
        self.first = threading.Event()        # set once the first call has returned
        self._halt = threading.Event()
        self.start()

    def run(self):
        while not self._halt.is_set():
            t0 = self.busy_since = time.monotonic()
            try:
                self.value = self.fn()
                self.error = None
            except Exception as exc:          # keep polling; .error holds the message
                self.error = f'{type(exc).__name__}: {exc}'
            self.busy_since = None
            self.first.set()
            self._halt.wait(max(0.05, self.period - (time.monotonic() - t0)))

    def stuck_for(self):
        """Seconds the call in progress has been running (0 between calls)."""
        t = self.busy_since
        return time.monotonic() - t if t is not None else 0.0

    def stop(self):
        self._halt.set()


# ───────────────────────────── hwmon sensors ────────────────────────────

class HwmonChip:
    """One /sys/class/hwmon/hwmonN device (static metadata read once)."""

    def __init__(self, path):
        self.path = path
        self.name = read(f'{path}/name') or os.path.basename(path)
        dev = f'{path}/device'
        self.device = os.path.realpath(dev) if os.path.exists(dev) else ''
        self.temps = []                                   # (label, input path, crit °C)
        for inp in sorted(glob.glob(f'{path}/temp*_input'), key=natural_key):
            base = inp[:-len('input')]
            label = read(base + 'label') or os.path.basename(base).rstrip('_')
            crit = read_int(base + 'crit')
            self.temps.append((label, inp, crit / 1000 if crit and 0 < crit < 200_000 else None))
        self.fans = []                                    # (label, input path)
        for inp in sorted(glob.glob(f'{path}/fan*_input'), key=natural_key):
            base = inp[:-len('input')]
            self.fans.append((read(base + 'label') or os.path.basename(base).rstrip('_'), inp))
        self.blocks = self._block_devices()

    def _block_devices(self):
        """Block devices whose temperature this chip reports (nvme / drivetemp)."""
        if not self.device:
            return []
        if self.name == 'nvme':
            found = glob.glob(self.device + '/nvme*n[0-9]*') + \
                glob.glob(self.device + '/nvme/nvme*/nvme*n[0-9]*')
            names = {os.path.basename(p) for p in found}
            return sorted((n for n in names if re.fullmatch(r'nvme\d+n\d+', n)), key=natural_key)
        if self.name == 'drivetemp':
            return sorted(os.path.basename(p) for p in glob.glob(self.device + '/block/*'))
        return []

    def read_temps(self):
        out = []
        for label, path, crit in self.temps:
            v = read_int(path)
            if v is not None and -40_000 < v < 150_000:
                out.append((label, v / 1000, crit))
        return out

    def read_fans(self):
        return [(label, read_int(path)) for label, path in self.fans]


class Sensors:
    # These chips query the drive itself on every read (~10 ms for NVMe;
    # drivetemp may even spin a disk up), so read them off-thread and rarely.
    SLOW = {'nvme': 5.0, 'drivetemp': 60.0}
    # DPTF placeholder zone that always reports a fixed 20 °C.
    SKIP_ZONES = {'INT3400 Thermal'}

    def __init__(self):
        self.chips = [HwmonChip(p) for p in
                      sorted(glob.glob('/sys/class/hwmon/hwmon*'), key=natural_key)]
        names = {c.name for c in self.chips}
        self.zones = []
        for z in sorted(glob.glob('/sys/class/thermal/thermal_zone*'), key=natural_key):
            typ = read(f'{z}/type') or os.path.basename(z)
            if typ in names or typ in self.SKIP_ZONES:
                continue              # already covered by hwmon (acpitz, iwlwifi, ...)
            if typ == 'x86_pkg_temp' and 'coretemp' in names:
                continue              # same value as coretemp "Package id 0"
            self.zones.append((typ, f'{z}/temp'))
        self.pollers = {c.path: Poller(c.read_temps, self.SLOW[c.name], f'hwmon-{c.name}')
                        for c in self.chips if c.name in self.SLOW and c.temps}
        self.fan_hist = {}

    def sample(self):
        chips = []
        for c in self.chips:
            p = self.pollers.get(c.path)
            temps = (p.value or []) if p else c.read_temps()
            fans = c.read_fans()
            if temps or fans:
                chips.append({'name': c.name, 'blocks': c.blocks, 'temps': temps, 'fans': fans})
        zones = []
        for typ, path in self.zones:
            v = read_int(path)
            if v is not None and -40_000 < v < 150_000:
                zones.append((typ, v / 1000))
        fans = []
        n_acpi = 0
        for ch in chips:
            for label, rpm in ch['fans']:
                if ch['name'] == 'acpi_fan':
                    n_acpi += 1
                    name = f'Fan {n_acpi}'
                else:
                    name = f"{ch['name']} {label}"
                h = self.fan_hist.setdefault(name, deque(maxlen=HISTORY))
                h.append(rpm or 0)
                fans.append({'name': name, 'rpm': rpm, 'hist': h})
        return {'chips': chips, 'zones': zones, 'fans': fans}


# ───────────────────────────────── CPU ──────────────────────────────────

class RAPL:
    """CPU energy counters (intel-rapl powercap). energy_uj is root-only."""
    NAMES = {'package-0': 'package', 'core': 'cores', 'uncore': 'uncore',
             'dram': 'dram', 'psys': 'platform'}

    def __init__(self):
        paths = sorted(glob.glob('/sys/class/powercap/intel-rapl:*'), key=natural_key)
        self.present = bool(paths)
        self.domains = []
        now = time.monotonic()
        for d in paths:
            e = read_int(f'{d}/energy_uj')
            if e is None:
                continue
            name = read(f'{d}/name') or os.path.basename(d)
            rng = read_int(f'{d}/max_energy_range_uj') or 2 ** 32
            self.domains.append([self.NAMES.get(name, name), f'{d}/energy_uj', rng, e, now])
        self.readable = bool(self.domains)

    def sample(self):
        out = {}
        for dom in self.domains:
            label, path, rng, last_e, last_t = dom
            e, now = read_int(path), time.monotonic()
            if e is None:
                continue
            de = e - last_e
            if de < 0:
                de += rng                       # counter wrapped
            if now - last_t > 0:
                out[label] = de / (now - last_t) / 1e6
            dom[3], dom[4] = e, now
        return out


class CPU:
    def __init__(self):
        self.n = psutil.cpu_count() or os.cpu_count() or 1
        self.model = self._model()
        p_cores = parse_cpulist(read('/sys/devices/cpu_core/cpus'))
        e_cores = parse_cpulist(read('/sys/devices/cpu_atom/cpus'))
        self.kind = ['P' if c in p_cores else 'E' if c in e_cores else '' for c in range(self.n)]
        self.core_id = [read_int(f'/sys/devices/system/cpu/cpu{c}/topology/core_id')
                        for c in range(self.n)]
        # Max turbo from the P-cores only: some E-cores report a bogus
        # cpuinfo_max_freq (6.5 GHz on a 275HX whose E-cores top out at 4.6 GHz).
        fast = [c for c in range(self.n) if self.kind[c] == 'P'] or range(self.n)
        self.max_mhz = max((read_int(f'/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq') or 0)
                           for c in fast) / 1000 or None
        # Intel processor_thermal_device: TCC offset in °C below TjMax, i.e. where
        # the CPU starts thermal throttling (BIOS/EC set it, often 5-20 °C).
        self.tcc_path = next(iter(sorted(glob.glob('/sys/bus/pci/devices/*/tcc_offset_degree_celsius'))), None)
        self.pkg_of = [read_int(f'/sys/devices/system/cpu/cpu{c}/topology/physical_package_id') or 0
                       for c in range(self.n)]
        self.rapl = RAPL()
        self.hist = deque(maxlen=HISTORY)
        self.throttle_prev = self._throttle_counters()
        psutil.cpu_percent(percpu=True)         # prime the delta counters
        psutil.cpu_percent()
        psutil.cpu_times_percent()

    @staticmethod
    def _model():
        for line in (read('/proc/cpuinfo') or '').splitlines():
            if line.startswith('model name'):
                name = line.split(':', 1)[1].replace('(R)', '').replace('(TM)', '')
                return re.sub(r'\s+', ' ', name).strip()
        return platform.processor() or 'CPU'

    def _throttle_counters(self):
        """Kernel thermal-throttle (PROCHOT / TCC) event counters: (time, core events
        summed over CPUs, package events, package throttled ms), or None if absent."""
        base = '/sys/devices/system/cpu/cpu{}/thermal_throttle/'
        core, pkg_events, pkg_ms = 0, {}, {}
        found = False
        for c in range(self.n):
            v = read_int(base.format(c) + 'core_throttle_count')
            if v is None:
                continue
            found = True
            core += v
            pk = self.pkg_of[c]
            if pk not in pkg_events:              # package counters repeat on every CPU
                pkg_events[pk] = read_int(base.format(c) + 'package_throttle_count') or 0
                pkg_ms[pk] = read_int(base.format(c) + 'package_throttle_total_time_ms') or 0
        if not found:
            return None
        return time.monotonic(), core, sum(pkg_events.values()), sum(pkg_ms.values())

    def _throttle_rates(self):
        cur, prev = self._throttle_counters(), self.throttle_prev
        self.throttle_prev = cur
        if not cur or not prev or cur[0] <= prev[0]:
            return None
        dt_s = cur[0] - prev[0]
        return {'core_eps': max(0, cur[1] - prev[1]) / dt_s,
                'pkg_eps': max(0, cur[2] - prev[2]) / dt_s,
                'pkg_time_pct': min(100.0, max(0, cur[3] - prev[3]) / (dt_s * 1000) * 100)}

    def sample(self, sensors):
        per = psutil.cpu_percent(percpu=True)
        total = psutil.cpu_percent()
        times = psutil.cpu_times_percent()._asdict()
        mhz = []
        for c in range(self.n):
            khz = read_int(f'/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq')
            mhz.append(khz / 1000 if khz else None)

        core_temp, pkg, tjmax = {}, None, None
        for ch in sensors['chips']:
            if ch['name'] == 'coretemp':
                for label, t, crit in ch['temps']:
                    m = re.fullmatch(r'Core (\d+)', label)
                    if m:
                        core_temp[int(m.group(1))] = t
                    elif label.startswith('Package'):
                        pkg = t if pkg is None else max(pkg, t)
                        tjmax = crit or tjmax
            elif ch['name'] in ('k10temp', 'zenpower') and pkg is None:
                pkg = next((t for label, t, _ in ch['temps'] if label in ('Tctl', 'Tdie')), None)
        offset = read_int(self.tcc_path) if self.tcc_path else None
        limit = tjmax - offset if tjmax and offset is not None else tjmax

        cores = [{'cpu': c, 'kind': self.kind[c],
                  'pct': per[c] if c < len(per) else None,
                  'mhz': mhz[c], 'temp': core_temp.get(self.core_id[c])}
                 for c in range(self.n)]
        temps = [k['temp'] for k in cores if k['temp'] is not None]
        freqs = [f for f in mhz if f]
        self.hist.append(total)
        min_khz = read_int('/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq')
        # Busy cores all running below the OS minimum clock means firmware/EC is
        # forcing them down (PROCHOT, TCC offset, power-delivery limit ...).
        # Judged per core, so a few busy threads on an otherwise idle CPU count too.
        busy = [k['mhz'] for k in cores if (k['pct'] or 0) >= 50 and k['mhz']]
        clamped = bool(busy and min_khz and max(busy) < min_khz / 1000)
        return {
            'model': self.model, 'n': self.n,
            'n_p': self.kind.count('P'), 'n_e': self.kind.count('E'),
            'total': total, 'times': times, 'cores': cores,
            'load': os.getloadavg(),
            'freq_avg': sum(freqs) / len(freqs) if freqs else None,
            'freq_top': max(freqs) if freqs else None, 'freq_max': self.max_mhz,
            'pkg_temp': pkg, 'temp_limit': limit,
            'core_temp_max': max(temps) if temps else None,
            'core_temp_avg': sum(temps) / len(temps) if temps else None,
            'governor': read('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'),
            'epp': read('/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference'),
            'power': self.rapl.sample(), 'power_readable': self.rapl.readable,
            'power_present': self.rapl.present,
            'throttle': self._throttle_rates(), 'clamped': clamped,
            'hist': self.hist,
        }


# ──────────────────────────────── memory ────────────────────────────────

def sample_memory():
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    info = {}
    for line in (read('/proc/meminfo') or '').splitlines():
        k, _, v = line.partition(':')
        if k in ('Dirty', 'Writeback'):
            info[k] = int(v.split()[0]) * 1024
    return {
        'total': vm.total, 'avail': vm.available, 'used': vm.total - vm.available,
        'pct': vm.percent,
        'cache': getattr(vm, 'cached', 0) + getattr(vm, 'buffers', 0),
        'dirty': info.get('Dirty'), 'writeback': info.get('Writeback'),
        'swap_total': sw.total, 'swap_used': sw.used, 'swap_pct': sw.percent,
    }


# ─────────────────────────────── NVIDIA GPU ─────────────────────────────

class _Util(ctypes.Structure):
    _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]


class _MemV1(ctypes.Structure):
    _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong),
                ('used', ctypes.c_ulonglong)]


class _MemV2(ctypes.Structure):
    _fields_ = [('version', ctypes.c_uint), ('total', ctypes.c_ulonglong),
                ('reserved', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong),
                ('used', ctypes.c_ulonglong)]


class _PciInfo(ctypes.Structure):
    _fields_ = [('busIdLegacy', ctypes.c_char * 16), ('domain', ctypes.c_uint),
                ('bus', ctypes.c_uint), ('device', ctypes.c_uint),
                ('pciDeviceId', ctypes.c_uint), ('pciSubSystemId', ctypes.c_uint),
                ('busId', ctypes.c_char * 32)]


class _ProcInfo(ctypes.Structure):       # nvmlProcessInfo_t (v2/v3 layout)
    _fields_ = [('pid', ctypes.c_uint), ('usedGpuMemory', ctypes.c_ulonglong),
                ('gpuInstanceId', ctypes.c_uint), ('computeInstanceId', ctypes.c_uint)]


class _Value(ctypes.Union):
    _fields_ = [('dVal', ctypes.c_double), ('uiVal', ctypes.c_uint), ('ulVal', ctypes.c_ulong),
                ('ullVal', ctypes.c_ulonglong), ('sllVal', ctypes.c_longlong)]


class _FieldValue(ctypes.Structure):     # nvmlFieldValue_t
    _fields_ = [('fieldId', ctypes.c_uint), ('scopeId', ctypes.c_uint),
                ('timestamp', ctypes.c_longlong), ('latencyUsec', ctypes.c_longlong),
                ('valueType', ctypes.c_uint), ('nvmlReturn', ctypes.c_uint), ('value', _Value)]


NVML_SUCCESS, NVML_INSUFFICIENT_SIZE = 0, 7
# Return codes that mean the GPU itself is in trouble rather than "not supported":
# TIMEOUT, GPU_IS_LOST (fell off the bus), RESET_REQUIRED, UNKNOWN.
NVML_GPU_FAILED = {10, 15, 16, 999}
NVML_FI_DEV_GET_GPU_RECOVERY_ACTION = 230      # what the driver wants done after a fault (Xid 154)
RECOVERY_ACTIONS = {1: 'GPU reset required', 2: 'reboot required',
                    3: 'drain P2P required', 4: 'drain and reset required'}
THROTTLE_BITS = [(0x1, 'idle'), (0x2, 'app-clocks'), (0x4, 'power-cap'),
                 (0x8, 'hw-slowdown'), (0x10, 'sync-boost'), (0x20, 'sw-thermal'),
                 (0x40, 'hw-thermal'), (0x80, 'power-brake'), (0x100, 'display-clock')]


class NvidiaGPUs:
    """NVIDIA GPUs through libnvidia-ml (NVML) via ctypes - no subprocess per tick.

    All NVML calls, including nvmlInit, run on background threads: when the GPU
    stops answering (Xid 119 'GSP RPC timeout' and the like) every call blocks
    for seconds, and that must not freeze the dashboard or the CSV log."""

    def __init__(self, interval=1.0):
        self.lib, self.gpus, self.error, self.driver = None, [], None, None
        self.fast = self.slow = None
        self.t0 = time.monotonic()
        self.inited = threading.Event()
        threading.Thread(target=self._start, args=(interval,), daemon=True, name='nvml-init').start()

    def _start(self, interval):
        try:
            self._init()
        except Exception as exc:
            self.error = f'NVML: {type(exc).__name__}: {exc}'
        if self.gpus:
            # PCIe throughput sampling blocks ~20 ms per call and the process lists
            # change slowly: poll those less often than the rest.
            self.slow = Poller(self._slow_sample, 2.0, 'nvml-slow')
            self.fast = Poller(self._sample, interval, 'nvml')
        self.inited.set()

    def wait_ready(self, timeout):
        """Block until the first GPU reading exists, or `timeout` s have passed."""
        end = time.monotonic() + timeout
        if self.inited.wait(timeout) and self.fast:
            self.fast.first.wait(max(0.0, end - time.monotonic()))

    def _init(self):
        try:
            lib = ctypes.CDLL('libnvidia-ml.so.1')
        except OSError:
            return                               # no NVIDIA driver installed: nothing to show
        lib.nvmlErrorString.restype = ctypes.c_char_p
        rc = lib.nvmlInit_v2()
        if rc != NVML_SUCCESS:
            self.error = f'nvmlInit failed: {lib.nvmlErrorString(rc).decode()}'
            return
        self.lib = lib
        buf = ctypes.create_string_buffer(96)
        if lib.nvmlSystemGetDriverVersion(buf, 96) == NVML_SUCCESS:
            self.driver = buf.value.decode()
        count = ctypes.c_uint(0)
        lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
        for i in range(count.value):
            h = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != NVML_SUCCESS:
                continue
            name = 'NVIDIA GPU'
            if lib.nvmlDeviceGetName(h, buf, 96) == NVML_SUCCESS:
                name = buf.value.decode()
            pci = _PciInfo()
            runtime = None
            if self._fn('nvmlDeviceGetPciInfo_v3') and \
                    lib.nvmlDeviceGetPciInfo_v3(h, ctypes.byref(pci)) == NVML_SUCCESS:
                runtime = (f'/sys/bus/pci/devices/{pci.domain:04x}:{pci.bus:02x}:'
                           f'{pci.device:02x}.0/power/runtime_status')
            limit = self._get('nvmlDeviceGetEnforcedPowerLimit', h)
            self.gpus.append({
                'index': i, 'handle': h, 'name': name, 'runtime': runtime,
                'max_clk': self._get('nvmlDeviceGetMaxClockInfo', h, 0),
                'max_mem_clk': self._get('nvmlDeviceGetMaxClockInfo', h, 2),
                'slowdown': self._get('nvmlDeviceGetTemperatureThreshold', h, 1),
                'power_limit': limit / 1000 if limit else None,
                'hist': deque(maxlen=HISTORY),
            })

    def _fn(self, name):
        return getattr(self.lib, name, None)

    def _get(self, fn_name, h, *args, ctype=ctypes.c_uint):
        fn = self._fn(fn_name)
        if fn is None:
            return None
        v = ctype()
        return v.value if fn(h, *args, ctypes.byref(v)) == NVML_SUCCESS else None

    def _asleep(self, g):
        """True while the dGPU is runtime-suspended; querying it would wake it."""
        return g['runtime'] is not None and read(g['runtime']) == 'suspended'

    def _procs(self, h):
        found = {}
        for fn_name, kind in (('nvmlDeviceGetComputeRunningProcesses_v3', 'C'),
                              ('nvmlDeviceGetGraphicsRunningProcesses_v3', 'G')):
            fn = self._fn(fn_name)
            if fn is None:
                continue
            n = ctypes.c_uint(0)
            rc = fn(h, ctypes.byref(n), None)
            if rc == NVML_SUCCESS or rc != NVML_INSUFFICIENT_SIZE:
                continue                     # none running, or unsupported
            size = n.value + 16
            arr = (_ProcInfo * size)()
            n = ctypes.c_uint(size)
            if fn(h, ctypes.byref(n), arr) != NVML_SUCCESS:
                continue
            for p in arr[:n.value]:
                mem = None if p.usedGpuMemory >= 2 ** 63 else p.usedGpuMemory
                prev = found.get(p.pid)
                if prev:
                    prev['kind'] = ''.join(sorted(set(prev['kind'] + kind)))
                    prev['mem'] = max(prev['mem'] or 0, mem or 0) or None
                else:
                    found[p.pid] = {'pid': p.pid, 'kind': kind, 'mem': mem}
        procs = sorted(found.values(), key=lambda p: -(p['mem'] or 0))
        for p in procs:
            p['cmd'] = cmdline(p['pid'])
        return procs

    def _recovery_action(self, h):
        """Pending driver-requested recovery after a fault (None when all is well)."""
        fn = self._fn('nvmlDeviceGetFieldValues')
        if fn is None:
            return None
        fv = _FieldValue(fieldId=NVML_FI_DEV_GET_GPU_RECOVERY_ACTION)
        if fn(h, 1, ctypes.byref(fv)) != NVML_SUCCESS or fv.nvmlReturn != NVML_SUCCESS:
            return None
        return RECOVERY_ACTIONS.get(fv.value.uiVal)

    def _slow_sample(self):
        out = {}
        for g in self.gpus:
            if self._asleep(g):
                continue
            h = g['handle']
            out[g['index']] = {
                'pcie_tx': (self._get('nvmlDeviceGetPcieThroughput', h, 0) or 0) * 1024,
                'pcie_rx': (self._get('nvmlDeviceGetPcieThroughput', h, 1) or 0) * 1024,
                'recovery': self._recovery_action(h),
                'procs': self._procs(h),
            }
        return out

    def sample(self):
        """(latest per-GPU readings, (status message, style) or None). Never blocks."""
        if not self.inited.is_set():
            waited = time.monotonic() - self.t0
            if waited < 5:
                return [], ('NVML starting…', 'dim')
            return [], (f'NVML init not answering for {waited:.0f} s - GPU or driver hung?', 'bold red')
        if not self.fast:
            return [], (self.error, 'yellow') if self.error else None
        stuck = max(self.fast.stuck_for(), self.slow.stuck_for())
        if stuck > 3:
            status = (f'NVML not answering for {stuck:.0f} s - GPU or driver hung? '
                      f'(showing the last readings)', 'bold red')
        elif self.fast.error:
            status = (f'NVML sampling failed: {self.fast.error}', 'yellow')
        else:
            status = None
        return self.fast.value or [], status

    def _sample(self):
        out = []
        slow = (self.slow.value if self.slow else None) or {}
        for g in self.gpus:
            d = {'index': g['index'], 'name': g['name'], 'driver': self.driver,
                 'power_limit': g['power_limit'], 'max_clk': g['max_clk'],
                 'max_mem_clk': g['max_mem_clk'], 'slowdown': g['slowdown'],
                 'hist': g['hist']}
            if self._asleep(g):
                d['asleep'] = True
                g['hist'].append(0)
                out.append(d)
                continue
            h, lib = g['handle'], self.lib
            t = ctypes.c_uint()
            rc = lib.nvmlDeviceGetTemperature(h, 0, ctypes.byref(t))
            if rc in NVML_GPU_FAILED:               # e.g. 'GPU is lost' after an Xid 79/119
                d['error'] = lib.nvmlErrorString(rc).decode()
                g['hist'].append(0)
                out.append(d)
                continue
            d['temp'] = t.value if rc == NVML_SUCCESS else None
            u = _Util()
            if lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(u)) == NVML_SUCCESS:
                d['util'], d['mem_util'] = u.gpu, u.memory
            m2 = _MemV2(version=ctypes.sizeof(_MemV2) | (2 << 24))
            if self._fn('nvmlDeviceGetMemoryInfo_v2') and \
                    lib.nvmlDeviceGetMemoryInfo_v2(h, ctypes.byref(m2)) == NVML_SUCCESS:
                d['vram_used'], d['vram_total'] = m2.used, m2.total    # matches nvidia-smi
            else:
                m1 = _MemV1()
                if lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m1)) == NVML_SUCCESS:
                    d['vram_used'], d['vram_total'] = m1.used, m1.total
            mw = self._get('nvmlDeviceGetPowerUsage', h)
            d['power'] = mw / 1000 if mw is not None else None
            d['clk'] = self._get('nvmlDeviceGetClockInfo', h, 0)
            d['mem_clk'] = self._get('nvmlDeviceGetClockInfo', h, 2)
            ps = self._get('nvmlDeviceGetPerformanceState', h, ctype=ctypes.c_int)
            d['pstate'] = f'P{ps}' if ps is not None and 0 <= ps < 32 else None
            d['fan'] = self._get('nvmlDeviceGetFanSpeed', h)     # N/A on most laptops
            for key, fn_name in (('enc', 'nvmlDeviceGetEncoderUtilization'),
                                 ('dec', 'nvmlDeviceGetDecoderUtilization')):
                v, period = ctypes.c_uint(), ctypes.c_uint()
                fn = self._fn(fn_name)
                if fn and fn(h, ctypes.byref(v), ctypes.byref(period)) == NVML_SUCCESS:
                    d[key] = v.value
            d['pcie_gen'] = self._get('nvmlDeviceGetCurrPcieLinkGeneration', h)
            d['pcie_width'] = self._get('nvmlDeviceGetCurrPcieLinkWidth', h)
            reasons = self._get('nvmlDeviceGetCurrentClocksEventReasons', h, ctype=ctypes.c_ulonglong)
            if reasons is None:
                reasons = self._get('nvmlDeviceGetCurrentClocksThrottleReasons', h,
                                    ctype=ctypes.c_ulonglong)
            if reasons is not None:
                d['throttle'] = [name for bit, name in THROTTLE_BITS if reasons & bit]
            d.update(slow.get(g['index'], {}))
            g['hist'].append(d.get('util') or 0)
            out.append(d)
        return out


# ─────────────────────────────── Intel iGPU ─────────────────────────────

class IntelGPU:
    """Intel integrated graphics (i915 / xe): frequency, and busy % derived from
    RC6 (idle-state) residency. RC6 residency is the only per-GT activity counter
    readable without root, so busy % is an approximation ('≈')."""

    def __init__(self):
        self.cards = []
        for card in sorted(glob.glob('/sys/class/drm/card[0-9]*'), key=natural_key):
            if '-' in os.path.basename(card):
                continue                                    # connector, not a GPU
            drv = os.path.basename(os.path.realpath(f'{card}/device/driver'))
            gts = []
            if drv == 'i915':
                act, top = f'{card}/gt_act_freq_mhz', read_int(f'{card}/gt_max_freq_mhz')
                for gt in sorted(glob.glob(f'{card}/gt/gt[0-9]*'), key=natural_key):
                    idx = int(re.search(r'(\d+)$', gt).group(1))
                    gts.append({'label': 'render' if idx == 0 else 'media' if idx == 1 else f'gt{idx}',
                                'idle': f'{gt}/rc6_residency_ms'})
                if not gts and os.path.exists(f'{card}/power/rc6_residency_ms'):
                    gts.append({'label': 'render', 'idle': f'{card}/power/rc6_residency_ms'})
            elif drv == 'xe':
                gt_dirs = sorted(glob.glob(f'{card}/device/tile*/gt[0-9]*'), key=natural_key)
                act = f'{gt_dirs[0]}/freq0/act_freq' if gt_dirs else None
                top = read_int(f'{gt_dirs[0]}/freq0/max_freq') if gt_dirs else None
                for gt in gt_dirs:
                    name = read(f'{gt}/gtidle/name') or os.path.basename(gt)
                    label = 'render' if name.endswith('rc') else 'media' if name.endswith('mc') else name
                    gts.append({'label': label, 'idle': f'{gt}/gtidle/idle_residency_ms'})
            else:
                continue
            for g in gts:
                g['prev'], g['prev_t'] = read_int(g['idle']), time.monotonic()
            # Frequency-limit reasons of the primary GT (pl1, prochot, thermal, ...).
            tdir = (f'{card}/gt/gt0' if drv == 'i915' else
                    next(iter(sorted(glob.glob(f'{card}/device/tile0/gt0/freq0/throttle'))), ''))
            prefix = 'throttle_reason_' if drv == 'i915' else 'reason_'
            reasons = [(os.path.basename(p)[len(prefix):], p)
                       for p in sorted(glob.glob(f'{tdir}/{prefix}*'))
                       if not p.endswith(prefix + 'status')] if tdir else []
            self.cards.append({'card': os.path.basename(card), 'driver': drv,
                               'act': act, 'max': top, 'gts': gts, 'reasons': reasons,
                               'hist': deque(maxlen=HISTORY)})

    def sample(self):
        out = []
        for c in self.cards:
            d = {'card': c['card'], 'driver': c['driver'],
                 'freq': read_int(c['act']) if c['act'] else None, 'max': c['max'],
                 'engines': [], 'hist': c['hist']}
            for g in c['gts']:
                v, now = read_int(g['idle']), time.monotonic()
                busy = None
                if v is not None and g['prev'] is not None and now > g['prev_t']:
                    idle_ms = v - g['prev']
                    busy = max(0.0, min(100.0, 100 - idle_ms / ((now - g['prev_t']) * 1000) * 100))
                g['prev'], g['prev_t'] = v, now
                d['engines'].append((g['label'], busy))
            d['limits'] = [name for name, path in c['reasons'] if read_int(path) == 1]
            render = next((b for label, b in d['engines'] if label == 'render'), None)
            c['hist'].append(render or 0)
            out.append(d)
        return out


# ───────────────────────────────── disks ────────────────────────────────

def parent_disk(name, depth=0):
    """'nvme0n1p5' -> 'nvme0n1'; dm/LUKS/LVM devices are followed to their disk."""
    sysp = f'/sys/class/block/{name}'
    if os.path.exists(f'{sysp}/partition'):
        return os.path.basename(os.path.dirname(os.path.realpath(sysp)))
    slaves = f'{sysp}/slaves'
    if depth < 8 and os.path.isdir(slaves):
        members = sorted(os.listdir(slaves))
        if members:
            return parent_disk(members[0], depth + 1)
    return name


class Disks:
    def __init__(self, want_smart=True):
        self._scan()
        self.prev, self.prev_t = None, None
        self.hist = {}
        self.usage = Poller(self._usage, 5.0, 'disk-usage')
        self.smart, self.smart_hint = None, None
        has_hdd = any(d['bus'] != 'NVMe' for d in self.disks.values())
        if want_smart and has_hdd:
            if os.geteuid() != 0:
                self.smart_hint = 'HDD temps: run with sudo (needs smartmontools)'
            elif not shutil.which('smartctl'):
                self.smart_hint = 'HDD temps: sudo apt install smartmontools'
            else:
                self.smart = Poller(self._smart, 60.0, 'smart')

    def _scan(self):
        disks = {}
        for name in sorted(os.listdir('/sys/block'), key=natural_key):
            if name.startswith(('loop', 'ram', 'zram', 'sr', 'fd', 'dm-')):
                continue
            sysp = f'/sys/block/{name}'
            size = (read_int(f'{sysp}/size') or 0) * 512
            if not size:
                continue                      # empty card-reader slots etc.
            real = os.path.realpath(sysp)
            vendor = read(f'{sysp}/device/vendor') or ''
            model = read(f'{sysp}/device/model') or ''
            if vendor and vendor not in ('ATA',) and vendor.lower() not in model.lower():
                model = f'{vendor} {model}'
            if name.startswith('nvme'):
                bus = 'NVMe'
            elif '/usb' in real:
                bus = 'USB'
            elif name.startswith('mmcblk'):
                bus = 'MMC'
            elif '/virtual/' in real:
                bus = 'virt'
            else:
                bus = 'SATA'
            disks[name] = {'name': name, 'model': model.strip(), 'size': size, 'bus': bus,
                           'rota': read_int(f'{sysp}/queue/rotational') == 1}
        self.disks = disks
        self.scanned = time.monotonic()

    def _usage(self):
        res, seen = {}, set()
        disks = self.disks
        for p in psutil.disk_partitions(all=False):
            if not p.device.startswith('/dev/'):
                continue
            dev = os.path.basename(os.path.realpath(p.device))
            if dev in seen:
                continue                      # bind mount of an already-listed device
            parent = parent_disk(dev)
            if parent not in disks:
                continue                      # loop devices (snaps) etc.
            try:
                u = psutil.disk_usage(p.mountpoint)
            except OSError:
                continue
            seen.add(dev)
            res.setdefault(parent, []).append({
                'mount': p.mountpoint, 'fstype': p.fstype, 'total': u.total,
                'used': u.used, 'free': u.free, 'pct': u.percent})
        return res

    def _smart(self):
        """Drive temperatures via smartctl. `-n standby` never spins up a sleeping disk."""
        out = {}
        for name, d in list(self.disks.items()):
            if d['bus'] == 'NVMe':
                continue                      # hwmon already reports NVMe temps
            for extra in ([], ['-d', 'sat']):
                try:
                    p = subprocess.run(['smartctl', '-n', 'standby', '-A', '-j', *extra, f'/dev/{name}'],
                                       capture_output=True, text=True, timeout=20, check=False)
                    j = json.loads(p.stdout or '{}')
                except (OSError, subprocess.TimeoutExpired, ValueError):
                    break
                temp = (j.get('temperature') or {}).get('current')
                msgs = ' '.join(m.get('string', '') for m in
                                (j.get('smartctl') or {}).get('messages', []))
                if temp is not None:
                    out[name] = float(temp)
                    break
                if 'STANDBY' in msgs.upper() or 'SLEEP' in msgs.upper():
                    out[name] = 'standby'
                    break
        return out

    def sample(self, sensors):
        now = time.monotonic()
        if now - self.scanned > 10:
            self._scan()                      # pick up hot-plugged drives
        io = psutil.disk_io_counters(perdisk=True) or {}
        temps = {}
        for ch in sensors['chips']:
            if ch['blocks'] and ch['temps']:
                for b in ch['blocks']:
                    temps[b] = ch['temps'][0][1]          # nvme: 'Composite'
        for name, v in ((self.smart.value if self.smart else None) or {}).items():
            temps.setdefault(name, v)
        usage = self.usage.value or {}
        dt_s = now - self.prev_t if self.prev_t else None
        rows = []
        for name, d in self.disks.items():
            r = dict(d, read=None, write=None, iops=None, busy=None,
                     temp=temps.get(name), mounts=usage.get(name, []))
            c = io.get(name)
            p = (self.prev or {}).get(name)
            if c and p and dt_s:
                r['read'] = max(0, c.read_bytes - p.read_bytes) / dt_s
                r['write'] = max(0, c.write_bytes - p.write_bytes) / dt_s
                r['iops'] = max(0, c.read_count - p.read_count + c.write_count - p.write_count) / dt_s
                r['busy'] = min(100.0, max(0, c.busy_time - p.busy_time) / (dt_s * 1000) * 100)
            h = self.hist.setdefault(name, deque(maxlen=HISTORY))
            h.append((r['read'] or 0) + (r['write'] or 0))
            r['hist'] = h
            rows.append(r)
        self.prev, self.prev_t = io, now
        stuck = self.usage.stuck_for()
        return {'disks': rows, 'smart_hint': self.smart_hint,
                'usage_hint': f'filesystem usage not updating for {stuck:.0f} s - '
                              f'a mount is probably hung' if stuck > 30 else None}


# ──────────────────────────────── network ───────────────────────────────

def nmcli_wifi():
    """Active Wi-Fi link per device from NetworkManager (cached scan, no rescan)."""
    p = subprocess.run(['nmcli', '-t', '-f', 'IN-USE,DEVICE,SSID,SIGNAL,RATE,CHAN',
                        'device', 'wifi', 'list', '--rescan', 'no'],
                       capture_output=True, text=True, timeout=10, check=False)
    out = {}
    for line in p.stdout.splitlines():
        f = [x.replace('\\:', ':') for x in re.split(r'(?<!\\):', line)]
        if len(f) >= 6 and f[0] == '*':
            out[f[1]] = {'ssid': f[2], 'signal': int(f[3]) if f[3].isdigit() else None,
                         'rate': f[4].replace(' Mbit/s', 'Mb/s'), 'chan': f[5]}
    return out


def tcp_latency(host, port=443):
    """Round-trip of a TCP handshake in ms (works where ICMP ping is blocked)."""
    t0 = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=3):
            return {'host': host, 'ms': (time.perf_counter() - t0) * 1000}
    except OSError:
        return {'host': host, 'ms': None}


class Network:
    def __init__(self, show_all=False, ping=None, bits=False):
        self.show_all, self.bits = show_all, bits
        self.prev, self.prev_t = None, None
        self.hist = {}
        self.addrs, self.addrs_t = {}, 0.0
        has_wifi = any(os.path.isdir(f'/sys/class/net/{n}/wireless')
                       for n in os.listdir('/sys/class/net'))
        self.wifi = Poller(nmcli_wifi, 15.0, 'nmcli') if has_wifi and shutil.which('nmcli') else None
        self.latency = Poller(lambda: tcp_latency(ping), 5.0, 'latency') if ping else None

    def sample(self):
        now = time.monotonic()
        cnt = psutil.net_io_counters(pernic=True)
        stats = psutil.net_if_stats()
        if now - self.addrs_t > 10:
            self.addrs, self.addrs_t = {}, now
            for nic, addrs in psutil.net_if_addrs().items():
                v4 = [a.address for a in addrs if a.family == socket.AF_INET]
                if v4:
                    self.addrs[nic] = v4[0]
        wifi = (self.wifi.value if self.wifi else None) or {}
        dt_s = now - self.prev_t if self.prev_t else None
        rows, hidden = [], []
        for nic in sorted(cnt, key=natural_key):
            if nic == 'lo' or nic.startswith('veth'):
                continue
            oper = read(f'/sys/class/net/{nic}/operstate') or 'unknown'
            st = stats.get(nic)
            up = bool(st and st.isup) and oper in ('up', 'unknown')
            if not up and not self.show_all:
                hidden.append(nic)
                continue
            c, p = cnt[nic], (self.prev or {}).get(nic)
            r = {'name': nic, 'up': up, 'addr': self.addrs.get(nic),
                 'speed': st.speed if st and st.speed and st.speed > 0 else None,
                 'wireless': os.path.isdir(f'/sys/class/net/{nic}/wireless'),
                 'wifi': wifi.get(nic), 'rx_total': c.bytes_recv, 'tx_total': c.bytes_sent,
                 'rx': None, 'tx': None, 'errs': None}
            if p and dt_s:
                r['rx'] = max(0, c.bytes_recv - p.bytes_recv) / dt_s
                r['tx'] = max(0, c.bytes_sent - p.bytes_sent) / dt_s
                r['errs'] = max(0, (c.errin + c.errout + c.dropin + c.dropout)
                                - (p.errin + p.errout + p.dropin + p.dropout))
            h = self.hist.setdefault(nic, deque(maxlen=HISTORY))
            h.append((r['rx'] or 0) + (r['tx'] or 0))
            r['hist'] = h
            rows.append(r)
        self.prev, self.prev_t = cnt, now
        return {'nics': rows, 'hidden': hidden, 'bits': self.bits,
                'latency': self.latency.value if self.latency else None}


# ───────────────────────────────── power ────────────────────────────────

def read_power_supplies():
    ac, bats, periph = None, [], []
    for ps in sorted(glob.glob('/sys/class/power_supply/*'), key=natural_key):
        ue = {}
        for line in (read(f'{ps}/uevent') or '').splitlines():
            k, _, v = line.partition('=')
            ue[k.removeprefix('POWER_SUPPLY_')] = v

        def num(key, ue=ue):
            v = ue.get(key, '')
            return int(v) if v.lstrip('-').isdigit() else None

        typ = ue.get('TYPE')
        if typ == 'Mains':
            ac = bool(ac) or ue.get('ONLINE') == '1'
        elif typ == 'Battery':
            name = ue.get('MODEL_NAME') or ue.get('NAME') or os.path.basename(ps)
            if ue.get('SCOPE') == 'Device':                  # mouse, keyboard, ...
                periph.append({'name': name, 'capacity': num('CAPACITY'),
                               'level': ue.get('CAPACITY_LEVEL'), 'status': ue.get('STATUS')})
                continue
            volt, cur, pw = num('VOLTAGE_NOW'), num('CURRENT_NOW'), num('POWER_NOW')
            if pw is not None:
                watts = abs(pw) / 1e6
            elif volt is not None and cur is not None:
                watts = abs(volt * cur) / 1e12
            else:
                watts = None
            if num('ENERGY_NOW') is not None:                 # µWh
                now_wh, full_wh = num('ENERGY_NOW') / 1e6, (num('ENERGY_FULL') or 0) / 1e6
                design_wh = (num('ENERGY_FULL_DESIGN') or 0) / 1e6
            elif num('CHARGE_NOW') is not None and volt:      # µAh × µV
                v_nom = num('VOLTAGE_MIN_DESIGN') or volt
                now_wh = num('CHARGE_NOW') * v_nom / 1e12
                full_wh = (num('CHARGE_FULL') or 0) * v_nom / 1e12
                design_wh = (num('CHARGE_FULL_DESIGN') or 0) * v_nom / 1e12
            else:
                now_wh = full_wh = design_wh = None
            status = ue.get('STATUS', '')
            left = None
            if watts and watts > 0.5 and now_wh is not None:
                if status == 'Discharging':
                    left = now_wh / watts * 3600
                elif status == 'Charging' and full_wh:
                    left = max(0.0, full_wh - now_wh) / watts * 3600
            bats.append({'name': os.path.basename(ps), 'capacity': num('CAPACITY'),
                         'status': status, 'watts': watts, 'time_left': left,
                         'health': full_wh / design_wh * 100 if full_wh and design_wh else None})
    return {'ac': ac, 'batteries': bats, 'peripherals': periph}


# ─────────────────────────────── processes ──────────────────────────────

class ProcessTable:
    """Top CPU consumers from /proc/<pid>/stat (one read per process, which is
    several times cheaper than psutil.process_iter on a busy box)."""

    def __init__(self, top=8):
        self.top = top
        self.clk = os.sysconf('SC_CLK_TCK')
        self.page = os.sysconf('SC_PAGE_SIZE')
        self.prev, self.prev_t = {}, None

    def _read_all(self):
        cur, states = {}, {}
        for e in os.scandir('/proc'):
            if not e.name.isdigit():
                continue
            s = read(f'/proc/{e.name}/stat')
            if not s:
                continue
            lp, rp = s.find('('), s.rfind(')')
            f = s[rp + 2:].split()
            if lp < 0 or rp < 0 or len(f) < 22:
                continue
            states[f[0]] = states.get(f[0], 0) + 1
            # fields (1-based): 3 state, 14 utime, 15 stime, 22 starttime, 24 rss
            cur[int(e.name)] = (s[lp + 1:rp], int(f[11]) + int(f[12]), int(f[21]) * self.page,
                                int(f[19]))
        return cur, states

    def scan(self):
        if self.prev_t is None:                 # prime so the first result has rates
            self.prev, _ = self._read_all()
            self.prev_t = time.monotonic()
            time.sleep(0.5)
        cur, states = self._read_all()
        now = time.monotonic()
        dt_s = now - self.prev_t
        rows = []
        for pid, (comm, ticks, rss, start) in cur.items():
            p = self.prev.get(pid)
            base = p[1] if p and p[3] == start else 0          # pid reuse => new process
            rows.append({'pid': pid, 'comm': comm, 'rss': rss,
                         'cpu': max(0, ticks - base) / self.clk / dt_s * 100})
        self.prev, self.prev_t = cur, now
        rows.sort(key=lambda r: -r['cpu'])
        top = rows[:self.top]
        for r in top:
            r['cmd'] = cmdline(r['pid'], r['comm'])
        return {'top': top, 'total': len(cur), 'running': states.get('R', 0),
                'blocked': states.get('D', 0), 'zombie': states.get('Z', 0)}


# ──────────────────────────── kernel events ─────────────────────────────

# Kernel messages that mean something is really wrong: NVIDIA GPU errors (Xid),
# kernel bugs, machine checks / PCIe AER errors, hung tasks, OOM kills, disk I/O
# errors and thermal emergencies.
KERNEL_EVENT_RE = ('NVRM: Xid|fallen off the bus|kernel BUG at|Oops:|BUG: |Hardware Error'
                   '|blocked for more than|Out of memory: Killed|I/O error, dev'
                   '|critical temperature|temperature above threshold')
# /proc/sys/kernel/tainted bits that signal trouble since boot (the O/E bits set
# by out-of-tree modules such as the NVIDIA driver are normal).
TAINT_FLAGS = [(4, 'M', 'machine check'), (5, 'B', 'bad page'), (7, 'D', 'oops/BUG'),
               (14, 'L', 'soft lockup')]


class KernelEvents:
    """Serious kernel-log messages of the current boot, read with journalctl every
    10 s (dmesg is root-only on most distros; the 'adm' / 'systemd-journal'
    groups may read the kernel journal)."""

    def __init__(self):
        self.events = deque(maxlen=20)        # (epoch s, message), oldest first
        self.count = 0
        self.last_ts = 0.0                    # newest event seen (dedupes overlapping polls)
        self.scanned_to = None                # wall time the previous poll started
        self.hint, self.poller = None, None
        if not shutil.which('journalctl'):
            return
        if not self._can_read():
            self.hint = ("kernel log not readable: add yourself to the 'adm' group "
                         "(sudo usermod -aG adm $USER, then log in again)")
            return
        self.poller = Poller(self._poll, 10.0, 'journal')

    @staticmethod
    def _can_read():
        if os.geteuid() == 0:
            return True
        names = set()
        for gid in os.getgroups():
            try:
                names.add(grp.getgrgid(gid).gr_name)
            except KeyError:
                pass
        return bool(names & {'adm', 'systemd-journal', 'wheel'})

    def _poll(self):
        started = time.time()
        cmd = ['journalctl', '-k', '-b', '-q', '--no-pager', '-o', 'json',
               '--output-fields=MESSAGE', '-g', KERNEL_EVENT_RE]
        if self.scanned_to:                   # only the new part (2 s overlap, deduped)
            cmd.append(f'--since=@{int(self.scanned_to) - 2}')
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        self.parse(p.stdout.splitlines())
        self.scanned_to = started

    def parse(self, lines):
        """Add journalctl JSON lines that are newer than anything seen so far."""
        for line in lines:
            try:
                j = json.loads(line)
                ts = int(j['__REALTIME_TIMESTAMP']) / 1e6
            except (ValueError, KeyError, TypeError):
                continue
            msg = j.get('MESSAGE')
            if isinstance(msg, list):         # journald stores non-UTF-8 text as a byte array
                msg = bytes(msg).decode('utf-8', 'replace')
            if ts <= self.last_ts or not isinstance(msg, str):
                continue
            self.events.append((ts, msg.strip()))
            self.count += 1
            self.last_ts = ts

    def sample(self):
        taint = read_int('/proc/sys/kernel/tainted') or 0
        return {'events': list(self.events), 'count': self.count, 'hint': self.hint,
                'taint': [(letter, what) for bit, letter, what in TAINT_FLAGS if taint >> bit & 1]}


# ──────────────────────────────── monitor ───────────────────────────────

class Monitor:
    def __init__(self, args):
        self.args = args
        self.sensors = Sensors()
        self.cpu = CPU()
        self.nvidia = NvidiaGPUs(args.interval)
        self.igpu = IntelGPU()
        self.kernel_events = KernelEvents()
        self.disks = Disks(want_smart=not args.no_smart)
        self.net = Network(show_all=args.all_nics, ping=args.ping, bits=args.bits)
        self.power = Poller(read_power_supplies, 5.0, 'power')   # battery/HID reads can be slow
        self.procs = Poller(ProcessTable().scan, 2.0, 'procs') if 'procs' in args.sections else None
        self.hostname = socket.gethostname()
        self.kernel = platform.release()
        self.boot = psutil.boot_time()

    def sample(self):
        t0 = time.monotonic()
        sens = self.sensors.sample()
        gpus, gpu_status = self.nvidia.sample()
        s = {
            'time': time.time(),
            'sensors': sens,
            'cpu': self.cpu.sample(sens),
            'mem': sample_memory(),
            'gpus': gpus, 'gpu_status': gpu_status,
            'igpus': self.igpu.sample(),
            'kernel': self.kernel_events.sample(),
            'disks': self.disks.sample(sens),
            'net': self.net.sample(),
            'power': self.power.value,
            'procs': self.procs.value if self.procs else None,
        }
        s['sample_ms'] = (time.monotonic() - t0) * 1000
        return s


# ─────────────────────────────── rendering ──────────────────────────────

def panel(body, title, width=None):
    return Panel(body, title=f'[bold]{title}[/]', title_align='left',
                 border_style='grey42', padding=(0, 1), width=width)


def gpu_short(name):
    """'NVIDIA GeForce RTX 5090 Laptop GPU' -> 'RTX 5090'"""
    return re.sub(r'( Laptop)?( GPU)?$', '', re.sub(r'^NVIDIA (GeForce )?', '', name))


def render_cpu(c, inner_w):
    lines = []
    head = Text(c['model'], style='bold', no_wrap=True, overflow='ellipsis')
    cores = f"{c['n']} threads" + (f" ({c['n_p']}P + {c['n_e']}E)" if c['n_p'] and c['n_e'] else '')
    head.append(f'  ·  {cores}', style='dim')
    if c['governor']:
        head.append(f"  ·  {c['governor']}" + (f"/{c['epp']}" if c['epp'] else ''), style='dim')
    lines.append(head)

    t = Text('Total ', style='bold')
    t.append(bar(c['total'], 24))
    t.append(f" {c['total']:5.1f}%  ", style=level_style(c['total'], 60, 85))
    t.append(sparkline(c['hist'], max(0, inner_w - 40), top=100, style='cyan'))
    lines.append(t)

    tm = c['times']
    lines.append(join([
        kv('load', ' '.join(f'{x:.2f}' for x in c['load']),
           style='yellow' if c['load'][0] > c['n'] else ''),
        kv('usr', f"{tm.get('user', 0):.1f}%"), kv('sys', f"{tm.get('system', 0):.1f}%"),
        kv('iowait', f"{tm.get('iowait', 0):.1f}%",
           style='yellow' if tm.get('iowait', 0) > 10 else ''),
        kv('irq', f"{tm.get('irq', 0) + tm.get('softirq', 0):.1f}%"),
        kv('nice', f"{tm.get('nice', 0):.1f}%"),
    ]))

    freq = f"{fmt_num(c['freq_avg'] and c['freq_avg'] / 1000, '.2f')} GHz avg, " \
           f"{fmt_num(c['freq_top'] and c['freq_top'] / 1000, '.2f')} top"
    if c['freq_max']:
        freq += f" (max {c['freq_max'] / 1000:.1f})"
    if c['power']:
        pw = kv('power', join([Text(f'{v:.1f} W', style='bold') + Text(f' {k}', style='dim')
                               for k, v in c['power'].items() if k != 'uncore'], '  '))
    elif c['power_present']:
        pw = kv('power', Text('needs sudo', style='dim'))
    else:
        pw = None
    lines.append(join([kv('freq', freq), pw], '   '))

    # Red from 2 °C below the throttle point (TjMax - TCC offset) when it is known.
    crit = c['temp_limit'] + 3 if c['temp_limit'] else None
    temp = join([temp_text(c['pkg_temp'], 'cpu', crit), Text('pkg', style='dim'),
                 temp_text(c['core_temp_max'], 'cpu', crit), Text('core max', style='dim')], ' ')
    if c['temp_limit']:
        at_limit = c['pkg_temp'] is not None and c['pkg_temp'] >= c['temp_limit'] - 1
        temp.append(f"  limit {c['temp_limit']:.0f}°C", style='bold red' if at_limit else 'dim')
    th = c['throttle']
    if th is None:
        throttle = None
    elif th['pkg_eps'] > 0 or th['core_eps'] > 0:
        throttle = Text(f"{th['pkg_eps']:.0f}/s pkg  {th['core_eps']:.0f}/s core", style='bold red')
        if th['pkg_time_pct'] >= 0.5:
            throttle.append(f"  {th['pkg_time_pct']:.0f}% of time", style='red')
    else:
        throttle = Text('none', style='green')
    lines.append(join([kv('temp', temp), kv('throttle', throttle) if throttle else None], '   '))
    if c['clamped']:
        lines.append(Text('busy cores forced below the OS minimum clock '
                          '(PROCHOT / TCC / EC power limit)', style='bold red'))

    ncols = max(1, min(6, (inner_w + 2) // 29))
    cores_list = c['cores']
    nrows = -(-len(cores_list) // ncols)
    grid = Table.grid(padding=(0, 2))
    for _ in range(ncols):
        grid.add_column(no_wrap=True)
    for r in range(nrows):
        cells = []
        for col in range(ncols):
            i = col * nrows + r
            if i >= len(cores_list):
                cells.append(Text(''))
                continue
            k = cores_list[i]
            cell = Text(f"{k['kind']}{k['cpu']}".ljust(4), style='bold' if k['kind'] == 'P' else 'dim')
            cell.append(bar(k['pct'], 8))
            cell.append(f" {fmt_num(k['pct'], '3.0f')}%", style=level_style(k['pct'], 60, 85))
            cell.append(f" {fmt_num(k['mhz'] and k['mhz'] / 1000, '.1f')}G", style='dim')
            if k['temp'] is None:
                cell.append('   –', style='dim')
            else:
                cell.append(f" {k['temp']:3.0f}°", style=temp_style(k['temp'], 'cpu', crit))
            cells.append(cell)
        grid.add_row(*cells)
    lines.append(grid)
    return panel(Group(*lines), 'CPU')


LIMIT_STYLES = {'hw-slowdown': 'bold red', 'hw-thermal': 'bold red', 'power-brake': 'bold red',
                'sw-thermal': 'red', 'power-cap': 'yellow'}


def render_gpu(gpus, igpus, status, inner_w):
    lines = []
    if status:
        lines.append(Text(status[0], style=status[1]))
    for g in gpus:
        head = Text(g['name'] if inner_w >= 70 else gpu_short(g['name']), style='bold green')
        if g.get('pstate'):
            head.append(f"  ·  {g['pstate']}", style='dim')
        if g.get('driver'):
            head.append(f"  ·  driver {g['driver']}", style='dim')
        lines.append(head)
        if g.get('asleep'):
            lines.append(Text('asleep (runtime D3) - not queried so it stays asleep', style='dim'))
            continue
        if g.get('error'):
            lines.append(Text(f"GPU not responding ({g['error']}) - see Alerts / kernel log; "
                              f"usually needs a reboot", style='bold red'))
            continue
        if g.get('recovery'):
            lines.append(Text(f"driver reports a fault: {g['recovery']}", style='bold red'))
        t = Text('Util ', style='bold')
        t.append(bar(g.get('util'), 20))
        t.append(f" {fmt_num(g.get('util'), '3.0f')}%  ", style=level_style(g.get('util'), 60, 85))
        t.append(sparkline(g['hist'], max(0, inner_w - 33), top=100, style='green'))
        lines.append(t)
        vt, vu = g.get('vram_total'), g.get('vram_used')
        vpct = vu / vt * 100 if vt and vu is not None else None
        t = Text('VRAM ', style='bold')
        t.append(bar(vpct, 20, 75, 92))
        t.append(f" {fmt_num(vpct, '3.0f')}%  {fmt_bytes(vu)} / {fmt_bytes(vt)}")
        t.append(f"  mem-bus {fmt_num(g.get('mem_util'), '.0f', '%')}", style='dim')
        lines.append(t)
        pw = f"{fmt_num(g.get('power'), '.1f')} W"
        if g.get('power_limit'):
            pw += f" / {g['power_limit']:.0f} W"
        slow = f" (slowdown {g['slowdown']}°C)" if g.get('slowdown') else ''
        lines.append(join([
            kv('temp', temp_text(g.get('temp'), 'gpu', g.get('slowdown')) + Text(slow, style='dim')),
            kv('power', pw),
            kv('fan', f"{g['fan']}%") if g.get('fan') is not None else None,   # n/a on most laptops
        ], '   '))
        lines.append(join([
            kv('clock', f"{fmt_num(g.get('clk'))}/{fmt_num(g.get('max_clk'))} MHz"),
            kv('mem', f"{fmt_num(g.get('mem_clk'))} MHz"),
            kv('enc', fmt_num(g.get('enc'), '.0f', '%')),
            kv('dec', fmt_num(g.get('dec'), '.0f', '%')),
        ], '   '))
        link = f"Gen{g['pcie_gen']} x{g['pcie_width']}" if g.get('pcie_gen') else '–'
        lines.append(join([
            kv('PCIe', link),
            kv('rx', fmt_bytes(g.get('pcie_rx'), rate=True)),
            kv('tx', fmt_bytes(g.get('pcie_tx'), rate=True)),
        ], '   '))
        limits = Text('clock limits ', style='dim')
        for i, name in enumerate(g.get('throttle') or ['none']):
            limits.append(', ' if i else '', style='dim')
            limits.append(name, style=LIMIT_STYLES.get(name, 'dim'))
        lines.append(limits)
        procs = g.get('procs') or []
        if procs:
            tbl = Table(box=None, padding=(0, 1), collapse_padding=True, pad_edge=False,
                        show_header=False, expand=True)
            tbl.add_column(justify='right', style='dim', no_wrap=True)
            tbl.add_column(justify='right', no_wrap=True)
            tbl.add_column(style='dim', no_wrap=True)
            tbl.add_column(no_wrap=True, overflow='ellipsis', ratio=1)
            for p in procs[:5]:
                tbl.add_row(str(p['pid']), fmt_bytes(p['mem']), p['kind'], Text(p['cmd']))
            if len(procs) > 5:
                tbl.add_row('', '', '', Text(f'… {len(procs) - 5} more', style='dim'))
            lines.append(tbl)
    for ig in igpus:
        engines = [kv(f'{label}≈', fmt_num(b, '.0f', '%'), style=level_style(b, 60, 85))
                   for label, b in ig['engines']]
        lines.append(join([
            Text(f"iGPU ({ig['driver']})", style='bold blue'),
            f"{fmt_num(ig['freq'])}/{fmt_num(ig['max'])} MHz",
            *engines,
            kv('limits', ', '.join(ig['limits']), style='yellow') if ig['limits'] else None,
        ], '  '))
    if not lines:
        lines.append(Text('no GPU found', style='dim'))
    return panel(Group(*lines), 'GPU')


def render_mem(m):
    ram = Text('RAM  ', style='bold')
    ram.append(bar(m['pct'], 24, 75, 90))
    ram.append(f" {m['pct']:3.0f}%  {fmt_bytes(m['used'])} / {fmt_bytes(m['total'])}")
    swap = Text('Swap ', style='bold')
    swap.append(bar(m['swap_pct'], 24, 25, 60))
    swap.append(f" {m['swap_pct']:3.0f}%  {fmt_bytes(m['swap_used'])} / {fmt_bytes(m['swap_total'])}")
    extra = [kv('avail', fmt_bytes(m['avail'])), kv('cache', fmt_bytes(m['cache']))]
    if m['dirty'] is not None:
        # Dirty pages waiting to be written back: grows when writes outpace slow (USB) disks.
        extra.append(kv('dirty', fmt_bytes(m['dirty']), style='yellow' if m['dirty'] > 1 << 30 else ''))
    if m['writeback']:
        extra.append(kv('writeback', fmt_bytes(m['writeback']), style='yellow'))
    return panel(Group(ram, swap, join(extra, '  ')), 'Memory')


def short_mount(path):
    """'/media/<user>/Backup' -> 'Backup' (removable drives); other mounts unchanged."""
    return re.sub(r'^(/run)?/media/[^/]+/', '', path) or path


def render_disks(d, inner_w):
    wide = inner_w >= 120                     # room for the model name
    compact = inner_w < 90                    # drop Type and IOPS
    tbl = Table(box=None, padding=(0, 1), collapse_padding=True, pad_edge=False,
                header_style='bold dim', expand=True)
    tbl.add_column('Disk', no_wrap=True)
    if wide:
        tbl.add_column('Model', no_wrap=True, overflow='ellipsis', ratio=1, style='dim')
    if not compact:
        tbl.add_column('Type', no_wrap=True, style='dim')
    tbl.add_column('Temp', justify='right', no_wrap=True)
    tbl.add_column('Read', justify='right', no_wrap=True)
    tbl.add_column('Write', justify='right', no_wrap=True)
    if not compact:
        tbl.add_column('IOPS', justify='right', no_wrap=True, style='dim')
    tbl.add_column('Busy', no_wrap=True)
    tbl.add_column('Mount', no_wrap=True, overflow='ellipsis', ratio=1, min_width=6)
    tbl.add_column('Used', no_wrap=True)
    tbl.add_column('Free', justify='right', no_wrap=True, style='dim')
    n_disk_cols = len(tbl.columns) - 3
    for r in d['disks']:
        kind = f"{r['bus']} {'HDD' if r['rota'] else 'SSD'}" if r['bus'] != 'NVMe' else 'NVMe'
        if r['temp'] == 'standby':
            temp = Text('zz', style='dim')
        else:
            temp = temp_text(r['temp'], 'hdd' if r['rota'] else 'nvme')
        busy = bar(r['busy'], 5, 50, 90)
        busy.append(f" {fmt_num(r['busy'], '3.0f')}%", style=level_style(r['busy'], 50, 90))
        mounts = r['mounts'] or [None]
        for i, mnt in enumerate(mounts):
            if mnt:
                used = bar(mnt['pct'], 6, 80, 90)
                used.append(f" {mnt['pct']:3.0f}% of {fmt_bytes(mnt['total'])}",
                            style=level_style(mnt['pct'], 80, 90))
                mcells = [Text(short_mount(mnt['mount'])), used, fmt_bytes(mnt['free'])]
            else:
                mcells = [Text('–', style='dim'), Text(f"size {fmt_bytes(r['size'])}", style='dim'), '']
            if i == 0:
                cells = [Text(r['name'], style='bold')]
                if wide:
                    cells.append(Text(r['model']))
                if not compact:
                    cells.append(kind)
                cells += [temp, fmt_bytes(r['read'], True), fmt_bytes(r['write'], True)]
                if not compact:
                    cells.append(fmt_num(r['iops']))
                cells.append(busy)
            else:
                cells = [''] * n_disk_cols
            tbl.add_row(*cells, *mcells)
    body = [tbl]
    for key in ('usage_hint', 'smart_hint'):
        if d.get(key):
            body.append(Text(d[key], style='yellow' if key == 'usage_hint' else 'dim italic'))
    return panel(Group(*body), 'Disks')


def render_net(n, inner_w):
    rate = fmt_bits if n['bits'] else (lambda v: fmt_bytes(v, True))
    tbl = Table(box=None, padding=(0, 1), pad_edge=False, header_style='bold dim')
    tbl.add_column('Iface', no_wrap=True)
    tbl.add_column('Address', no_wrap=True, style='dim')
    tbl.add_column('Link', no_wrap=True, overflow='ellipsis', max_width=30)
    tbl.add_column('↓ Rx', justify='right', style='cyan')
    tbl.add_column('↑ Tx', justify='right', style='magenta')
    spark_w = max(0, min(30, inner_w - 100))
    if spark_w >= 6:
        tbl.add_column('Activity', no_wrap=True)
    tbl.add_column('Total ↓/↑', justify='right', style='dim', no_wrap=True)
    for r in n['nics']:
        w = r['wifi']
        if w:
            link = Text(f"{w['ssid']} ")
            link.append(f"{fmt_num(w['signal'])}%", style=level_style(100 - (w['signal'] or 0), 50, 70))
            link.append(f" {w['rate']} ch{w['chan']}", style='dim')
        elif r['speed']:
            link = Text(f"{r['speed']} Mb/s" if r['speed'] < 1000 else f"{r['speed'] / 1000:g} Gb/s")
        else:
            link = Text('Wi-Fi' if r['wireless'] else ('up' if r['up'] else 'down'), style='dim')
        if r['errs']:
            link.append(f"  {r['errs']} err/drop", style='yellow')
        cells = [Text(r['name'], style='bold'), r['addr'] or '–', link, rate(r['rx']), rate(r['tx'])]
        if spark_w >= 6:
            cells.append(sparkline(r['hist'], spark_w, style='blue'))
        cells.append(f"{fmt_bytes(r['rx_total'])} / {fmt_bytes(r['tx_total'])}")
        tbl.add_row(*cells)
    body = [tbl]
    foot = []
    if n.get('latency'):
        lat = n['latency']
        foot.append(kv(f"internet ({lat['host']})",
                       f"{lat['ms']:.0f} ms" if lat['ms'] is not None else 'unreachable',
                       style='green' if lat['ms'] is not None and lat['ms'] < 80 else
                       'yellow' if lat['ms'] is not None else 'bold red'))
    if n['hidden']:
        foot.append(Text(f"down: {', '.join(n['hidden'])}", style='dim'))
    if foot:
        body.append(join(foot, '   '))
    return panel(Group(*body), 'Network')


def collect_temps(s):
    """Every temperature on the box as (label, °C, kind, crit)."""
    out = []
    c = s['cpu']
    if c['pkg_temp'] is not None:
        out.append(('CPU package', c['pkg_temp'], 'cpu', None))
    if c['core_temp_max'] is not None:
        out.append(('CPU cores max', c['core_temp_max'], 'cpu', None))
        out.append(('CPU cores avg', c['core_temp_avg'], 'cpu', None))
    for g in s['gpus']:
        if g.get('temp') is not None:
            out.append((f"GPU{g['index']} {gpu_short(g['name'])}"[:22], g['temp'], 'gpu', g.get('slowdown')))
    for ch in s['sensors']['chips']:
        name = ch['name']
        if name in ('coretemp', 'k10temp', 'zenpower'):
            continue
        for label, t, crit in ch['temps']:
            if name == 'nvme':
                dev = ch['blocks'][0] if ch['blocks'] else 'NVMe'
                out.append((f'{dev} {label}', t, 'nvme', crit))
            elif name == 'drivetemp':
                out.append((f"{ch['blocks'][0] if ch['blocks'] else 'drive'}", t, 'hdd', crit))
            elif name == 'acpitz':
                out.append(('ACPI zone', t, 'other', crit))
            elif name.startswith('iwlwifi'):
                out.append(('Wi-Fi card', t, 'other', crit))
            else:
                out.append((f'{name} {label}', t, 'other', crit))
    for r in s['disks']['disks']:
        if r['bus'] != 'NVMe' and isinstance(r['temp'], float):
            out.append((f"{r['name']} HDD", r['temp'], 'hdd', None))
    for typ, t in s['sensors']['zones']:
        out.append((typ, t, 'cpu' if typ.startswith('TCPU') else 'other', None))
    return out


def render_sensors(s, inner_w):
    temps = collect_temps(s)
    cell_w = 24
    ncols = max(1, (inner_w + 2) // (cell_w + 2))
    nrows = -(-len(temps) // ncols) if temps else 0
    grid = Table.grid(padding=(0, 2))
    for _ in range(ncols):
        grid.add_column(no_wrap=True)
    for r in range(nrows):
        cells = []
        for col in range(ncols):
            i = col * nrows + r
            if i < len(temps):
                label, t, kind, crit = temps[i]
                cell = Text(label[:cell_w - 7].ljust(cell_w - 6), style='dim')
                cell.append(f'{t:4.0f}°C', style=temp_style(t, kind, crit))
                cells.append(cell)
            else:
                cells.append(Text(''))
        grid.add_row(*cells)
    body = [Text('Temperatures', style='bold'), grid]

    fans = list(s['sensors']['fans'])
    for g in s['gpus']:
        if g.get('fan') is not None:
            fans.append({'name': f"GPU{g['index']} fan", 'pct': g['fan']})
    if fans:
        body.append(Text('Fans', style='bold'))
        fgrid = Table.grid(padding=(0, 2))
        fgrid.add_column(style='dim')
        fgrid.add_column(justify='right')
        fgrid.add_column()
        for f in fans:
            if 'pct' in f:
                fgrid.add_row(f['name'], f"{f['pct']}%", '')
            else:
                rpm = f['rpm']
                val = Text('stopped', style='dim') if rpm == 0 else Text(f'{fmt_num(rpm)} RPM', style='cyan')
                # Auto-range over the visible window, but never tighter than 1000 RPM,
                # so ramps show up without magnifying a few RPM of jitter.
                w = max(0, min(40, inner_w - 30))
                recent = list(f['hist'])[-w:] if w else []
                lo, hi = (min(recent), max(recent)) if recent else (0, 0)
                if hi - lo < 1000:
                    lo = max(0, (hi + lo) / 2 - 500)
                    hi = lo + 1000
                fgrid.add_row(f['name'], val, sparkline(recent, w, top=hi, lo=lo, style='cyan'))
        body.append(fgrid)
    return panel(Group(*body), 'Sensors')


def render_power(s):
    p = s['power']
    lines = []
    if p is None:
        return panel(Text('reading…', style='dim'), 'Power')
    parts = []
    if p['ac'] is not None:
        parts.append(kv('AC', Text('● online', style='green') if p['ac'] else Text('○ on battery', style='yellow')))
    for b in p['batteries']:
        t = Text()
        t.append(bar(b['capacity'], 10, 101, 101))    # never red: full battery is good
        t.append(f" {fmt_num(b['capacity'])}%  {b['status'].lower()}")
        if b['watts'] is not None and b['watts'] > 0.05:
            t.append(f"  {b['watts']:.1f} W", style='bold')
        if b['time_left']:
            t.append(f"  {fmt_duration(b['time_left'])} {'left' if b['status'] == 'Discharging' else 'to full'}",
                     style='dim')
        if b['health']:
            t.append(f"  health {b['health']:.0f}%", style='dim')
        parts.append(kv(b['name'], t))
    lines.append(join(parts, '   '))
    draws = []
    cpu_pw = s['cpu']['power']
    for k, v in cpu_pw.items():
        draws.append(kv(f'CPU {k}' if k not in ('platform',) else 'platform', f'{v:.1f} W'))
    for g in s['gpus']:
        if g.get('power') is not None:
            draws.append(kv(f"GPU{g['index']}", f"{g['power']:.1f} W"))
    if not cpu_pw and s['cpu']['power_present']:
        draws.append(Text('(CPU power: run with sudo)', style='dim'))
    if draws:
        lines.append(join(draws, '   '))
    for d in p['peripherals']:
        lvl = f"{d['capacity']}%" if d['capacity'] is not None else (d['level'] or '?').lower()
        lines.append(kv(d['name'], f"{lvl}  {(d['status'] or '').lower()}"))
    return panel(Group(*lines), 'Power')


def render_procs(pr, inner_w):
    if pr is None:
        return panel(Text('reading…', style='dim'), 'Processes')
    head = join([kv('total', pr['total']), kv('running', pr['running']),
                 kv('blocked (D)', pr['blocked'], style='yellow' if pr['blocked'] else ''),
                 kv('zombie', pr['zombie'], style='yellow' if pr['zombie'] else '')], '   ')
    tbl = Table(box=None, padding=(0, 1), collapse_padding=True, pad_edge=False,
                header_style='bold dim', expand=True)
    tbl.add_column('PID', justify='right', style='dim', no_wrap=True)
    tbl.add_column('CPU%', justify='right', no_wrap=True)
    tbl.add_column('RSS', justify='right', style='dim', no_wrap=True)
    tbl.add_column('Command', no_wrap=True, overflow='ellipsis', ratio=1)
    for r in pr['top']:
        tbl.add_row(str(r['pid']), Text(f"{r['cpu']:.0f}", style=level_style(r['cpu'], 50, 95)),
                    fmt_bytes(r['rss']), Text(r['cmd']))
    return panel(Group(head, tbl), 'Processes')


def render_alerts(k):
    """Serious kernel events of this boot; None (no panel) while there are none."""
    if not (k['events'] or k['taint'] or k['hint']):
        return None
    lines = []
    head = []
    if k['count']:
        head.append(Text(f"{k['count']} serious kernel message{'s' if k['count'] != 1 else ''} "
                         f"this boot (journalctl -k -b)", style='bold red'))
    if k['taint']:
        head.append(kv('kernel tainted', ', '.join(f'{letter} ({what})' for letter, what in k['taint']),
                       style='bold red'))
    if head:
        lines.append(join(head, '   '))
    now = time.time()
    for ts, msg in k['events'][-3:]:
        t = Text(f'{dt.datetime.fromtimestamp(ts):%m-%d %H:%M:%S}  ', style='dim',
                 no_wrap=True, overflow='ellipsis')
        t.append(msg, style='bold red' if now - ts < 900 else 'yellow')   # red for 15 min
        lines.append(t)
    if k['hint']:
        lines.append(Text(k['hint'], style='dim'))
    return panel(Group(*lines), 'Alerts')


def render_header(mon, s, width):
    """Title line. Parts are dropped, least useful first, so it never wraps."""
    parts = [                                  # (text, style, keep priority: lower = keep longer)
        (mon.hostname, 'dim', 1),
        (mon.kernel, 'dim', 4),
        (f"up {fmt_duration(s['time'] - mon.boot)}", 'dim', 2),
        (f"{dt.datetime.fromtimestamp(s['time']):%Y-%m-%d %H:%M:%S}", 'dim', 0),
        (f"every {mon.args.interval:g}s", 'dim', 3),
        (f"sample {s['sample_ms']:.0f} ms", 'dim', 5),
        (f'logging → {mon.args.log}', 'green', 0) if mon.args.log else None,
        ('Ctrl+C to quit', 'dim', 6),
    ]
    parts = [p for p in parts if p]
    title = ' hw_monitor '
    keep = sorted(parts, key=lambda p: p[2])
    while keep and len(title) + sum(len(p[0]) + 5 for p in keep) > width:
        keep.pop()                             # drop the least important part left
    t = Text(title, style='bold reverse', no_wrap=True, overflow='ellipsis')
    for i, (txt, style, _) in enumerate(p for p in parts if p in keep):
        t.append('  ' if i == 0 else '  ·  ', style='dim')
        t.append(txt, style=style)
    return t


def render(mon, s, width):
    sec = mon.args.sections
    two_col = width >= 150
    left_w = int(width * 0.6) if two_col else width
    right_w = width - left_w if two_col else width
    inner_l, inner_r = left_w - 4, right_w - 4
    P = {}
    if 'cpu' in sec:
        P['cpu'] = render_cpu(s['cpu'], inner_l)
    if 'gpu' in sec:
        P['gpu'] = render_gpu(s['gpus'], s['igpus'], s['gpu_status'], inner_r)
    if 'mem' in sec:
        P['mem'] = render_mem(s['mem'])
    if 'disk' in sec:
        P['disk'] = render_disks(s['disks'], inner_l)
    if 'net' in sec:
        P['net'] = render_net(s['net'], inner_l)
    if 'sensors' in sec:
        P['sensors'] = render_sensors(s, inner_l)
    if 'power' in sec:
        P['power'] = render_power(s)
    if 'procs' in sec:
        P['procs'] = render_procs(s['procs'], inner_r)
    top = [render_header(mon, s, width), render_alerts(s['kernel'])]
    top = [x for x in top if x is not None]
    if not two_col:
        order = ('cpu', 'gpu', 'mem', 'disk', 'net', 'sensors', 'power', 'procs')
        return Group(*top, *[P[k] for k in order if k in P])
    left = [P[k] for k in ('cpu', 'disk', 'net', 'sensors') if k in P]
    right = [P[k] for k in ('gpu', 'mem', 'power', 'procs') if k in P]
    grid = Table.grid(expand=True)
    grid.add_column(width=left_w)
    grid.add_column(width=right_w)
    grid.add_row(Group(*left), Group(*right))
    return Group(*top, grid)


def text_line(s):
    """Compact single-line summary for --no-ui."""
    c, m = s['cpu'], s['mem']
    parts = [f"{dt.datetime.fromtimestamp(s['time']):%H:%M:%S}",
             f"cpu {c['total']:4.1f}% {fmt_num(c['pkg_temp'], '.0f', 'C')} "
             f"{fmt_num(c['freq_avg'] and c['freq_avg'] / 1000, '.1f', 'GHz')}"
             + (f" {c['power']['package']:.0f}W" if 'package' in c['power'] else ''),
             f"mem {m['pct']:.0f}%"]
    for g in s['gpus']:
        if g.get('asleep'):
            parts.append(f"gpu{g['index']} asleep")
        elif g.get('error'):
            parts.append(f"gpu{g['index']} ERROR {g['error']}")
        else:
            parts.append(f"gpu{g['index']} {fmt_num(g.get('util'))}% {fmt_num(g.get('temp'))}C "
                         f"{fmt_num(g.get('power'), '.0f')}W {fmt_bytes(g.get('vram_used'))}")
    if s['gpu_status'] and s['gpu_status'][1] != 'dim':
        parts.append(s['gpu_status'][0])
    rd = sum(r['read'] or 0 for r in s['disks']['disks'])
    wr = sum(r['write'] or 0 for r in s['disks']['disks'])
    parts.append(f"disk r {fmt_bytes(rd, True)} w {fmt_bytes(wr, True)}")
    rx = sum(r['rx'] or 0 for r in s['net']['nics'])
    tx = sum(r['tx'] or 0 for r in s['net']['nics'])
    parts.append(f"net ↓{fmt_bytes(rx, True)} ↑{fmt_bytes(tx, True)}")
    fans = [str(f['rpm']) for f in s['sensors']['fans'] if f['rpm'] is not None]
    if fans:
        parts.append(f"fan {'/'.join(fans)}")
    if s['kernel']['count']:
        parts.append(f"kernel events {s['kernel']['count']}")
    return ' | '.join(parts)


# ─────────────────────────────── CSV logging ────────────────────────────

def flatten(s):
    """One flat {column: value} row per sample for the CSV log."""
    r = {'time': dt.datetime.fromtimestamp(s['time']).isoformat(timespec='seconds'),
         'epoch': round(s['time'], 3)}

    def put(key, v, nd=1):
        r[key] = '' if v is None else (round(v, nd) if isinstance(v, float) else v)

    c = s['cpu']
    put('cpu_pct', c['total'])
    for k in ('user', 'nice', 'system', 'iowait'):
        put(f'cpu_{k}_pct', c['times'].get(k))
    put('load1', c['load'][0], 2)
    put('cpu_freq_avg_mhz', c['freq_avg'], 0)
    put('cpu_pkg_temp_c', c['pkg_temp'])
    put('cpu_core_temp_max_c', c['core_temp_max'])
    put('cpu_temp_limit_c', c['temp_limit'])
    for k, v in c['power'].items():
        put(f'cpu_power_{k}_w', v, 2)
    th = c['throttle'] or {}
    put('cpu_throttle_pkg_events_ps', th.get('pkg_eps'))
    put('cpu_throttle_core_events_ps', th.get('core_eps'))
    put('cpu_throttle_pkg_time_pct', th.get('pkg_time_pct'))
    put('cpu_clamped', int(c['clamped']))
    for k in c['cores']:
        put(f"cpu{k['cpu']}_pct", k['pct'])
        put(f"cpu{k['cpu']}_temp_c", k['temp'])
    m = s['mem']
    put('mem_used_pct', m['pct'])
    put('mem_used_gib', m['used'] / 2 ** 30, 2)
    put('swap_used_pct', m['swap_pct'])
    put('mem_dirty_mib', m['dirty'] / 2 ** 20 if m['dirty'] is not None else None, 1)
    put('gpu_status', s['gpu_status'][0] if s['gpu_status'] else None)
    for g in s['gpus']:
        i = f"gpu{g['index']}"
        put(f'{i}_error', g.get('error') or g.get('recovery'))
        put(f'{i}_util_pct', g.get('util'))
        put(f'{i}_mem_util_pct', g.get('mem_util'))
        put(f'{i}_vram_used_mib', g['vram_used'] / 2 ** 20 if g.get('vram_used') is not None else None, 0)
        put(f'{i}_temp_c', g.get('temp'))
        put(f'{i}_power_w', g.get('power'), 2)
        put(f'{i}_clock_mhz', g.get('clk'))
        put(f'{i}_mem_clock_mhz', g.get('mem_clk'))
        put(f'{i}_fan_pct', g.get('fan'))
        put(f'{i}_pcie_rx_Bps', g.get('pcie_rx'), 0)
        put(f'{i}_pcie_tx_Bps', g.get('pcie_tx'), 0)
        put(f'{i}_pstate', g.get('pstate'))
        put(f'{i}_limits', '+'.join(g.get('throttle') or []) if 'throttle' in g else None)
    for ig in s['igpus']:
        put(f"igpu_{ig['card']}_freq_mhz", ig['freq'])
        for label, b in ig['engines']:
            put(f"igpu_{ig['card']}_{label}_busy_pct", b)
    for d in s['disks']['disks']:
        n = d['name']
        put(f'disk_{n}_read_Bps', d['read'], 0)
        put(f'disk_{n}_write_Bps', d['write'], 0)
        put(f'disk_{n}_iops', d['iops'], 1)
        put(f'disk_{n}_busy_pct', d['busy'])
        put(f'disk_{n}_temp_c', d['temp'] if isinstance(d['temp'], float) else None)
        for mnt in d['mounts']:
            put(f"fs_{slug(mnt['mount']) or 'root'}_used_pct", mnt['pct'])
    for nic in s['net']['nics']:
        put(f"net_{nic['name']}_rx_Bps", nic['rx'], 0)
        put(f"net_{nic['name']}_tx_Bps", nic['tx'], 0)
    if s['net'].get('latency'):
        put('net_latency_ms', s['net']['latency']['ms'])
    for label, t, _, _ in collect_temps(s):
        put(f'temp_{slug(label)}_c', t)
    for f in s['sensors']['fans']:
        put(f"fan_{slug(f['name'])}_rpm", f['rpm'])
    p = s['power']
    if p:
        put('ac_online', int(p['ac']) if p['ac'] is not None else None)
        for b in p['batteries']:
            put(f"{slug(b['name'])}_pct", b['capacity'])
            put(f"{slug(b['name'])}_power_w", b['watts'], 2)
    k = s['kernel']
    put('kernel_events', k['count'])
    put('kernel_taint', ''.join(letter for letter, _ in k['taint']))
    return r


class CSVLogger:
    """Appends one row per sample. The column set is fixed by the file's existing
    header (or the first row); columns that appear later are dropped, missing
    ones are left empty.

    Rows are fsync'ed every SYNC_EVERY seconds: plain buffered writes can sit in
    the page cache for ~30 s, and after a hard crash those last seconds are
    exactly the ones worth having."""

    SYNC_EVERY = 5.0

    def __init__(self, path):
        self.path = os.path.expanduser(path)
        self.f = self.w = None
        self.synced = 0.0

    def write(self, row):
        if self.f is None:
            fields = None
            if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
                with open(self.path, newline='') as f:
                    fields = next(csv.reader(f), None)
            new = not fields
            self.f = open(self.path, 'a', newline='')
            self.w = csv.DictWriter(self.f, fieldnames=fields or list(row), extrasaction='ignore', restval='')
            if new:
                self.w.writeheader()
        self.w.writerow(row)
        self.f.flush()
        now = time.monotonic()
        if now - self.synced >= self.SYNC_EVERY:
            os.fsync(self.f.fileno())
            self.synced = now

    def close(self):
        if self.f:
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()
            self.f = None


# ───────────────────────────────── main ─────────────────────────────────

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description='Live hardware monitor: CPU, GPU, memory, disks, network, temperatures, fans, power.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='CPU power (RAPL) and HDD temperatures (smartctl) need sudo.')
    ap.add_argument('-V', '--version', action='version', version=f'%(prog)s {__version__}')
    ap.add_argument('-i', '--interval', type=float, default=1.0,
                    help='seconds between updates (default 1.0; 2-5 is plenty for long logs)')
    ap.add_argument('--log', metavar='FILE.csv', help='append one CSV row per update to FILE')
    ap.add_argument('--no-ui', action='store_true',
                    help='no dashboard: print one summary line per update (quiet with --log)')
    ap.add_argument('--once', action='store_true', help='print a single snapshot and exit')
    ap.add_argument('--sections', default=','.join(SECTIONS),
                    help=f'comma-separated panels to show (default: all = {",".join(SECTIONS)})')
    ap.add_argument('--all-nics', action='store_true', help='also list network interfaces that are down')
    ap.add_argument('--bits', action='store_true', help='show network rates in bit/s instead of byte/s')
    ap.add_argument('--ping', nargs='?', const='1.1.1.1', metavar='HOST',
                    help='measure internet latency (TCP handshake to HOST:443, default 1.1.1.1) every 5 s')
    ap.add_argument('--no-smart', action='store_true', help='never call smartctl for HDD temperatures')
    args = ap.parse_args(argv)
    if args.interval < 0.2:
        ap.error('--interval must be at least 0.2 s')
    args.sections = {x.strip() for x in args.sections.split(',') if x.strip()}
    unknown = args.sections - set(SECTIONS)
    if unknown:
        ap.error(f"unknown section(s): {', '.join(sorted(unknown))}; choose from {', '.join(SECTIONS)}")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not sys.platform.startswith('linux'):
        sys.exit('hw_monitor reads /proc and /sys, so it only runs on Linux.')

    def on_term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:   # keep nohup's "ignore hangups"
        signal.signal(signal.SIGHUP, on_term)

    console = Console()
    logger = CSVLogger(args.log) if args.log else None
    try:
        mon = Monitor(args)
        if logger or args.once or args.no_ui:
            # The CSV columns are fixed by the first row, so let NVML report first
            # (the live dashboard instead shows "NVML starting…" for a moment).
            mon.nvidia.wait_ready(5.0)
        # First rates need two samples; give background pollers a moment too.
        mon.sample()
        time.sleep(min(args.interval, 0.5) if not args.once else args.interval)

        if args.once:
            s = mon.sample()
            if logger:
                logger.write(flatten(s))
            if args.no_ui:
                print(text_line(s))
            else:
                console.print(render(mon, s, console.width))
            return
        next_t = time.monotonic()
        if args.no_ui:
            if logger:
                print(f'hw_monitor: logging every {args.interval:g}s to {logger.path} (Ctrl+C to stop)',
                      file=sys.stderr)
            while True:
                s = mon.sample()
                if logger:
                    logger.write(flatten(s))
                else:
                    print(text_line(s), flush=True)
                next_t += args.interval
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.monotonic()
        with Live(console=console, screen=True, auto_refresh=False) as live:
            while True:
                s = mon.sample()
                if logger:
                    logger.write(flatten(s))
                live.update(render(mon, s, console.width), refresh=True)
                next_t += args.interval
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.monotonic()       # fell behind (e.g. suspend): resync
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:                   # e.g. `hw_monitor.py --no-ui | head`
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    finally:
        if logger:
            logger.close()


if __name__ == '__main__':
    main()
