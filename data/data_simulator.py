from __future__ import annotations

import csv
import heapq
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal


@dataclass(frozen=True)
class JobRecord:
	job_id: int
	population: str
	arrival_t: float
	accepted: bool
	refused_reason: str

	s1_start_t: float | None
	s1_end_t: float | None
	s2_start_t: float | None
	s2_end_t: float | None

	stage2_refused: bool
	backed_up: bool
	lost_result: bool

	def sojourn_time(self) -> float | None:
		if not self.accepted:
			return None
		if self.lost_result:
			return None
		if self.s2_end_t is None:
			return None
		return self.s2_end_t - self.arrival_t


@dataclass(frozen=True)
class RunMetrics:
	seed: int
	horizon: float

	arrivals: int
	accepted: int
	refused_stage1: int
	stage2_refused: int

	blank_pages: int
	permanent_blanks: int

	completed: int
	mean_sojourn: float | None
	var_sojourn: float | None

	def to_dict(self) -> dict:
		return asdict(self)


def _expovariate(rng: random.Random, rate: float) -> float:
	if rate <= 0:
		raise ValueError("rate must be > 0")
	return rng.expovariate(rate)


def mmck_stationary_probabilities(
	arrival_rate: float, service_rate: float, servers: int, system_capacity: int
) -> list[float]:
	"""Stationary probabilities for an M/M/c/K birth-death chain.

	- `servers` is c.
	- `system_capacity` is K, the maximum number of jobs *in system* (in service + waiting).

	Returns pi[0..K].
	"""
	if arrival_rate < 0 or service_rate <= 0:
		raise ValueError("invalid rates")
	if servers <= 0:
		raise ValueError("servers must be >= 1")
	if system_capacity < 0:
		raise ValueError("system_capacity must be >= 0")
	if system_capacity == 0:
		return [1.0]

	c = servers
	K = system_capacity
	a = arrival_rate / service_rate

	terms: list[float] = []
	for n in range(0, K + 1):
		if n <= c:
			term = (a**n) / math.factorial(n)
		else:
			term = (a**n) / (math.factorial(c) * (c ** (n - c)))
		terms.append(term)
	z = sum(terms)
	return [t / z for t in terms]


def mmck_blocking_probability(
	arrival_rate: float, service_rate: float, servers: int, system_capacity: int
) -> float:
	pi = mmck_stationary_probabilities(arrival_rate, service_rate, servers, system_capacity)
	return pi[-1]


def summarise_sojourn(times: Iterable[float]) -> tuple[float | None, float | None, int]:
	vals = [t for t in times if t is not None and math.isfinite(t)]
	if not vals:
		return None, None, 0
	if len(vals) == 1:
		return vals[0], 0.0, 1
	return statistics.fmean(vals), statistics.pvariance(vals), len(vals)


def write_job_records_csv(path: str | Path, jobs: list[JobRecord]) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=list(asdict(jobs[0]).keys()) if jobs else [])
		if jobs:
			writer.writeheader()
			for job in jobs:
				writer.writerow(asdict(job))


def write_metrics_csv(path: str | Path, rows: list[RunMetrics]) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=list(rows[0].to_dict().keys()) if rows else [])
		if rows:
			writer.writeheader()
			for r in rows:
				writer.writerow(r.to_dict())


EventType = Literal[
	"arrival",
	"s1_done",
	"s2_done",
]


BackupStrategy = Literal["none", "random", "systematic"]


@dataclass(order=True)
class _Event:
	t: float
	seq: int
	kind: EventType
	job_id: int


def simulate_waterfall(
	*,
	seed: int,
	horizon: float,
	arrival_rate: float,
	exec_servers: int,
	exec_service_rate: float,
	send_service_rate: float,
	ks: int | None = None,
	kf: int | None = None,
	backup_probability: float = 0.0,
	backup_strategy: BackupStrategy | None = None,
) -> tuple[RunMetrics, list[JobRecord]]:
	"""Two-stage waterfall (execution then send).

	Model (default):
	- External arrivals: Poisson(arrival_rate)
	- Stage 1 service: Exp(exec_service_rate) with `exec_servers` parallel servers
	- Stage 2 service: Exp(send_service_rate) with a single server

	Capacities:
	- `ks`: max waiting room for stage 1 (None => infinite). Total stage-1 system capacity = exec_servers + ks.
	- `kf`: max waiting room for stage 2 (None => infinite). Total stage-2 system capacity = 1 + kf.

	Backup strategies:
	- systematic: backup every accepted job after stage 1 (before attempting stage 2)
	- random: backup each accepted job after stage 1 with probability `backup_probability`
	- none: never backup

	Losses:
	- If stage-1 full: arrival refused (push tag error)
	- If stage-2 full: result refused -> blank page. If a backup exists, the blank page is not permanent.
	"""
	if horizon <= 0:
		raise ValueError("horizon must be > 0")
	if arrival_rate < 0:
		raise ValueError("arrival_rate must be >= 0")
	if exec_servers <= 0:
		raise ValueError("exec_servers must be >= 1")
	if exec_service_rate <= 0 or send_service_rate <= 0:
		raise ValueError("service rates must be > 0")
	if ks is not None and ks < 0:
		raise ValueError("ks must be >= 0 or None")
	if kf is not None and kf < 0:
		raise ValueError("kf must be >= 0 or None")
	if not (0.0 <= backup_probability <= 1.0):
		raise ValueError("backup_probability must be in [0, 1]")
	if backup_strategy is None:
		backup_strategy = "random" if backup_probability > 0 else "none"
	if backup_strategy not in ("none", "random", "systematic"):
		raise ValueError("backup_strategy must be one of: none, random, systematic")

	rng = random.Random(seed)
	event_seq = 0

	def push_event(t: float, kind: EventType, job_id: int) -> None:
		nonlocal event_seq
		event_seq += 1
		heapq.heappush(events, _Event(t=t, seq=event_seq, kind=kind, job_id=job_id))

	events: list[_Event] = []
	now = 0.0

	next_job_id = 0

	# Stage 1 state
	s1_busy = 0
	s1_queue: list[int] = []

	# Stage 2 state
	s2_busy = 0
	s2_queue: list[int] = []

	# Per-job mutable state
	population = "ALL"
	arrival_t: dict[int, float] = {}
	accepted: dict[int, bool] = {}
	refused_reason: dict[int, str] = {}

	s1_start_t: dict[int, float | None] = {}
	s1_end_t: dict[int, float | None] = {}
	s2_start_t: dict[int, float | None] = {}
	s2_end_t: dict[int, float | None] = {}

	stage2_refused: dict[int, bool] = {}
	backed_up: dict[int, bool] = {}
	lost_result: dict[int, bool] = {}

	def stage1_system_capacity() -> int | None:
		if ks is None:
			return None
		return exec_servers + ks

	def stage2_system_capacity() -> int | None:
		if kf is None:
			return None
		return 1 + kf

	def stage1_size() -> int:
		return s1_busy + len(s1_queue)

	def stage2_size() -> int:
		return s2_busy + len(s2_queue)

	def try_start_s1() -> None:
		nonlocal s1_busy, now
		while s1_busy < exec_servers and s1_queue:
			jid = s1_queue.pop(0)
			s1_busy += 1
			s1_start_t[jid] = now
			service_t = _expovariate(rng, exec_service_rate)
			push_event(now + service_t, "s1_done", jid)

	def try_start_s2() -> None:
		nonlocal s2_busy, now
		while s2_busy < 1 and s2_queue:
			jid = s2_queue.pop(0)
			s2_busy += 1
			s2_start_t[jid] = now
			service_t = _expovariate(rng, send_service_rate)
			push_event(now + service_t, "s2_done", jid)

	def schedule_next_arrival(t: float) -> None:
		if arrival_rate == 0:
			return
		dt = _expovariate(rng, arrival_rate)
		if t + dt <= horizon:
			push_event(t + dt, "arrival", -1)

	# First arrival
	push_event(0.0, "arrival", -1)

	while events:
		ev = heapq.heappop(events)
		now = ev.t
		if now > horizon:
			break

		if ev.kind == "arrival":
			next_job_id += 1
			jid = next_job_id
			arrival_t[jid] = now
			accepted[jid] = True
			refused_reason[jid] = ""
			s1_start_t[jid] = None
			s1_end_t[jid] = None
			s2_start_t[jid] = None
			s2_end_t[jid] = None
			stage2_refused[jid] = False
			backed_up[jid] = False
			lost_result[jid] = False

			cap1 = stage1_system_capacity()
			if cap1 is not None and stage1_size() >= cap1:
				accepted[jid] = False
				refused_reason[jid] = "stage1_full"
			else:
				s1_queue.append(jid)
				try_start_s1()

			schedule_next_arrival(now)

		elif ev.kind == "s1_done":
			jid = ev.job_id
			s1_busy -= 1
			s1_end_t[jid] = now
			try_start_s1()

			# Decide/write backup at the end of stage 1 (before attempting stage 2)
			if backup_strategy == "systematic":
				backed_up[jid] = True
			elif backup_strategy == "random":
				backed_up[jid] = rng.random() < backup_probability

			cap2 = stage2_system_capacity()
			if cap2 is not None and stage2_size() >= cap2:
				stage2_refused[jid] = True
				lost_result[jid] = not backed_up[jid]
			else:
				s2_queue.append(jid)
				try_start_s2()

		elif ev.kind == "s2_done":
			jid = ev.job_id
			s2_busy -= 1
			s2_end_t[jid] = now
			try_start_s2()
		else:
			raise RuntimeError(f"unknown event kind {ev.kind}")

	jobs: list[JobRecord] = []
	for jid in sorted(arrival_t.keys()):
		jobs.append(
			JobRecord(
				job_id=jid,
				population=population,
				arrival_t=arrival_t[jid],
				accepted=accepted[jid],
				refused_reason=refused_reason[jid],
				s1_start_t=s1_start_t[jid],
				s1_end_t=s1_end_t[jid],
				s2_start_t=s2_start_t[jid],
				s2_end_t=s2_end_t[jid],
				stage2_refused=stage2_refused[jid],
				backed_up=backed_up[jid],
				lost_result=lost_result[jid],
			)
		)

	arrivals = len(jobs)
	accepted_count = sum(1 for j in jobs if j.accepted)
	refused_stage1 = sum(1 for j in jobs if not j.accepted and j.refused_reason == "stage1_full")
	s2_refused = sum(1 for j in jobs if j.accepted and j.stage2_refused)
	blank_pages = sum(1 for j in jobs if j.accepted and j.stage2_refused)
	permanent_blanks = sum(1 for j in jobs if j.accepted and j.lost_result)
	sojourns = [j.sojourn_time() for j in jobs]
	mean_s, var_s, completed = summarise_sojourn(t for t in sojourns if t is not None)

	metrics = RunMetrics(
		seed=seed,
		horizon=horizon,
		arrivals=arrivals,
		accepted=accepted_count,
		refused_stage1=refused_stage1,
		stage2_refused=s2_refused,
		blank_pages=blank_pages,
		permanent_blanks=permanent_blanks,
		completed=completed,
		mean_sojourn=mean_s,
		var_sojourn=var_s,
	)
	return metrics, jobs


@dataclass(frozen=True)
class Population:
	name: str
	arrival_rate: float
	exec_service_rate: float


Policy = Literal["fifo", "priority_ing", "split_servers"]


def simulate_channels_and_dams(
	*,
	seed: int,
	horizon: float,
	exec_servers: int,
	send_service_rate: float,
	ks: int | None,
	kf: int | None,
	backup_probability: float = 0.0,
	backup_strategy: BackupStrategy | None = None,
	populations: list[Population],
	dam_tb: float | None = None,
	dam_population: str = "ING",
	policy: Policy = "fifo",
	split_servers_for: dict[str, int] | None = None,
) -> tuple[dict[str, RunMetrics], list[JobRecord]]:
	"""Waterfall with multiple populations + optional periodic dam (on/off acceptance) + policy.

	- Arrivals are the superposition of independent Poisson processes (one per population).
	- Stage 1 service rate depends on population.
	- The dam, if enabled, alternates: closed for tb, open for tb/2, repeating.
	  When closed, arrivals of `dam_population` are refused.

	Policies:
	- fifo: single shared FIFO queue
	- priority_ing: ING jobs have priority over others in stage-1 waiting line
	- split_servers: separate stage-1 queues per population, each with a fixed number of servers.
	  Provide `split_servers_for`, e.g. {"ING": 2, "PREPA": 1}.
	"""
	if horizon <= 0:
		raise ValueError("horizon must be > 0")
	if exec_servers <= 0:
		raise ValueError("exec_servers must be >= 1")
	if send_service_rate <= 0:
		raise ValueError("send_service_rate must be > 0")
	if ks is not None and ks < 0:
		raise ValueError("ks must be >= 0 or None")
	if kf is not None and kf < 0:
		raise ValueError("kf must be >= 0 or None")
	if not (0.0 <= backup_probability <= 1.0):
		raise ValueError("backup_probability must be in [0, 1]")
	if backup_strategy is None:
		backup_strategy = "random" if backup_probability > 0 else "none"
	if backup_strategy not in ("none", "random", "systematic"):
		raise ValueError("backup_strategy must be one of: none, random, systematic")
	if not populations:
		raise ValueError("populations must be non-empty")
	if dam_tb is not None and dam_tb <= 0:
		raise ValueError("dam_tb must be > 0")
	if policy == "split_servers" and not split_servers_for:
		raise ValueError("split_servers_for must be provided for split_servers policy")

	rng = random.Random(seed)
	pop_by_name = {p.name: p for p in populations}
	if len(pop_by_name) != len(populations):
		raise ValueError("population names must be unique")

	def is_dam_open(t: float) -> bool:
		if dam_tb is None:
			return True
		cycle = dam_tb + dam_tb / 2.0
		x = t % cycle
		return x >= dam_tb

	event_seq = 0
	events: list[tuple[float, int, str, int, str]] = []
	# (t, seq, kind, job_id, population)

	def push_event(t: float, kind: str, job_id: int, population_name: str) -> None:
		nonlocal event_seq
		event_seq += 1
		heapq.heappush(events, (t, event_seq, kind, job_id, population_name))

	now = 0.0
	next_job_id = 0

	# Stage 1 state
	s1_busy_total = 0
	s1_busy_by_pop: dict[str, int] = {p.name: 0 for p in populations}
	s1_queue: list[tuple[str, int]] = []  # (pop, jid)
	s1_queue_by_pop: dict[str, list[int]] = {p.name: [] for p in populations}

	# Stage 2 state (shared)
	s2_busy = 0
	s2_queue: list[int] = []

	arrival_t: dict[int, float] = {}
	population_of: dict[int, str] = {}
	accepted: dict[int, bool] = {}
	refused_reason: dict[int, str] = {}
	s1_start_t: dict[int, float | None] = {}
	s1_end_t: dict[int, float | None] = {}
	s2_start_t: dict[int, float | None] = {}
	s2_end_t: dict[int, float | None] = {}
	stage2_refused: dict[int, bool] = {}
	backed_up: dict[int, bool] = {}
	lost_result: dict[int, bool] = {}

	def stage1_system_capacity() -> int | None:
		if ks is None:
			return None
		return exec_servers + ks

	def stage2_system_capacity() -> int | None:
		if kf is None:
			return None
		return 1 + kf

	def stage1_size() -> int:
		if policy == "split_servers":
			return s1_busy_total + sum(len(q) for q in s1_queue_by_pop.values())
		return s1_busy_total + len(s1_queue)

	def stage2_size() -> int:
		return s2_busy + len(s2_queue)

	def pop_server_budget(name: str) -> int:
		if policy != "split_servers":
			return exec_servers
		n = split_servers_for.get(name, 0) if split_servers_for else 0
		return n

	def try_start_s1() -> None:
		nonlocal s1_busy_total
		if policy == "split_servers":
			for pop_name, q in s1_queue_by_pop.items():
				budget = pop_server_budget(pop_name)
				while s1_busy_by_pop[pop_name] < budget and q:
					jid = q.pop(0)
					s1_busy_by_pop[pop_name] += 1
					s1_busy_total += 1
					s1_start_t[jid] = now
					service_t = _expovariate(rng, pop_by_name[pop_name].exec_service_rate)
					push_event(now + service_t, "s1_done", jid, pop_name)
			return

		# shared server pool
		while s1_busy_total < exec_servers and s1_queue:
			pop_name, jid = s1_queue.pop(0)
			s1_busy_total += 1
			s1_busy_by_pop[pop_name] += 1
			s1_start_t[jid] = now
			service_t = _expovariate(rng, pop_by_name[pop_name].exec_service_rate)
			push_event(now + service_t, "s1_done", jid, pop_name)

	def try_start_s2() -> None:
		nonlocal s2_busy
		while s2_busy < 1 and s2_queue:
			jid = s2_queue.pop(0)
			s2_busy += 1
			s2_start_t[jid] = now
			service_t = _expovariate(rng, send_service_rate)
			push_event(now + service_t, "s2_done", jid, population_of[jid])

	def schedule_next_arrival(t: float, pop_name: str) -> None:
		rate = pop_by_name[pop_name].arrival_rate
		if rate == 0:
			return
		dt = _expovariate(rng, rate)
		if t + dt <= horizon:
			push_event(t + dt, "arrival", -1, pop_name)

	# seed arrivals for each population
	for p in populations:
		push_event(0.0, "arrival", -1, p.name)

	while events:
		t, _seq, kind, jid, pop_name = heapq.heappop(events)
		now = t
		if now > horizon:
			break

		if kind == "arrival":
			next_job_id += 1
			jid = next_job_id
			arrival_t[jid] = now
			accepted[jid] = True
			refused_reason[jid] = ""
			s1_start_t[jid] = None
			s1_end_t[jid] = None
			s2_start_t[jid] = None
			s2_end_t[jid] = None
			stage2_refused[jid] = False
			backed_up[jid] = False
			lost_result[jid] = False
			population_of[jid] = pop_name  # FIX: always register jid's population

			if pop_name == dam_population and not is_dam_open(now):
				accepted[jid] = False
				refused_reason[jid] = "dam_closed"
			else:
				cap1 = stage1_system_capacity()
				if cap1 is not None and stage1_size() >= cap1:
					accepted[jid] = False
					refused_reason[jid] = "stage1_full"
				else:
					if policy == "split_servers":
						s1_queue_by_pop[pop_name].append(jid)
					elif policy == "priority_ing":
						# ING first, but preserve FIFO within each class
						if pop_name == "ING":
							s1_queue.insert(0, (pop_name, jid))
						else:
							s1_queue.append((pop_name, jid))
					else:
						s1_queue.append((pop_name, jid))
					try_start_s1()

			schedule_next_arrival(now, pop_name)

		elif kind == "s1_done":
			s1_busy_total -= 1
			s1_busy_by_pop[pop_name] -= 1
			s1_end_t[jid] = now
			try_start_s1()

			# Decide/write backup at the end of stage 1 (before attempting stage 2)
			if backup_strategy == "systematic":
				backed_up[jid] = True
			elif backup_strategy == "random":
				backed_up[jid] = rng.random() < backup_probability

			cap2 = stage2_system_capacity()
			if cap2 is not None and stage2_size() >= cap2:
				stage2_refused[jid] = True
				lost_result[jid] = not backed_up[jid]
			else:
				s2_queue.append(jid)
				try_start_s2()

		elif kind == "s2_done":
			s2_busy -= 1
			s2_end_t[jid] = now
			try_start_s2()
		else:
			raise RuntimeError(f"unknown kind {kind}")

	jobs: list[JobRecord] = []
	for jid in sorted(arrival_t.keys()):
		jobs.append(
			JobRecord(
				job_id=jid,
				population=population_of[jid],
				arrival_t=arrival_t[jid],
				accepted=accepted[jid],
				refused_reason=refused_reason[jid],
				s1_start_t=s1_start_t[jid],
				s1_end_t=s1_end_t[jid],
				s2_start_t=s2_start_t[jid],
				s2_end_t=s2_end_t[jid],
				stage2_refused=stage2_refused[jid],
				backed_up=backed_up[jid],
				lost_result=lost_result[jid],
			)
		)

	# Aggregate metrics per population
	metrics_by_pop: dict[str, RunMetrics] = {}
	for pop in populations:
		pop_jobs = [j for j in jobs if j.population == pop.name]
		arrivals = len(pop_jobs)
		accepted_count = sum(1 for j in pop_jobs if j.accepted)
		refused_stage1 = sum(1 for j in pop_jobs if not j.accepted and j.refused_reason == "stage1_full")
		s2_refused = sum(1 for j in pop_jobs if j.accepted and j.stage2_refused)
		blank_pages = sum(1 for j in pop_jobs if j.accepted and j.stage2_refused)
		permanent_blanks = sum(1 for j in pop_jobs if j.accepted and j.lost_result)
		mean_s, var_s, completed = summarise_sojourn(
			j.sojourn_time() for j in pop_jobs if j.sojourn_time() is not None
		)
		metrics_by_pop[pop.name] = RunMetrics(
			seed=seed,
			horizon=horizon,
			arrivals=arrivals,
			accepted=accepted_count,
			refused_stage1=refused_stage1,
			stage2_refused=s2_refused,
			blank_pages=blank_pages,
			permanent_blanks=permanent_blanks,
			completed=completed,
			mean_sojourn=mean_s,
			var_sojourn=var_s,
		)

	return metrics_by_pop, jobs


def waterfall_server_metrics(jobs: list[JobRecord], horizon: float, exec_servers: int) -> dict[str, float]:
    # Utilisation par intégrale des temps de service observés
    s1_busy = 0.0
    s2_busy = 0.0
    backups_written = 0
    for j in jobs:
        if not j.accepted:
            continue
        if j.backed_up:
            backups_written += 1
        if j.s1_start_t is not None and j.s1_end_t is not None:
            s1_busy += max(0.0, min(j.s1_end_t, horizon) - max(j.s1_start_t, 0.0))
        if (not j.stage2_refused) and j.s2_start_t is not None and j.s2_end_t is not None:
            s2_busy += max(0.0, min(j.s2_end_t, horizon) - max(j.s2_start_t, 0.0))

    s1_util = s1_busy / (max(1, exec_servers) * horizon)
    s2_util = s2_busy / (1.0 * horizon)

    series = waterfall_series_df(jobs, horizon)
    avg = (
        series.groupby('series', as_index=False)
        .apply(lambda g: time_average_step(g[['t', 'x']], horizon), include_groups=False)
        .reset_index()
    )
    # pandas >=2.0: .apply returns DataFrame with column None, not 0
    if None in avg.columns:
        avg = avg.rename(columns={None: 'time_avg'})
    elif 0 in avg.columns:
        avg = avg.rename(columns={0: 'time_avg'})
    avg_map = dict(zip(avg['series'], avg['time_avg']))

    return {
        's1_util': float(s1_util),
        's2_util': float(s2_util),
        's1_system_timeavg': float(avg_map.get('S1_system', 0.0)),
        's1_queue_timeavg': float(avg_map.get('S1_queue', 0.0)),
        's2_system_timeavg': float(avg_map.get('S2_system', 0.0)),
        's2_queue_timeavg': float(avg_map.get('S2_queue', 0.0)),
        'backups_written': float(backups_written),
    }

