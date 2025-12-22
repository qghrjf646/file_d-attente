use std::{
    sync::{
        Arc,
        atomic::{AtomicU32, AtomicU64},
    },
    thread::{self, sleep},
    time::Duration,
};

use common::types::PushTagQuery;
use crossbeam_channel::{Receiver, Sender, unbounded};
use rand::distr::Distribution;

/// Number of workers running jobs
const N_WORKERS: u32 = 10;
/// Makes time values go this much faster
const TIME_MULTIPLIER: u32 = 10;
/// Average time to complete a job
const JOB_DURATION: Duration = Duration::from_mins(5);
/// Random ratio amount for the job duration
const JOB_DURATION_RAND_RANGE: f32 = 0.08 / 2.;

/// Workers consume tags at a given interval
#[derive(Clone)]
pub struct Workers {
    /// Queue containing the separated tags
    single_tag_output: Receiver<Tag>,
    global_stats: Arc<WorkerStats>,
}

struct Worker {
    single_tag_output: Receiver<Tag>,
    global_stats: Arc<WorkerStats>,
}

// TODO: keep a status for each tags invidually too
struct WorkerStats {
    // TODO: number of tags processed, average job time etc
    working_count: AtomicU32,
    tags_processed: AtomicU64,
}

struct Tag {
    // uid: Uuid,
}

impl WorkerStats {
    fn new() -> Self {
        Self {
            working_count: AtomicU32::new(0),
            tags_processed: AtomicU64::new(0),
        }
    }
}
impl Tag {
    fn new() -> Self {
        Self {}
    }
}

impl Workers {
    pub fn spawn(tags_request_input: Receiver<PushTagQuery>) -> Self {
        let global_stats = Arc::new(WorkerStats::new());
        let (single_tag_input, single_tag_output) = unbounded();
        thread::spawn(|| create_tags(tags_request_input, single_tag_input));

        let _workers: Vec<_> = (0..N_WORKERS)
            .map(|_| Worker::spawn(single_tag_output.clone(), global_stats.clone()))
            .collect();

        Self {
            single_tag_output,
            global_stats,
        }
    }

    pub fn queue_len(&self) -> usize {
        self.single_tag_output.len()
    }

    pub fn num_workers(&self) -> u32 {
        // TODO: add variable amount of worker
        N_WORKERS
    }

    pub fn num_working(&self) -> u32 {
        self.global_stats
            .working_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn total_tags_processed(&self) -> u64 {
        self.global_stats
            .tags_processed
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Worker {
    fn spawn(
        single_tag_output: Receiver<Tag>,
        global_stats: Arc<WorkerStats>,
    ) -> thread::JoinHandle<()> {
        let worker = Self {
            single_tag_output,
            global_stats,
        };
        thread::spawn(|| worker.consume())
    }
    fn consume(self) {
        let distribution =
            rand::distr::Uniform::new(1. - JOB_DURATION_RAND_RANGE, 1. + JOB_DURATION_RAND_RANGE)
                .unwrap();
        let mut rng = rand::rng();
        while let Ok(_next_tag) = self.single_tag_output.recv() {
            let job_duration =
                (JOB_DURATION / TIME_MULTIPLIER).as_nanos() as f32 * distribution.sample(&mut rng);
            let job_duration = Duration::from_nanos(job_duration as u64);
            // Simulate a job running
            // TODO: add randomness here
            self.global_stats
                .working_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            thread::sleep(job_duration);

            self.global_stats
                .working_count
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            self.global_stats
                .tags_processed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Split tag requests into individual tags
fn create_tags(tags_request_input: Receiver<PushTagQuery>, single_tag_input: Sender<Tag>) {
    // FIXME: this thread can get blocked by a long running request instead of consuming all of them at the same time
    while let Ok(tag_request) = tags_request_input.recv() {
        let time_between_tags = if let Some(interval) = tag_request.interval {
            interval / tag_request.num_tags / TIME_MULTIPLIER
        } else {
            Duration::ZERO
        };

        for _ in 0..tag_request.num_tags {
            single_tag_input.send(Tag::new()).unwrap();
            thread::sleep(time_between_tags);
        }
    }
    sleep(Duration::MAX);
}
