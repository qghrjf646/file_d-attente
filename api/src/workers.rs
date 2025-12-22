use std::{
    thread::{self, JoinHandle, sleep},
    time::Duration,
};

use common::types::PushTagQuery;
use crossbeam_channel::{Receiver, Sender, unbounded};
// use uuid::Uuid;

const N_WORKERS: u32 = 10;
const TIME_MULTIPLIER: u32 = 100;
const JOB_DURATION: Duration = Duration::from_mins(5);

/// Workers consume tags at a given interval
pub struct Workers {
    /// Queue containing the separated tags
    single_tag_output: Receiver<Tag>,
    #[allow(unused)]
    workers: Vec<JoinHandle<WorkerStats>>,
}

struct Worker {
    single_tag_output: Receiver<Tag>,
}

// TODO: keep a status for each tags invidually too
struct WorkerStats {
    // TODO: number of tags processed, average job time etc
}

struct Tag {
    // uid: Uuid,
}

impl Tag {
    fn new() -> Self {
        Self {
            // uid: Uuid::new_v4(),
        }
    }
}

impl Workers {
    pub fn spawn(tags_request_input: Receiver<PushTagQuery>) -> Self {
        let (single_tag_input, single_tag_output) = unbounded();
        thread::spawn(|| create_tags(tags_request_input, single_tag_input));

        let workers = (0..N_WORKERS)
            .map(|_| Worker::spawn(single_tag_output.clone()))
            .collect();

        Self {
            single_tag_output,
            workers,
        }
    }

    pub fn queue_len(&self) -> usize {
        self.single_tag_output.len()
    }
}

impl Worker {
    fn spawn(single_tag_output: Receiver<Tag>) -> thread::JoinHandle<WorkerStats> {
        let worker = Self { single_tag_output };
        thread::spawn(|| worker.consume())
    }
    fn consume(self) -> WorkerStats {
        while let Ok(_next_tag) = self.single_tag_output.recv() {
            // Simulate a job running
            thread::sleep(JOB_DURATION / TIME_MULTIPLIER);
        }
        WorkerStats {}
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
