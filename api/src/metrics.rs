use metrics::{describe_gauge, gauge};
use metrics_exporter_prometheus::PrometheusBuilder;
use std::thread;
use std::time::Duration;

use crate::workers::Workers;

pub fn init(workers: Workers) {
    let builder = PrometheusBuilder::new().with_http_listener(([0, 0, 0, 0], 9000));

    builder.install().expect("failed to install recorder");

    let queue_length = gauge!("queue_length");
    describe_gauge!(
        "queue_length",
        "Number of tags waiting to be run by workers"
    );
    let worker_count = gauge!("worker_count");
    describe_gauge!("worker_count", "Number of workers available to run tags");

    thread::spawn(move || {
        loop {
            queue_length.set(workers.queue_len() as f64);
            worker_count.set(workers.num_workers() as f64);

            thread::sleep(Duration::from_millis(50));
        }
    });
}
