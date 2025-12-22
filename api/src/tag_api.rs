use std::sync::Mutex;

use common::types::{PushTagQuery, Response, UploadForgeCSV};
use crossbeam_channel::{Sender, unbounded};
use log::debug;
use poem_openapi::{
    OpenApi,
    payload::{Json, PlainText},
};

use crate::workers::Workers;
pub struct TagApi {
    tags_input: Mutex<Sender<PushTagQuery>>,
    workers: Workers,
}

#[OpenApi]
impl TagApi {
    pub fn new() -> Self {
        let (tags_input, tags_output) = unbounded();
        let workers = Workers::spawn(tags_output);

        let tags_input = Mutex::new(tags_input);
        TagApi {
            tags_input,
            workers,
        }
    }

    /// Check Health
    ///
    /// Responds with `I'm Healthy`
    #[oai(path = "/healthcheck", method = "get")]
    async fn health(&self) -> PlainText<String> {
        PlainText("I'm Healthy!".to_string())
    }

    /// Push tags
    ///
    /// Push some tags manually over a defined period of time
    #[oai(path = "/push-tag-manual", method = "post")]
    async fn push_tag_manual(&self, query: Json<PushTagQuery>) -> PlainText<String> {
        debug!("Received tag push request: {:?}", &query.0);
        self.tags_input
            .lock()
            .unwrap()
            .send(query.0)
            .expect("Workers are not dead so the queue is still open");
        PlainText("OK".to_string())
    }

    /// Push real tags csv
    ///
    /// Push the tags from the downloaded csv from the forge
    #[oai(path = "/push-tag-csv", method = "post")]
    async fn push_tag_csv(&self, query: UploadForgeCSV) -> Response<PlainText<String>> {
        if query
            .file
            .content_type()
            .is_some_and(|mime| mime != "text/csv")
        {
            return Response::WrongFileFormat;
        }
        debug!("Received tag push request: {query:?}");
        todo!()
    }

    /// Queue length
    ///
    /// Returns the amount of running tags
    #[oai(path = "/queue-length", method = "get")]
    async fn queue_length(&self) -> Json<usize> {
        let running_tags = self.workers.queue_len();
        debug!("Currently running tags: {}", running_tags);
        Json(running_tags)
    }
}
