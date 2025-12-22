use std::time::Duration;

use poem::IntoResponse;
use poem_openapi::{ApiResponse, Multipart, Object, ResponseContent, types::multipart::Upload};
use serde::Serialize;

/// Manual batch tag push
#[derive(Debug, Object, Serialize)]
pub struct PushTagQuery {
    /// Number of tags to add to the queue
    pub num_tags: u32,
    /// Time it will take to add the tags, at a regular interval
    #[serde(default)]
    #[serde(with = "humantime_serde")]
    pub interval: Option<Duration>,
}

/// Add tags from the csv downloaded from forge's Grafana
#[derive(Debug, Multipart)]
pub struct UploadForgeCSV {
    pub file: Upload,
}

#[derive(ApiResponse)]
// #[oai(bad_request_handler = "my_bad_request_handler")]
pub enum Response<T: IntoResponse + ResponseContent + Send + Sync> {
    #[oai(status = 200)]
    Ok(T),
    /// UploadForgeCsv file should be a csv!
    #[oai(status = 400)]
    WrongFileFormat,
}
