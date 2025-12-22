use common::types::{PushTagQuery, Response, UploadForgeCSV};
use log::debug;
use poem_openapi::{
    OpenApi,
    payload::{Json, PlainText},
};

pub struct TagApi;

#[OpenApi]
impl TagApi {
    #[oai(path = "/healthcheck", method = "get")]
    async fn health(&self) -> PlainText<String> {
        PlainText("I'm Healthy!".to_string())
    }

    #[oai(path = "/push-tag-manual", method = "post")]
    async fn push_tag_manual(&self, query: Json<PushTagQuery>) -> PlainText<String> {
        debug!("Received tag push request: {query:?}");
        PlainText("OK".to_string())
    }

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
        Response::Ok(PlainText("OK".to_string()))
    }
}
