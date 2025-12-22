mod metrics;
mod tag_api;
mod workers;

use poem::{Route, Server, listener::TcpListener};
use poem_openapi::OpenApiService;
use tag_api::TagApi;

#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    tracing_subscriber::fmt::init();

    let tag_api = TagApi::new();
    metrics::init(tag_api.get_stats());

    let api_service = OpenApiService::new(tag_api, "Moulinette simulation api", "0.1")
        .server("http://localhost:3000/api");
    let ui = api_service.rapidoc();

    Server::new(TcpListener::bind("0.0.0.0:3000"))
        .run(Route::new().nest("/api", api_service).nest("/", ui))
        .await
}
