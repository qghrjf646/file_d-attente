use std::path::PathBuf;

use clap::Parser;
use colored::Colorize;
use humantime::{Duration, format_duration};

const BACKEND_URL: &str = "http://localhost:3000/api";

#[derive(Debug, Parser)]
struct PushTagQuery {
    /// Number of tags to add to the queue
    #[arg(long)]
    num_tags: u32,
    /// Time it will take to add the tags, at a regular interval
    #[arg(long)]
    interval: Option<Duration>,
}

impl From<PushTagQuery> for common::types::PushTagQuery {
    fn from(val: PushTagQuery) -> Self {
        common::types::PushTagQuery {
            num_tags: val.num_tags,
            interval: val.interval.map(|humantime| humantime.into()),
        }
    }
}

#[derive(Debug, Parser)]
pub struct UploadForgeCSV {
    /// Path to the forge's grafana csv export
    #[arg(long)]
    pub file: PathBuf,
}

#[derive(Parser)]
enum Commands {
    /// Ping the api to check that it's running
    Ping,
    /// Manual batch tag push
    PushTagManual(PushTagQuery),
    /// Add tags from the csv downloaded from forge's Grafana
    PushTagCsv(UploadForgeCSV),
}

#[tokio::main]
async fn main() {
    let command = Commands::parse();

    match command {
        Commands::Ping => {
            let response = reqwest::get(format!("{BACKEND_URL}/healthcheck")).await;
            match response {
                Ok(response) => {
                    println!(
                        "Server answered with: {}",
                        response
                            .text()
                            .await
                            .expect("Server to have answered with a string")
                            .green()
                            .bold()
                    );
                }
                Err(err) => println!("Server is down: {}", err.to_string().white().on_red()),
            }
        }
        Commands::PushTagManual(push_tag_query) => {
            let push_tag_query: common::types::PushTagQuery = push_tag_query.into();
            let response = reqwest::Client::new()
                .post(format!("{BACKEND_URL}/push-tag-manual"))
                .json(&push_tag_query)
                .send()
                .await;
            match response {
                Ok(_) => println!(
                    "Successfully sent {} tags over the next {}",
                    push_tag_query.num_tags.to_string().bold(),
                    format_duration(push_tag_query.interval.unwrap_or(std::time::Duration::ZERO))
                        .to_string()
                        .bold()
                ),
                Err(_) => println!("{}", "Server didn't respond!".white().on_red()),
            }
        }
        Commands::PushTagCsv(_upload_forge_csv) => todo!(),
    }
}
