# ERO2: simulating the moulinette

## API/CLI usage
### Usage
```sh
# Start the api/backend
cargo run --bin api

# In another terminal, send some jobs manually (1000 tags over 1 minute)
cargo run --bin cli push-tag-manual --num-tags 1000 --interval 1m

# Or, send jobs downloaded from the forge's Grafana
cargo run --bin cli push-tag-csv --file ./Pipelines-data-2025-12-21\ 16_53_00.csv
```

### Metrics
```sh
# Start Grafana & Prometheus
docker compose up -d
```

Grafana now runs at: http://localhost:4000
Prometheus now runs at: http://localhost:9090
