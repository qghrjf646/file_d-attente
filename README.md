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

| Service    | Address               |
|------------|-----------------------|
| Api doc    | http://localhost:3000 |
| Grafana    | http://localhost:4000 |
| Prometheus | http://localhost:9090 |

### Preview
<img width="1904" height="1183" alt="Grafana dashboard screenshot" src="https://github.com/user-attachments/assets/4aedf78c-eaf0-4aee-ad2e-195098ccfa10" />
