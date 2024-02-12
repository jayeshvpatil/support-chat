gcloud auth configure-docker \
    us-central1-docker.pkg.dev

docker build -t us-central1-docker.pkg.dev/dce-gcp-training/jp/support_chat:v1 .
docker push us-central1-docker.pkg.dev/dce-gcp-training/jp/support_chat:v1 .
docker run -dp 127.0.0.1:8502:8080 us-central1-docker.pkg.dev/dce-gcp-training/jp/support_chat:v1