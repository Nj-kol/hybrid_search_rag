## Qdrant Vector DB

### With CPU

```shell
docker pull qdrant/qdrant:v1.13.1

docker run -d \
--name qdrant-standalone \
-p 6333:6333 -p 6334:6334 \
qdrant/qdrant:v1.13.1
```

### With GPU

If you have a GPU on your machine, you can use this :

```shell
docker pull qdrant/qdrant:v1.13.1-gpu-nvidia

# `--gpus=all` flag says to Docker that we want to use GPUs.
# `-e QDRANT__GPU__INDEXING=1` flag says to Qdrant that we want to use GPUs for indexing.
docker run -d \
--gpus=all \
--name qdrant-standalone \
-p 6333:6333 -p 6334:6334 \
-e QDRANT__GPU__INDEXING=1 \
qdrant/qdrant:v1.13.1-gpu-nvidia
```

Qdrant is now accessible:
- Web UI: http://localhost:6333/dashboard
- REST API: [localhost:6333](http://localhost:6333/)
- GRPC API: [localhost:6334](http://localhost:6334/)

## References

- https://qdrant.tech/
- https://qdrant.tech/documentation/