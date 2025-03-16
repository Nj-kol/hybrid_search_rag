## Open WebUI



### With CPU

The assumption here is Ollama is on your computer :

```shell
docker run -d -p 3000:8080 \
--add-host=host.docker.internal:host-gateway \
-v open-webui:/app/backend/data \
--name open-webui \
ghcr.io/open-webui/open-webui:main
```

If Ollama is on a Different Server, use this command:

To connect to Ollama on another server, change the OLLAMA_BASE_URL to the server's URL:

```shell
docker run -d -p 3000:8080 \
-e OLLAMA_BASE_URL=https://example.com \
-v open-webui:/app/backend/data \
--name open-webui \
ghcr.io/open-webui/open-webui:main
```

### With GPU

```shell
docker run -d -p 3000:8080 \
--add-host=host.docker.internal:host-gateway \
-v open-webui:/app/backend/data \
--gpus all \
--name open-webui \
ghcr.io/open-webui/open-webui:cuda
```

- UI : http://localhost:3000/


Lifecycle commands :

```shell
docker container stop open-webui
docker container start open-webui
```

## References
- https://github.com/open-webui/open-webui
- https://openwebui.com/




