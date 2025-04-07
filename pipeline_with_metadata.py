"""
title: Pipeline Enriched with Metadata
inspiration: https://openwebui.com/f/gregorbiswanger/ollama_api_facade_metadata
author: Cody Sandahl
version: 1.1.0
date: 2025-03-25
license: MIT
description: A Pipe that enriches the pipelines server by including the metadata dictionary. You should disable the normal pipelines connection if you use this.

USAGE:
1. Add this to OpenWebUI as a Function in the Admin Panel.
2. Configure the valves to connect to your Pipelines server if needed.
3. Pipelines will now receive the metadata dictionary as part of the body.
4. NOTE: if your Pipelines code directly passes the body dictionary to an LLM, you will probably need to strip out the metadata field first. Otherwise it will break the LLM call.

body
├── metadata
│   ├── user_id
│   ├── chat_id
│   ├── message_id
│   ├── session_id
│   ├── tool_ids
│   ├── files
│   ├── features
│   │   ├── image_generation
│   │   ├── code_interpreter
│   │   ├── web_search
│   ├── variables
│   │   ├── {{USER_NAME}}
│   │   ├── {{USER_LOCATION}}
│   │   ├── {{CURRENT_DATETIME}}
│   │   ├── {{CURRENT_DATE}}
│   │   ├── {{CURRENT_TIME}}
│   │   ├── {{CURRENT_WEEKDAY}}
│   │   ├── {{CURRENT_TIMEZONE}}
│   │   ├── {{USER_LANGUAGE}}
│   ├── model
│   │   ├── id
│   │   ├── name
│   │   ├── object
│   │   ├── created
│   │   ├── owned_by
│   │   ├── pipeline
│   │   │   ├── type
│   │   │   ├── valves
│   │   ├── openai
│   │   │   ├── id
│   │   │   ├── name
│   │   │   ├── object
│   │   │   ├── created
│   │   │   ├── owned_by
│   │   │   ├── pipeline
│   │   │       ├── type
│   │   │       ├── valves
│   ├── urlIdx
│   ├── info
│   │   ├── id
│   │   ├── user_id
│   │   ├── base_model_id
│   │   ├── name
│   │   ├── params
│   │   ├── meta
│   │   │   ├── profile_image_url
│   │   │   ├── description
│   │   │   ├── capabilities
│   │   │   │   ├── vision
│   │   │   │   ├── citations
│   │   │   ├── suggestion_prompts
│   │   │   ├── tags
│   │   │   ├── filterIds
│   │   ├── access_control
│   │   │   ├── read
│   │   │   │   ├── group_ids
│   │   │   │   ├── user_ids
│   │   │   ├── write
│   │   │       ├── group_ids
│   │   │       ├── user_ids
│   │   ├── is_active
│   │   ├── updated_at
│   │   ├── created_at
│   ├── actions
│   ├── direct
"""

from pydantic import BaseModel, Field
from typing import Union, Generator, Iterator
import json
import requests


class Pipe:
    class Valves(BaseModel):
        PIPELINE_API_BASE_URL: str = Field(
            default="http://host.docker.internal:9099",
            description="Base URL for accessing the Pipeline API.",
        )
        PIPELINE_API_KEY: str = Field(
            default="0p3n-w3bu!",
            description="API key for accessing the Pipeline API.",
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()

    def pipes(self):
        """Get a list of available models from the pipeline server."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.valves.PIPELINE_API_KEY}",
            }
            response = requests.get(
                f"{self.valves.PIPELINE_API_BASE_URL}/models", headers=headers
            )
            response.raise_for_status()
            models = response.json().get("data", [])
            # print(f"Debug: Available models: {models}")
            return [{"id": model["id"], "name": model["name"]} for model in models]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return [{"id": "error", "name": "Failed to fetch models."}]

    def pipe(
        self, body: dict, __user__: dict, __metadata__: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Pipe the request to the pipeline server with enriched metadata.
        Recent versions of OpenWebUI will pass metadata to this pipe if the __metadata__ field is present in the method signature.
        These values are not passed onto the pipeline server, but by embedding them into the body, we can access them in the pipeline server.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.PIPELINE_API_KEY}",
        }

        # print(f"Body\n{body}") # uncomment this to see the pre-transformed body

        # enrich the body with metadata
        model_name = body["model"].split(".")[-1]
        body["metadata"] = __metadata__
        body["model"] = model_name

        # print(f"Enriched request to pipeline server: {json.dumps(body, indent=2)}") # uncomment this to see the enriched body

        try:
            # send the request to the pipeline server
            r = requests.post(
                url=f"{self.valves.PIPELINE_API_BASE_URL}/chat/completions",
                json=body,
                headers=headers,
                stream=True,
            )
            r.raise_for_status()

            # return the response from the pipeline server
            if body.get("stream", False):
                return r.iter_lines()
            else:
                return r.json()

        except Exception as e:
            print(f"Error during request: {e}")
            return json.dumps({"error": f"Request error: {e}"})