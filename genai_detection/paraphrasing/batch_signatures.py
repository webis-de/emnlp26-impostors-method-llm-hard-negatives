# Copyright 2026 Klara M. Gutekunst, Webis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Structured signature helpers for OpenAI batch paraphrasing."""

from __future__ import annotations

import json
import re
from typing import Any, Dict


class DSPyOneStepBatchSignature:
    """Batch API representation of DSPyOneStepSignature."""

    name = "DSPyOneStepSignature"
    schema_name = "dspy_one_step_paraphrase"
    input_fields = ["text"]
    output_fields = ["paraphrase"]

    @classmethod
    def system_instruction(cls) -> str:
        return (
            "You are a paraphrasing assistant implementing:\n"
            "text: original text to paraphrase.\n"
            "paraphrase: only the final paraphrased text, preserving meaning "
            "and tone while using different wording and sentence structure. Avoid copying original phrasing.\n"
            "Return JSON that matches the provided schema."
        )

    @classmethod
    def input_content(cls, text: str) -> str:
        return json.dumps(
            {
                "text": text,
            },
            ensure_ascii=True,
        )

    @classmethod
    def response_format(cls) -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": cls.schema_name,
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "paraphrase": {
                            "type": "string",
                            "description": (
                                "A faithful paraphrase of the input text using "
                                "different wording and sentence structure while "
                                "preserving meaning and tone. Avoid copying original phrasing."
                            ),
                        }
                    },
                    "required": ["paraphrase"],
                    "additionalProperties": False,
                },
            },
        }

    @classmethod
    def metadata(cls) -> Dict[str, Any]:
        return {
            "framework": "dspy",
            "name": cls.name,
            "schema_name": cls.schema_name,
            "input_fields": cls.input_fields,
            "output_fields": cls.output_fields,
        }

    @classmethod
    def parse_chat_completion_body(cls, response_body: Dict[str, Any]) -> str:
        choices = response_body.get("choices", [])
        if not choices:
            return ""
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        content = str(content).strip()
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                content = str(parsed.get("paraphrase", ""))
        except json.JSONDecodeError:
            pass
        return re.sub(r"\s+", " ", content).strip()
