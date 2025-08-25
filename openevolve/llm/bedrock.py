"""
AWS Bedrock API interface for LLMs
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from openevolve.config import LLMConfig
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class BedrockLLM(LLMInterface):
    """LLM interface using AWS Bedrock"""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.random_seed = getattr(model_cfg, "random_seed", None)

        # AWS Configuration
        self.aws_region = model_cfg.aws_region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self.aws_profile = model_cfg.aws_profile
        self.aws_access_key_id = model_cfg.aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = model_cfg.aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")

        # Set up AWS session and client
        self._setup_aws_client()

        # Detect model provider based on model name
        self.model_provider = self._detect_model_provider(self.model)

        # Only log unique models to reduce duplication
        if not hasattr(logger, "_initialized_bedrock_models"):
            logger._initialized_bedrock_models = set()

        if self.model not in logger._initialized_bedrock_models:
            logger.info(f"Initialized Bedrock LLM with model: {self.model} (provider: {self.model_provider})")
            logger._initialized_bedrock_models.add(self.model)

    def _setup_aws_client(self):
        """Set up AWS Bedrock client with proper authentication"""
        try:
            # Configure session based on available credentials
            session_kwargs = {}
            
            if self.aws_profile:
                session_kwargs["profile_name"] = self.aws_profile
            elif self.aws_access_key_id and self.aws_secret_access_key:
                session_kwargs["aws_access_key_id"] = self.aws_access_key_id
                session_kwargs["aws_secret_access_key"] = self.aws_secret_access_key

            session = boto3.Session(**session_kwargs)
            
            # Create Bedrock Runtime client
            self.client = session.client(
                service_name="bedrock-runtime",
                region_name=self.aws_region
            )
            
            logger.debug(f"AWS Bedrock client initialized for region: {self.aws_region}")
            
        except NoCredentialsError:
            logger.error(
                "AWS credentials not found. Please configure credentials via:\n"
                "1. AWS CLI: aws configure\n"
                "2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\n"
                "3. IAM roles (if running on AWS)\n"
                "4. Config file: aws_access_key_id, aws_secret_access_key fields"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AWS Bedrock client: {str(e)}")
            raise

    def _detect_model_provider(self, model_name: str) -> str:
        """Detect the model provider based on model name"""
        model_name_lower = model_name.lower()
        
        if "anthropic" in model_name_lower or "claude" in model_name_lower:
            return "anthropic"
        elif "amazon" in model_name_lower or "titan" in model_name_lower:
            return "amazon"
        elif "meta" in model_name_lower or "llama" in model_name_lower:
            return "meta"
        elif "mistral" in model_name_lower:
            return "mistral"
        elif "cohere" in model_name_lower:
            return "cohere"
        else:
            logger.warning(f"Unknown model provider for: {model_name}, defaulting to anthropic format")
            return "anthropic"

    def _format_request(self, system_message: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Format request based on model provider"""
        
        if self.model_provider == "anthropic":
            # Anthropic Claude format
            # Convert messages to Anthropic format
            anthropic_messages = []
            for msg in messages:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "system": system_message,
                "messages": anthropic_messages
            }
            
            # Add temperature and top_p if specified
            temperature = kwargs.get("temperature", self.temperature)
            if temperature is not None:
                request_body["temperature"] = temperature
                
            top_p = kwargs.get("top_p", self.top_p)
            if top_p is not None:
                request_body["top_p"] = top_p

        elif self.model_provider == "amazon":
            # Amazon Titan format
            # Combine system message and user messages
            prompt_text = system_message + "\n\n"
            for msg in messages:
                if msg["role"] == "user":
                    prompt_text += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt_text += f"Assistant: {msg['content']}\n"
            prompt_text += "Assistant:"
            
            request_body = {
                "inputText": prompt_text,
                "textGenerationConfig": {
                    "maxTokenCount": kwargs.get("max_tokens", self.max_tokens),
                    "temperature": kwargs.get("temperature", self.temperature),
                    "topP": kwargs.get("top_p", self.top_p),
                }
            }

        elif self.model_provider == "meta":
            # Meta Llama format
            # Format as chat completion
            prompt_text = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n"
            for msg in messages:
                if msg["role"] == "user":
                    prompt_text += f"{msg['content']} [/INST] "
                elif msg["role"] == "assistant":
                    prompt_text += f"{msg['content']} </s><s>[INST] "
            
            request_body = {
                "prompt": prompt_text,
                "max_gen_len": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
            }

        else:
            # Default to Anthropic format for unknown providers
            anthropic_messages = []
            for msg in messages:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "system": system_message,
                "messages": anthropic_messages
            }
            
            temperature = kwargs.get("temperature", self.temperature)
            if temperature is not None:
                request_body["temperature"] = temperature
                
            top_p = kwargs.get("top_p", self.top_p)
            if top_p is not None:
                request_body["top_p"] = top_p

        return request_body

    def _extract_response(self, response_body: Dict[str, Any]) -> str:
        """Extract text response based on model provider"""
        
        if self.model_provider == "anthropic":
            # Anthropic Claude response format
            if "content" in response_body:
                content = response_body["content"]
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text", "")
            return response_body.get("completion", "")

        elif self.model_provider == "amazon":
            # Amazon Titan response format
            results = response_body.get("results", [])
            if results:
                return results[0].get("outputText", "")

        elif self.model_provider == "meta":
            # Meta Llama response format
            return response_body.get("generation", "")

        else:
            # Default to Anthropic format
            if "content" in response_body:
                content = response_body["content"]
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text", "")
            return response_body.get("completion", "")
        
        return ""

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        
        # Format request for the specific model provider
        request_body = self._format_request(system_message, messages, **kwargs)
        
        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(request_body), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, request_body: Dict[str, Any]) -> str:
        """Make the actual Bedrock API call"""
        try:
            # Use asyncio to run the blocking API call in a thread pool
            loop = asyncio.get_event_loop()
            
            def invoke_model():
                return self.client.invoke_model(
                    body=json.dumps(request_body),
                    modelId=self.model,
                    accept="application/json",
                    contentType="application/json"
                )
            
            response = await loop.run_in_executor(None, invoke_model)
            
            # Parse response
            response_body = json.loads(response.get("body").read())
            
            # Extract text from response based on provider
            generated_text = self._extract_response(response_body)
            
            # Logging
            logger.debug(f"Bedrock API request body: {request_body}")
            logger.debug(f"Bedrock API response: {generated_text}")
            
            return generated_text
            
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            
            if error_code == "ValidationException":
                logger.error(f"Invalid request to Bedrock: {error_message}")
                raise ValueError(f"Invalid Bedrock request: {error_message}")
            elif error_code == "ResourceNotFoundException":
                logger.error(f"Model not found: {self.model}")
                raise ValueError(f"Bedrock model not found: {self.model}")
            elif error_code == "AccessDeniedException":
                logger.error(f"Access denied to Bedrock model: {self.model}")
                raise PermissionError(f"Access denied to Bedrock model: {self.model}")
            elif error_code == "ThrottlingException":
                logger.warning(f"Bedrock API throttled, will retry")
                raise  # Let retry logic handle this
            else:
                logger.error(f"Bedrock API error [{error_code}]: {error_message}")
                raise
                
        except Exception as e:
            logger.error(f"Unexpected error calling Bedrock API: {str(e)}")
            raise