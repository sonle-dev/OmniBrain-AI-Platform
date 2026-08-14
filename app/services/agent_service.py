import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from app.services.llm_factory import LLMFactory, _build_model_chain, _get_api_key_for_model
from app.core.config import settings
from app.services.tools.lms_tools import (
    get_lecturer_info,
    get_course_schedule,
    get_exam_schedule,
    LMS_TOOLS_SCHEMA
)

logger = logging.getLogger(__name__)


# Sample Tools Implementations
def get_database_records(table_name: str, query: str) -> str:
    """Queries the internal database tables like 'products', 'courses', 'users'."""
    logger.info(f"Executing Tool 'get_database_records' on table '{table_name}' with query: '{query}'")
    if "course" in table_name.lower() or "lms" in table_name.lower():
        return json.dumps([
            {"course_id": "CS101", "title": "Introduction to Artificial Intelligence", "instructor": "Dr. Sarah Connor", "enrolled": 152},
            {"course_id": "CS102", "title": "Advanced GraphRAG Systems", "instructor": "Prof. Alan Turing", "enrolled": 89}
        ])
    elif "product" in table_name.lower():
        return json.dumps([
            {"product_id": "P990", "name": "OmniBrain AI Hub License", "price": "$499/mo", "status": "In Stock"},
            {"product_id": "P101", "name": "DeepSeek R1 Compute Credits", "price": "$10/10M tokens", "status": "Available"}
        ])
    return json.dumps({"status": "success", "message": f"Query executed on {table_name}. Found 0 records matching '{query}'."})


def schedule_calendar_event(title: str, date_time: str, attendee_email: str) -> str:
    """Schedules a calendar event for a user or meeting."""
    logger.info(f"Executing Tool 'schedule_calendar_event' with title '{title}', time '{date_time}' for '{attendee_email}'")
    return json.dumps({
        "status": "scheduled",
        "event_id": "evt_998127391823",
        "title": title,
        "date_time": date_time,
        "attendee": attendee_email,
        "hangout_link": "https://meet.google.com/abc-defg-hij"
    })


def dispatch_notification(channel: str, recipient: str, message: str) -> str:
    """Dispatches a notification email, SMS, or Slack message."""
    logger.info(f"Executing Tool 'dispatch_notification' via channel '{channel}' to '{recipient}'")
    return json.dumps({
        "status": "dispatched",
        "channel": channel,
        "recipient": recipient,
        "message_preview": message[:30] + "..." if len(message) > 30 else message,
        "timestamp": "2026-08-13T21:50:00Z"
    })


class AgentService:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: List[Dict[str, Any]] = []
        self._register_default_tools()

    def register_tool(self, name: str, func: Callable, schema: Dict[str, Any]):
        """Registers a custom Python function as an LLM tool."""
        self._tools[name] = func
        self._tool_schemas.append(schema)

    def _register_default_tools(self):
        # 1. LMS Tools
        for schema in LMS_TOOLS_SCHEMA:
            tool_name = schema["function"]["name"]
            if tool_name == "get_lecturer_info":
                self.register_tool(tool_name, lambda subject_name: json.dumps(get_lecturer_info(subject_name), ensure_ascii=False), schema)
            elif tool_name == "get_course_schedule":
                self.register_tool(tool_name, lambda subject_name, class_id=None: json.dumps(get_course_schedule(subject_name, class_id), ensure_ascii=False), schema)
            elif tool_name == "get_exam_schedule":
                self.register_tool(tool_name, lambda student_id=None, subject_code=None: json.dumps(get_exam_schedule(student_id, subject_code), ensure_ascii=False), schema)

        # 2. Database records tool schema
        self.register_tool(
            name="get_database_records",
            func=get_database_records,
            schema={
                "type": "function",
                "function": {
                    "name": "get_database_records",
                    "description": "Access the platform databases to search records in 'products', 'courses', 'users'. Use when query involves search or retrieval of DB objects.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "The table name (e.g., 'courses', 'products', 'users')"
                            },
                            "query": {
                                "type": "string",
                                "description": "The search keyword or lookup term"
                            }
                        },
                        "required": ["table_name", "query"]
                    }
                }
            }
        )

        # 3. Schedule calendar event tool schema
        self.register_tool(
            name="schedule_calendar_event",
            func=schedule_calendar_event,
            schema={
                "type": "function",
                "function": {
                    "name": "schedule_calendar_event",
                    "description": "Schedules a calendar meeting event. Use when user wants to book or schedule a slot.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "The title of the meeting or event"
                            },
                            "date_time": {
                                "type": "string",
                                "description": "The date and time string (e.g. '2026-08-15 14:00')"
                            },
                            "attendee_email": {
                                "type": "string",
                                "description": "Email address of the participant"
                            }
                        },
                        "required": ["title", "date_time", "attendee_email"]
                    }
                }
            }
        )

        # 4. Notification tool schema
        self.register_tool(
            name="dispatch_notification",
            func=dispatch_notification,
            schema={
                "type": "function",
                "function": {
                    "name": "dispatch_notification",
                    "description": "Dispatches a message/alert to a recipient via 'email', 'sms', or 'slack'. Use when user requests sending or dispatching reports/messages.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "channel": {
                                "type": "string",
                                "enum": ["email", "sms", "slack"],
                                "description": "Communication channel to use"
                            },
                            "recipient": {
                                "type": "string",
                                "description": "Recipient address, email, phone number, or slack user/channel ID"
                            },
                            "message": {
                                "type": "string",
                                "description": "The content body of the message to send"
                            }
                        },
                        "required": ["channel", "recipient", "message"]
                    }
                }
            }
        )

    async def execute_agent(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs the LLM function calling loop. Feeds tools to the model, executes tools suggested,
        and returns the final reasoning answer.
        """
        import litellm

        messages = []
        messages.append({
            "role": "system",
            "content": (
                "You are an advanced AI Agent for the ICTU LMS platform equipped with tools for LMS database querying "
                "(lecturer details, course schedules, exam dates), calendar scheduling, and notifications. "
                "Use your tools whenever appropriate to retrieve real live facts. When you obtain results from a tool, "
                "synthesize the final answer clearly in Vietnamese for the user. Do not explain technical tool details unless asked."
            )
        })

        if history:
            for item in history:
                messages.append(item)

        messages.append({"role": "user", "content": prompt})

        # Build dynamic model chain for reasoning/agent task
        models_to_try = await _build_model_chain("chat", override_model=model)
        if not models_to_try:
            return {
                "text": "Hệ thống AI chưa sẵn sàng. Vui lòng kiểm tra lại cấu hình.",
                "trace": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

        selected_model = models_to_try[0]
        max_steps = 5
        step = 0
        execution_trace = []
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        while step < max_steps:
            step += 1
            logger.info(f"Agent Loop Step {step} with model {selected_model}")

            try:
                call_kwargs = {
                    "model": selected_model,
                    "messages": messages,
                    "tools": self._tool_schemas,
                    "tool_choice": "auto"
                }
                api_key = _get_api_key_for_model(selected_model)
                if api_key:
                    call_kwargs["api_key"] = api_key
                if selected_model.startswith("ollama/"):
                    call_kwargs["api_base"] = settings.OLLAMA_BASE_URL

                response = await litellm.acompletion(**call_kwargs)
            except Exception as e:
                logger.error(f"LiteLLM tool call failed for {selected_model}: {e}. Executing direct fallback...")
                fallback_res = await LLMFactory.chat_completion(messages=messages, model=selected_model)
                return {
                    "text": fallback_res.get("text", ""),
                    "trace": execution_trace,
                    "usage": fallback_res.get("usage", {})
                }

            choice = response.choices[0]
            message = choice.message

            if getattr(response, "usage", None):
                total_tokens["prompt_tokens"] += getattr(response.usage, "prompt_tokens", 0) or 0
                total_tokens["completion_tokens"] += getattr(response.usage, "completion_tokens", 0) or 0
                total_tokens["total_tokens"] += getattr(response.usage, "total_tokens", 0) or 0

            messages.append(message)

            if getattr(message, "tool_calls", None):
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
                    tool_call_id = tool_call.id

                    logger.info(f"Model requested tool '{tool_name}' with args {tool_args}")

                    if tool_name in self._tools:
                        try:
                            tool_func = self._tools[tool_name]
                            if asyncio.iscoroutinefunction(tool_func):
                                tool_result = await tool_func(**tool_args)
                            else:
                                tool_result = await asyncio.to_thread(tool_func, **tool_args)

                            execution_trace.append({
                                "tool": tool_name,
                                "arguments": tool_args,
                                "status": "success",
                                "result": tool_result
                            })
                        except Exception as tool_err:
                            logger.error(f"Error executing tool {tool_name}: {tool_err}")
                            tool_result = json.dumps({"status": "error", "detail": str(tool_err)})
                            execution_trace.append({
                                "tool": tool_name,
                                "arguments": tool_args,
                                "status": "failed",
                                "result": tool_result
                            })
                    else:
                        tool_result = json.dumps({"status": "error", "detail": f"Tool '{tool_name}' not found."})

                    messages.append({
                        "role": "tool",
                        "name": tool_name,
                        "tool_call_id": tool_call_id,
                        "content": str(tool_result)
                    })

                continue
            else:
                return {
                    "text": message.content or "",
                    "trace": execution_trace,
                    "usage": total_tokens
                }

        return {
            "text": "Agent execution timeout or max steps exceeded.",
            "trace": execution_trace,
            "usage": total_tokens
        }


# Singleton Instance
agent_service = AgentService()
