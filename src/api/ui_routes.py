"""UI routes for the demo web interface using HTMX and Jinja2."""

import json as json_lib
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.db.postgres import get_postgres
from src.db.qdrant import get_qdrant
from src.llm.client import get_claude, get_embedding_client
from src.memory.recall import HybridRecallEngine
from src.memory.retention import RetentionEngine

logger = structlog.get_logger()

router = APIRouter(prefix="/ui", tags=["ui"])

templates = Jinja2Templates(directory="src/templates")


async def get_memory_stats(user_id: str) -> dict[str, Any]:
    """Get memory statistics for a user."""
    pg = await get_postgres()

    total = await pg.fetchval(
        "SELECT COUNT(*) FROM memories WHERE user_id = $1",
        user_id
    )

    opinions = await pg.fetchval(
        "SELECT COUNT(*) FROM memories WHERE user_id = $1 AND memory_type = 'opinion'",
        user_id
    )

    facts = await pg.fetchval(
        "SELECT COUNT(*) FROM memories WHERE user_id = $1 AND memory_type = 'fact'",
        user_id
    )

    avg_confidence = await pg.fetchval(
        "SELECT COALESCE(AVG(confidence), 0) FROM memories WHERE user_id = $1 AND memory_type = 'opinion'",
        user_id
    )

    return {
        "total": total or 0,
        "opinions": opinions or 0,
        "facts": facts or 0,
        "avg_confidence": float(avg_confidence or 0)
    }


async def get_user_memories(
    user_id: str,
    memory_type: str | None = None
) -> list[dict[str, Any]]:
    """Get all memories for a user, optionally filtered by type."""
    pg = await get_postgres()

    if memory_type:
        rows = await pg.fetch(
            """
            SELECT id, user_id, memory_type, content, confidence, evidence_count,
                   created_at, updated_at
            FROM memories
            WHERE user_id = $1 AND memory_type = $2
            ORDER BY updated_at DESC
            """,
            user_id, memory_type
        )
    else:
        rows = await pg.fetch(
            """
            SELECT id, user_id, memory_type, content, confidence, evidence_count,
                   created_at, updated_at
            FROM memories
            WHERE user_id = $1
            ORDER BY updated_at DESC
            """,
            user_id
        )

    return [dict(row) for row in rows]


def parse_memories_json(memories_data: Any) -> list:
    """Parse memories_used field which may be string or already parsed."""
    if isinstance(memories_data, str):
        try:
            return json_lib.loads(memories_data)
        except (json_lib.JSONDecodeError, TypeError):
            return []
    return memories_data if memories_data else []


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    user_id: str = Query(default="demo-user"),
    conversation_id: str | None = Query(default=None)
):
    """Render the chat page with conversation history sidebar."""
    pg = await get_postgres()

    # Load all conversations for sidebar
    conversations = await pg.get_conversations(user_id, limit=50)
    conversations_list = [
        {
            "id": str(conv["id"]),
            "title": conv["title"] or (conv["first_message"][:30] + "..." if conv["first_message"] and len(conv["first_message"]) > 30 else conv["first_message"]) or "New Chat",
            "updated_at": conv["updated_at"],
            "is_active": conversation_id and str(conv["id"]) == conversation_id
        }
        for conv in conversations
    ]

    # Load messages for selected conversation
    messages = []
    active_conversation_id = None

    if conversation_id:
        try:
            conv_uuid = UUID(conversation_id)
            chat_history = await pg.get_conversation_messages(conv_uuid, limit=100)
            active_conversation_id = conversation_id
            for row in chat_history:
                messages.append({
                    "role": row["role"],
                    "content": row["content"],
                    "memories_used": parse_memories_json(row["memories_used"]),
                    "created_at": row["created_at"]
                })
        except (ValueError, TypeError):
            pass
    elif conversations_list:
        # Load the most recent conversation by default
        active_conversation_id = conversations_list[0]["id"]
        conversations_list[0]["is_active"] = True
        conv_uuid = UUID(active_conversation_id)
        chat_history = await pg.get_conversation_messages(conv_uuid, limit=100)
        for row in chat_history:
            messages.append({
                "role": row["role"],
                "content": row["content"],
                "memories_used": parse_memories_json(row["memories_used"]),
                "created_at": row["created_at"]
            })

    # Load recent memories for sidebar
    memories = await get_user_memories(user_id)

    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "user_id": user_id,
            "conversation_id": active_conversation_id,
            "conversations": conversations_list,
            "messages": messages,
            "memories": memories[:10]
        }
    )


@router.post("/conversations/new", response_class=HTMLResponse)
async def create_new_conversation(
    request: Request,
    user_id: str = Form(...)
):
    """Create a new conversation and return redirect info."""
    pg = await get_postgres()
    conv_id = await pg.create_conversation(user_id)

    # Return HX-Redirect header to reload the page with new conversation
    response = HTMLResponse(content="")
    response.headers["HX-Redirect"] = f"/ui/?user_id={user_id}&conversation_id={conv_id}"
    return response


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(
    request: Request,
    user_id: str = Query(default="demo-user"),
    conversation_id: str | None = Query(default=None)
):
    """Render the chat interface (redirects to index)."""
    if conversation_id:
        return templates.TemplateResponse(
            "redirect.html",
            {"request": request, "url": f"/ui/?user_id={user_id}&conversation_id={conversation_id}"}
        )
    return templates.TemplateResponse(
        "redirect.html",
        {"request": request, "url": f"/ui/?user_id={user_id}"}
    )


@router.get("/memories", response_class=HTMLResponse)
async def memories_page(
    request: Request,
    user_id: str = Query(default="demo-user")
):
    """Render the memories view page."""
    memories = await get_user_memories(user_id)
    stats = await get_memory_stats(user_id)

    return templates.TemplateResponse(
        "memories.html",
        {
            "request": request,
            "user_id": user_id,
            "memories": memories,
            "stats": stats
        }
    )


@router.post("/chat/send", response_class=HTMLResponse)
async def send_message(
    request: Request,
    message: str = Form(...),
    user_id: str = Form(default="demo-user"),
    conversation_id: str | None = Form(default=None),
    use_memory: str | None = Form(default=None)
):
    """Process a chat message and return the response with memory updates."""
    should_use_memory = use_memory == "true"

    try:
        pg = await get_postgres()
        qdrant = await get_qdrant()
        claude = get_claude()
        embedder = get_embedding_client()

        # Create conversation if none exists
        if not conversation_id:
            conv_uuid = await pg.create_conversation(user_id)
            conversation_id = str(conv_uuid)
        else:
            conv_uuid = UUID(conversation_id)

        memories_used = []
        confidence_updates = []

        if should_use_memory:
            recall_engine = HybridRecallEngine(pg, qdrant, embedder)
            retention_engine = RetentionEngine(pg, qdrant, claude, embedder)

            recall_result = await recall_engine.recall_with_auto_classification(
                user_id=user_id,
                query=message,
                limit=5
            )
            recalled = recall_result.memories
            memories_used = recalled

            memory_context = "\n".join([
                f"- [{m.memory_type}] {m.content}" +
                (f" (confidence: {m.confidence:.0%})" if m.memory_type == "opinion" else "")
                for m in recalled
            ])

            system_prompt = f"""You are a helpful assistant with memory of past conversations.

Here are relevant memories about this user:
{memory_context if memory_context else "No relevant memories found."}

Use these memories to provide personalized, consistent responses. If you learn new information
about the user's preferences or opinions, naturally incorporate it into your response."""

            response_text = await claude.complete(
                system=system_prompt,
                user_message=message
            )

            conversation_text = f"User: {message}\nAssistant: {response_text}"
            retention_result = await retention_engine.process_conversation(
                user_id=user_id,
                conversation=conversation_text
            )

            for update in retention_result.updated_opinions:
                confidence_updates.append({
                    "content": str(update.opinion_id)[:8] + "...",
                    "evidence_type": update.evidence_type,
                    "old_confidence": update.old_confidence,
                    "new_confidence": update.new_confidence
                })
        else:
            response_text = await claude.complete(
                system="You are a helpful assistant.",
                user_message=message
            )

        memories_used_dicts = [
            {
                "memory_type": m.memory_type.value if hasattr(m.memory_type, 'value') else str(m.memory_type),
                "content": m.content,
                "confidence": m.confidence
            }
            for m in memories_used
        ]

        # Store chat messages with conversation_id
        await pg.insert_chat_message(
            conversation_id=conv_uuid,
            user_id=user_id,
            role="user",
            content=message,
        )
        await pg.insert_chat_message(
            conversation_id=conv_uuid,
            user_id=user_id,
            role="assistant",
            content=response_text,
            memories_used=memories_used_dicts,
        )

        return templates.TemplateResponse(
            "partials/chat_response.html",
            {
                "request": request,
                "user_message": message,
                "response": response_text,
                "memories_used": memories_used_dicts,
                "confidence_updates": confidence_updates,
                "conversation_id": conversation_id
            }
        )

    except Exception as e:
        logger.exception("Error processing chat message")
        return templates.TemplateResponse(
            "partials/chat_response.html",
            {
                "request": request,
                "user_message": message,
                "response": f"Error: {str(e)}",
                "memories_used": [],
                "confidence_updates": []
            }
        )


@router.post("/memories/add", response_class=HTMLResponse)
async def add_memory(
    request: Request,
    user_id: str = Form(...),
    memory_type: str = Form(...),
    content: str = Form(...),
    confidence: int = Form(default=70)
):
    """Manually add a memory."""
    try:
        pg = await get_postgres()
        qdrant = await get_qdrant()
        embedder = get_embedding_client()

        # Generate embedding
        embedding = await embedder.embed(content)

        # Insert into PostgreSQL
        memory_id = await pg.insert_memory(
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            confidence=confidence / 100.0
        )

        # Insert into Qdrant
        await qdrant.upsert(
            collection_name="memories",
            points=[{
                "id": str(memory_id),
                "vector": embedding,
                "payload": {
                    "user_id": user_id,
                    "memory_type": memory_type,
                    "content": content,
                    "confidence": confidence / 100.0
                }
            }]
        )

        logger.info("Memory added manually", memory_id=str(memory_id), memory_type=memory_type)

        # Redirect to reload page with updated stats
        response = HTMLResponse(content="")
        response.headers["HX-Redirect"] = f"/ui/memories?user_id={user_id}"
        return response

    except Exception as e:
        logger.exception("Error adding memory")
        return HTMLResponse(
            content=f'<div class="error">Fehler: {str(e)}</div>',
            status_code=500
        )


@router.get("/partials/memory-list", response_class=HTMLResponse)
async def memory_list_partial(
    request: Request,
    user_id: str = Query(default="demo-user"),
    memory_type: str = Query(default="")
):
    """Return the memory list partial for HTMX updates."""
    memories = await get_user_memories(
        user_id,
        memory_type if memory_type else None
    )

    return templates.TemplateResponse(
        "partials/memory_list.html",
        {
            "request": request,
            "memories": memories
        }
    )


@router.get("/partials/recalled-memories", response_class=HTMLResponse)
async def recalled_memories_partial(
    request: Request,
    user_id: str = Query(default="demo-user")
):
    """Return recent memories for the sidebar."""
    memories = await get_user_memories(user_id)

    return templates.TemplateResponse(
        "partials/recalled_memories.html",
        {
            "request": request,
            "memories": memories[:10]
        }
    )


@router.delete("/memories/{user_id}/{memory_id}", response_class=HTMLResponse)
async def delete_memory(
    request: Request,
    user_id: str,
    memory_id: UUID
):
    """Delete a memory and return empty response for HTMX swap."""
    try:
        pg = await get_postgres()
        qdrant = await get_qdrant()

        # Delete from PostgreSQL
        await pg.execute(
            "DELETE FROM memories WHERE id = $1 AND user_id = $2",
            memory_id, user_id
        )

        # Delete from Qdrant
        try:
            await qdrant.delete(
                collection_name="memories",
                points_selector={"points": [str(memory_id)]}
            )
        except Exception:
            pass  # Qdrant deletion is best-effort

        logger.info("Memory deleted", memory_id=str(memory_id), user_id=user_id)

        # Return empty string - HTMX will remove the row
        return HTMLResponse(content="")

    except Exception as e:
        logger.exception("Error deleting memory")
        return HTMLResponse(
            content=f'<tr><td colspan="6">Error: {str(e)}</td></tr>',
            status_code=500
        )


@router.delete("/conversations/{conversation_id}", response_class=HTMLResponse)
async def delete_conversation(
    request: Request,
    conversation_id: str,
    user_id: str = Query(default="demo-user")
):
    """Delete a specific conversation."""
    try:
        pg = await get_postgres()
        conv_uuid = UUID(conversation_id)
        await pg.delete_conversation(conv_uuid)
        logger.info("Conversation deleted", conversation_id=conversation_id)

        # Redirect to main page
        response = HTMLResponse(content="")
        response.headers["HX-Redirect"] = f"/ui/?user_id={user_id}"
        return response

    except Exception as e:
        logger.exception("Error deleting conversation")
        return HTMLResponse(
            content=f'<div class="error">Error: {str(e)}</div>',
            status_code=500
        )


@router.delete("/chat/{user_id}", response_class=HTMLResponse)
async def clear_all_chat_history(
    request: Request,
    user_id: str
):
    """Clear all chat history for a user."""
    try:
        pg = await get_postgres()
        count = await pg.clear_chat_history(user_id)
        logger.info("All chat history cleared", user_id=user_id, conversations_deleted=count)

        # Redirect to main page
        response = HTMLResponse(content="")
        response.headers["HX-Redirect"] = f"/ui/?user_id={user_id}"
        return response

    except Exception as e:
        logger.exception("Error clearing chat history")
        return HTMLResponse(
            content=f'<div class="error">Error: {str(e)}</div>',
            status_code=500
        )
