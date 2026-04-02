"""Chat and insights API endpoints (REST + WebSocket)."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from netpulse_ml.api.schemas import ChatRequest, ChatResponse, InsightResponse
from netpulse_ml.config import settings
from netpulse_ml.llm.rag import RAGPipeline

router = APIRouter()


def _get_rag(request: Request) -> RAGPipeline:
    """Get RAG pipeline from app state."""
    rag = getattr(request.app.state, "rag_pipeline", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="LLM/RAG pipeline not initialized")
    return rag


@router.get("/insights/fleet-summary", response_model=InsightResponse)
async def fleet_insight(request: Request) -> InsightResponse:
    """Generate an AI-powered fleet health summary."""
    rag = _get_rag(request)

    try:
        summary = await rag.fleet_insight()
    except Exception as e:
        raise HTTPException(status_code=502, detail="LLM generation temporarily unavailable")

    return InsightResponse(
        content=summary,
        generatedAt=datetime.now(timezone.utc),
        model=request.app.state.ollama_provider.model_name,
    )


@router.get("/insights/device/{device_id}", response_model=InsightResponse)
async def device_insight(request: Request, device_id: str) -> InsightResponse:
    """Generate an AI-powered diagnosis for a specific device."""
    rag = _get_rag(request)

    try:
        diagnosis = await rag.device_diagnosis(device_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail="LLM generation temporarily unavailable")

    return InsightResponse(
        content=diagnosis,
        generatedAt=datetime.now(timezone.utc),
        model=request.app.state.ollama_provider.model_name,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Send a message and get a RAG-powered response (non-streaming)."""
    rag = _get_rag(request)

    try:
        answer = await rag.chat(body.message)
    except Exception as e:
        raise HTTPException(status_code=502, detail="LLM generation temporarily unavailable")

    return ChatResponse(
        message=body.message,
        response=answer,
        generatedAt=datetime.now(timezone.utc),
        model=request.app.state.ollama_provider.model_name,
    )


@router.websocket("/chat/ws")
async def chat_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming chat responses token by token."""
    # Authenticate WebSocket via query param (router-level Depends doesn't apply to WS)
    if settings.api_key:
        token = websocket.query_params.get("token", "")
        if token != settings.api_key:
            await websocket.close(code=4001)
            return

    await websocket.accept()

    rag = getattr(websocket.app.state, "rag_pipeline", None)
    if rag is None:
        await websocket.send_json({"error": "LLM/RAG pipeline not initialized"})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")

            if not message:
                await websocket.send_json({"error": "Empty message"})
                continue

            # Stream tokens back
            await websocket.send_json({"type": "start", "message": message})

            try:
                async for token in rag.chat_stream(message):
                    await websocket.send_json({"type": "token", "content": token})

                await websocket.send_json({"type": "end"})
            except Exception as e:
                await websocket.send_json({"type": "error", "content": str(e)})

    except WebSocketDisconnect:
        pass
