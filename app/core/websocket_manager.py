"""
WebSocket connection manager
"""

import logging
import asyncio
import uuid
from typing import Dict, Set
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from app.config import WebSocketConfig

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manager untuk WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, dict] = {}
        self.max_connections = WebSocketConfig.MAX_CONNECTIONS
        
    async def connect(self, websocket: WebSocket) -> str:
        """Accept WebSocket connection dan return client ID"""
        
        # Check connection limit
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=1008, reason="Too many connections")
            raise Exception("Connection limit exceeded")
        
        # Accept connection
        await websocket.accept()
        
        # Generate client ID
        client_id = str(uuid.uuid4())
        
        # Store connection
        self.active_connections[client_id] = websocket
        self.connection_info[client_id] = {
            "connected_at": datetime.now(),
            "last_activity": datetime.now(),
            "frames_processed": 0,
            "errors": 0
        }
        
        # Send connection confirmation
        await websocket.send_json({
            "event": "connected",
            "data": {
                "client_id": client_id,
                "server_time": datetime.now().isoformat()
            }
        })
        
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
        return client_id
    
    async def disconnect(self, client_id: str) -> None:
        """Disconnect client"""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket for {client_id}: {e}")
            
            # Remove from active connections
            del self.active_connections[client_id]
            
            # Clean up connection info
            if client_id in self.connection_info:
                connection_time = datetime.now() - self.connection_info[client_id]["connected_at"]
                logger.info(
                    f"Client {client_id} disconnected. "
                    f"Connection time: {connection_time.total_seconds():.1f}s, "
                    f"Frames processed: {self.connection_info[client_id]['frames_processed']}"
                )
                del self.connection_info[client_id]
            
            logger.info(f"Client {client_id} removed. Total connections: {len(self.active_connections)}")
    
    async def disconnect_all(self) -> None:
        """Disconnect all clients"""
        logger.info("Disconnecting all clients...")
        
        disconnect_tasks = []
        for client_id in list(self.active_connections.keys()):
            disconnect_tasks.append(self.disconnect(client_id))
        
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        logger.info("All clients disconnected")
    
    async def send_to_client(self, client_id: str, message: dict) -> bool:
        """Send message to specific client"""
        if client_id not in self.active_connections:
            logger.warning(f"Client {client_id} not found")
            return False
        
        try:
            websocket = self.active_connections[client_id]
            await websocket.send_json(message)
            
            # Update activity
            if client_id in self.connection_info:
                self.connection_info[client_id]["last_activity"] = datetime.now()
            
            return True
            
        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected during send")
            await self.disconnect(client_id)
            return False
        except Exception as e:
            logger.error(f"Error sending message to {client_id}: {e}")
            
            # Increment error count
            if client_id in self.connection_info:
                self.connection_info[client_id]["errors"] += 1
            
            return False
    
    async def broadcast(self, message: dict, exclude_client: str = None) -> int:
        """Broadcast message to all connected clients"""
        sent_count = 0
        
        for client_id in list(self.active_connections.keys()):
            if exclude_client and client_id == exclude_client:
                continue
            
            if await self.send_to_client(client_id, message):
                sent_count += 1
        
        return sent_count
    
    def update_client_stats(self, client_id: str, frames_processed: int = 0, errors: int = 0) -> None:
        """Update client statistics"""
        if client_id in self.connection_info:
            if frames_processed > 0:
                self.connection_info[client_id]["frames_processed"] += frames_processed
            if errors > 0:
                self.connection_info[client_id]["errors"] += errors
            
            self.connection_info[client_id]["last_activity"] = datetime.now()
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_client_info(self, client_id: str) -> dict:
        """Get information about specific client"""
        if client_id not in self.connection_info:
            return {}
        
        info = self.connection_info[client_id].copy()
        info["connection_duration"] = (datetime.now() - info["connected_at"]).total_seconds()
        return info
    
    def get_all_clients_info(self) -> Dict[str, dict]:
        """Get information about all connected clients"""
        result = {}
        for client_id in self.connection_info:
            result[client_id] = self.get_client_info(client_id)
        return result
    
    async def cleanup_inactive_connections(self, timeout_seconds: int = None) -> int:
        """Clean up inactive connections"""
        if timeout_seconds is None:
            timeout_seconds = WebSocketConfig.CONNECTION_TIMEOUT
        
        current_time = datetime.now()
        inactive_clients = []
        
        for client_id, info in self.connection_info.items():
            inactive_duration = (current_time - info["last_activity"]).total_seconds()
            if inactive_duration > timeout_seconds:
                inactive_clients.append(client_id)
        
        # Disconnect inactive clients
        for client_id in inactive_clients:
            logger.info(f"Disconnecting inactive client {client_id}")
            await self.disconnect(client_id)
        
        return len(inactive_clients)
    
    def get_stats(self) -> dict:
        """Get overall WebSocket statistics"""
        total_frames = sum(info["frames_processed"] for info in self.connection_info.values())
        total_errors = sum(info["errors"] for info in self.connection_info.values())
        
        return {
            "active_connections": len(self.active_connections),
            "max_connections": self.max_connections,
            "total_frames_processed": total_frames,
            "total_errors": total_errors,
            "average_frames_per_client": total_frames / len(self.connection_info) if self.connection_info else 0
        } 