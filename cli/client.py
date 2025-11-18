"""API client for kgirl services"""

import httpx
from typing import Dict, Any, Optional, List
from rich.console import Console

console = Console()


class KGirlClient:
    """Client for interacting with kgirl services"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.main_api_url = config.get("main_api_url", "http://localhost:8000")
        self.skin_os_url = config.get("skin_os_url", "http://localhost:8001")
        self.api_key = config.get("api_key", "")
        self.timeout = 30.0

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers with API key if configured"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def check_health(self) -> bool:
        """Check if main API is healthy"""
        try:
            response = httpx.get(
                f"{self.main_api_url}/healthz",
                timeout=5.0,
                headers=self._get_headers()
            )
            return response.status_code == 200
        except Exception:
            return False

    def check_skin_os_health(self) -> bool:
        """Check if skin-OS API is healthy"""
        try:
            response = httpx.get(
                f"{self.skin_os_url}/health",
                timeout=5.0
            )
            return response.status_code == 200
        except Exception:
            return False

    def check_database(self) -> bool:
        """Check database connectivity"""
        try:
            response = httpx.get(
                f"{self.main_api_url}/healthz",
                timeout=5.0,
                headers=self._get_headers()
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("database") == "ok"
            return False
        except Exception:
            return False

    # Main API methods
    def lease_job(self, queue_id: str = "default") -> Optional[Dict[str, Any]]:
        """Lease a job from the queue"""
        try:
            response = httpx.post(
                f"{self.main_api_url}/pq/lease",
                json={"queue_id": queue_id},
                timeout=self.timeout,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            console.print(f"[red]Error leasing job: {e}[/red]")
            return None

    def publish_rfv(self, rfv_data: Dict[str, Any]) -> bool:
        """Publish RFV (Representation Feature Vector) metadata"""
        try:
            response = httpx.post(
                f"{self.main_api_url}/rfv/publish",
                json=rfv_data,
                timeout=self.timeout,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return True
        except Exception as e:
            console.print(f"[red]Error publishing RFV: {e}[/red]")
            return False

    def select_data(self, query_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate SQL query for data selection"""
        try:
            response = httpx.post(
                f"{self.main_api_url}/ds/select",
                json=query_params,
                timeout=self.timeout,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            console.print(f"[red]Error selecting data: {e}[/red]")
            return None

    def train_step(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Submit training step metrics"""
        try:
            response = httpx.post(
                f"{self.main_api_url}/ml2/train_step",
                json=metrics,
                timeout=self.timeout,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            console.print(f"[red]Error submitting training step: {e}[/red]")
            return None

    def create_snapshot(self, snapshot_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create RFV snapshot"""
        try:
            response = httpx.post(
                f"{self.main_api_url}/rfv/snapshot",
                json=snapshot_data,
                timeout=self.timeout,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            console.print(f"[red]Error creating snapshot: {e}[/red]")
            return None

    # skin-OS API methods
    def process_packet(self, packet_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a data packet through skin-OS layers"""
        try:
            response = httpx.post(
                f"{self.skin_os_url}/process",
                json=packet_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            console.print(f"[red]Error processing packet: {e}[/red]")
            return None

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get skin-OS metrics"""
        try:
            response = httpx.get(
                f"{self.skin_os_url}/metrics",
                timeout=5.0
            )
            response.raise_for_status()
            return response.text  # Prometheus format
        except Exception as e:
            console.print(f"[red]Error getting metrics: {e}[/red]")
            return None

    def enrich_packet(self, packet_id: str, enrichment_data: Dict[str, Any]) -> bool:
        """Enrich a packet with additional data"""
        try:
            response = httpx.post(
                f"{self.skin_os_url}/enrich/{packet_id}",
                json=enrichment_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return True
        except Exception as e:
            console.print(f"[red]Error enriching packet: {e}[/red]")
            return False
