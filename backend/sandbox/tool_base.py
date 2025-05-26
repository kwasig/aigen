from typing import Optional

from agentpress.thread_manager import ThreadManager
from agentpress.tool import Tool
from daytona_sdk import Sandbox
from daytona_api_client.models.workspace_state import WorkspaceState # Import WorkspaceState
from sandbox.sandbox import get_or_start_sandbox
from utils.logger import logger
from utils.files_utils import clean_path

class SandboxToolsBase(Tool):
    """Base class for all sandbox tools that provides project-based sandbox access."""
    
    # Class variable to track if sandbox URLs have been printed
    _urls_printed = False
    
    def __init__(self, project_id: str, thread_manager: Optional[ThreadManager] = None):
        super().__init__()
        self.project_id = project_id
        self.thread_manager = thread_manager
        self.workspace_path = "/workspace"
        self._sandbox = None
        self._cached_sandbox_instance: Optional[Sandbox] = None # Added for caching
        self._sandbox_id = None
        self._sandbox_pass = None

    async def _ensure_sandbox(self) -> Sandbox:
        """Ensure we have a valid sandbox instance, retrieving it from the project if needed, with caching."""
        if self._cached_sandbox_instance:
            try:
                # Validate the state of the cached sandbox instance
                current_state = self._cached_sandbox_instance.instance.state
                if current_state == WorkspaceState.RUNNING:
                    logger.info(f"Using cached and running sandbox instance (ID: {self._cached_sandbox_instance.id}) for project {self.project_id}")
                    self._sandbox = self._cached_sandbox_instance # Ensure self._sandbox is also updated
                    return self._sandbox
                else:
                    logger.info(f"Cached sandbox instance (ID: {self._cached_sandbox_instance.id}) for project {self.project_id} is not RUNNING (state: {current_state}). Will refresh.")
                    self._cached_sandbox_instance = None # Invalidate cache
            except AttributeError:
                logger.warning(f"Could not determine state of cached sandbox for project {self.project_id} (AttributeError). Will refresh.")
                self._cached_sandbox_instance = None # Invalidate cache
            except Exception as e:
                logger.warning(f"Error checking cached sandbox state for project {self.project_id}: {e}. Will refresh.")
                self._cached_sandbox_instance = None # Invalidate cache
        
        # Cache miss or invalid state, proceed to get/start sandbox
        logger.info(f"No valid cached sandbox or cache invalidated for project {self.project_id}. Fetching or starting new one.")
        try:
            # Get database client
            if not self.thread_manager or not hasattr(self.thread_manager, 'db'):
                 raise RuntimeError("ThreadManager or DB connection not available in SandboxToolsBase.")
            client = await self.thread_manager.db.client
            
            # Get project data
            project_result = await client.table('projects').select('sandbox').eq('project_id', self.project_id).execute() 
            if not project_result.data or len(project_result.data) == 0:
                raise ValueError(f"Project {self.project_id} not found")
            
            project_data = project_result.data[0]
            sandbox_info = project_data.get('sandbox', {})
            
            if not sandbox_info.get('id'):
                raise ValueError(f"No sandbox ID found in project data for project {self.project_id}")
            
            self._sandbox_id = sandbox_info['id']
            self._sandbox_pass = sandbox_info.get('pass') 
            
            logger.debug(f"Attempting to get or start sandbox with ID: {self._sandbox_id} for project {self.project_id}")
            current_sandbox = await get_or_start_sandbox(self._sandbox_id) 
            self._sandbox = current_sandbox 
            self._cached_sandbox_instance = self._sandbox 
            logger.info(f"Sandbox instance (ID: {self._sandbox.id}) fetched/started and cached for project {self.project_id}.")
            
            # Commented out URL printing logic
            # if not SandboxToolsBase._urls_printed:
            #     vnc_link = self._sandbox.get_preview_link(6080)
            #     website_link = self._sandbox.get_preview_link(8080)
            #     vnc_url = vnc_link.url if hasattr(vnc_link, 'url') else str(vnc_link)
            #     website_url = website_link.url if hasattr(website_link, 'url') else str(website_link)
            #     print("\033[95m***")
            #     print(f"VNC URL: {vnc_url}")
            #     print(f"Website URL: {website_url}")
            #     print("***\033[0m")
            #     SandboxToolsBase._urls_printed = True
                
        except Exception as e: # Manually "retyped" this line and the following two
            logger.error(f"Error retrieving sandbox for project {self.project_id}: {str(e)}", exc_info=True)
            raise e
        
        return self._sandbox

    @property
    def sandbox(self) -> Sandbox:
        """Get the sandbox instance, ensuring it exists."""
        if self._sandbox is None:
            raise RuntimeError("Sandbox not initialized. Call _ensure_sandbox() first.")
        return self._sandbox

    @property
    def sandbox_id(self) -> str:
        """Get the sandbox ID, ensuring it exists."""
        if self._sandbox_id is None:
            raise RuntimeError("Sandbox ID not initialized. Call _ensure_sandbox() first.")
        return self._sandbox_id

    def clean_path(self, path: str) -> str:
        """Clean and normalize a path to be relative to /workspace."""
        cleaned_path = clean_path(path, self.workspace_path)
        logger.debug(f"Cleaned path: {path} -> {cleaned_path}")
        return cleaned_path