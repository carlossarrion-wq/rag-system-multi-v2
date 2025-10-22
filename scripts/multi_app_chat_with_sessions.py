#!/usr/bin/env python3
"""
Multi-Application Conversational RAG Chat Interface with Session Support
Supports multiple concurrent users with isolated conversation sessions
"""

import sys
import argparse
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.multi_app_config_manager import MultiAppConfigManager
from src.agent.advanced_conversational_agent import AdvancedConversationalAgent, ConversationResponse
from src.agent.advanced_memory import MemoryContext
from src.agent.reasoning_agent_fixed import ReasoningResult
from src.agent.document_context_enhancer import DocumentContextEnhancer
from src.utils.connection_manager import ConnectionManager
from src.generation.structured_response_parser import StructuredResponseParser
from src.agent.session_manager import SessionManager, get_session_manager, create_user_session
import tempfile
import os
import yaml


class SessionAwareMultiAppAgent(AdvancedConversationalAgent):
    """Agente conversacional multi-aplicación con soporte para sesiones"""
    
    def __init__(self, app_name: Optional[str] = None, 
                 config_path: str = "config/multi_app_config.yaml",
                 session_id: Optional[str] = None,
                 user_id: Optional[str] = None):
        """
        Initialize session-aware multi-application conversational agent.
        
        Args:
            app_name: Name of the application to use
            config_path: Path to multi-application configuration
            session_id: Existing session ID (optional)
            user_id: User ID for session management (optional)
        """
        self.config_manager = MultiAppConfigManager(config_path)
        self.app_name = app_name or self.config_manager.default_app
        self.application_info = self.config_manager.get_application_info(self.app_name)
        self.user_id = user_id
        
        # Validate application
        if not self.config_manager.validate_application(self.app_name):
            available_apps = ', '.join(self.config_manager.get_available_applications())
            raise ValueError(f"Application '{self.app_name}' not found. Available: {available_apps}")
        
        # Initialize session manager
        self.session_manager = get_session_manager()
        
        # Create or get session
        if session_id:
            # Use existing session
            session_info = self.session_manager.get_session(session_id)
            if not session_info:
                raise ValueError(f"Session {session_id} not found or expired")
            self.session_id = session_id
            self.user_id = session_info.user_id or user_id
        else:
            # Create new session
            self.session_id = create_user_session(
                app_name=self.app_name,
                user_id=user_id,
                metadata={'created_by': 'multi_app_chat_with_sessions'}
            )
        
        # Create temporary legacy config file for compatibility
        self.legacy_config = self.config_manager.create_legacy_config(self.app_name)
        self.temp_config_file = self._create_temp_config_file()
        
        # Initialize components needed for parent class
        from src.generation.llm_client_fixed import LLMClient
        from src.generation.citation_manager_fixed import FixedCitationManager as CitationManager
        from src.retrieval.specialized_retrievers import RetrieverFactory
        from src.utils.config_loader import ConfigLoader
        
        # Load config using the temporary config file
        config_loader = ConfigLoader()
        config = config_loader.load_config(self.temp_config_file)
        
        # Initialize connection manager
        connection_manager = ConnectionManager(config_path=self.temp_config_file)
        
        # Initialize components with correct parameters
        llm_client = LLMClient(config_path=self.temp_config_file)
        
        # RetrieverFactory is a static class, just pass the class itself
        retriever_factory = RetrieverFactory
        
        citation_manager = CitationManager()
        
        # Initialize document context enhancer for context reduction
        try:
            document_enhancer = DocumentContextEnhancer(self.app_name, config_path)
            # Connect document enhancer to LLM client for context reduction
            llm_client.document_enhancer = document_enhancer
            print(f"🔗 Document enhancer connected to LLM client for context reduction")
        except Exception as e:
            print(f"⚠️  Warning: Could not connect document enhancer to LLM client: {e}")
        
        # Initialize image summary retriever for image fallback
        try:
            from src.generation.image_summary_retriever import ImageSummaryRetriever
            image_summary_retriever = ImageSummaryRetriever(self.app_name, config_path)
            # Connect image summary retriever to LLM client
            llm_client.image_summary_retriever = image_summary_retriever
            print(f"📸 Image summary retriever connected to LLM client")
        except Exception as e:
            print(f"⚠️  Warning: Could not connect image summary retriever to LLM client: {e}")
        
        # Get session-specific memory file
        session_memory_file = self.session_manager.get_memory_file_path(self.session_id)
        
        # Initialize parent class with session-aware memory
        super().__init__(
            llm_client=llm_client,
            retriever_factory=retriever_factory,
            citation_manager=citation_manager,
            memory_file=session_memory_file,
            config_path=self.temp_config_file,
            session_id=self.session_id  # Pass session_id to AdvancedMemory
        )
        
        # Override system prompt with application-specific one
        self.system_prompt = self.config_manager.get_system_prompt(self.app_name)
        
        # Override LLM client's system prompt method to use application-specific prompt
        original_get_system_prompt = self.llm_client._get_system_prompt
        self.llm_client._get_system_prompt = lambda custom_prompt=None: custom_prompt or self.system_prompt
        
        # Initialize document context enhancer for 2048+ char context (Haiku 3 caching)
        try:
            self.context_enhancer = DocumentContextEnhancer(self.app_name, config_path)
            print(f"📋 Document context enhancer initialized (Haiku 3 caching optimization)")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize context enhancer: {e}")
            self.context_enhancer = None
        
        # Initialize structured response parser
        self.response_parser = StructuredResponseParser()
        
        # Get session info for display
        session_info = self.session_manager.get_session(self.session_id)
        
        print(f"🚀 Session-Aware Multi-App RAG Agent initialized")
        print(f"👤 Session ID: {self.session_id}")
        print(f"🆔 User ID: {self.user_id or 'Anonymous'}")
        print(f"📱 Application: {self.application_info['name']}")
        print(f"🔍 Index: {self.config_manager.get_opensearch_index_name(self.app_name)}")
        print(f"📦 S3 Bucket: {self.config_manager.get_s3_config(self.app_name)['bucket']}")
        print(f"💾 Memory File: {session_memory_file}")
        print(f"💬 Custom system prompt loaded")
        print("-" * 60)
    
    def _create_temp_config_file(self) -> str:
        """Create temporary configuration file for legacy compatibility."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.legacy_config, temp_file, default_flow_style=False)
        temp_file.close()
        return temp_file.name
    
    def __del__(self):
        """Clean up temporary configuration file."""
        if hasattr(self, 'temp_config_file') and os.path.exists(self.temp_config_file):
            os.unlink(self.temp_config_file)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        session_info = self.session_manager.get_session(self.session_id)
        if session_info:
            return {
                'session_id': session_info.session_id,
                'user_id': session_info.user_id,
                'app_name': session_info.app_name,
                'created_at': session_info.created_at.isoformat(),
                'last_activity': session_info.last_activity.isoformat(),
                'metadata': session_info.metadata
            }
        return {}
    
    def list_user_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions for the current user"""
        if not self.user_id:
            return []
        
        sessions = self.session_manager.list_user_sessions(self.user_id, self.app_name)
        return [
            {
                'session_id': s.session_id,
                'app_name': s.app_name,
                'created_at': s.created_at.isoformat(),
                'last_activity': s.last_activity.isoformat(),
                'is_current': s.session_id == self.session_id
            }
            for s in sessions
        ]
    
    def switch_session(self, session_id: str) -> bool:
        """Switch to a different session"""
        try:
            session_info = self.session_manager.get_session(session_id)
            if not session_info:
                return False
            
            # Verify session belongs to current user (if user_id is set)
            if self.user_id and session_info.user_id != self.user_id:
                return False
            
            # Update current session
            self.session_id = session_id
            
            # Reinitialize memory with new session
            session_memory_file = self.session_manager.get_memory_file_path(session_id)
            
            # Create new memory instance with session
            from src.agent.advanced_memory import AdvancedMemory
            self.memory = AdvancedMemory(
                llm_client=self.llm_client,
                memory_file=session_memory_file,
                session_id=session_id
            )
            
            print(f"✅ Switched to session: {session_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error switching session: {e}")
            return False
    
    def _display_comprehensive_response(self, response):
        """Display comprehensive structured response information"""
        # === ASSISTANT RESPONSE SECTION ===
        print(f"\n{'='*60}")
        print(f"🤖  ASSISTANT RESPONSE")
        print(f"{'='*60}")
        
        # Check if we have structured response data
        structured_data = response.metadata.get('structured_response') if hasattr(response, 'metadata') else None
        
        if structured_data and 'answer' in structured_data:
            # Display the clean answer from structured JSON
            clean_answer = structured_data['answer']
            print(f"{clean_answer}")
        else:
            # Fallback to original answer for non-structured responses
            print(f"{response.answer}")
        
        # === CONFIDENCE SECTION ===
        print(f"\n{'='*60}")
        print(f"📊 CONFIDENCE ASSESSMENT")
        print(f"{'='*60}")
        print(f"{response.confidence_emoji} Score: {response.confidence_score*100:.1f}% ({response.confidence_level})")
        
        # Show confidence rationale - check multiple possible locations
        rationale_found = False
        
        # First, try to get rationale from structured response (JSON format)
        if hasattr(response, 'metadata') and response.metadata.get('structured_response'):
            structured_data = response.metadata['structured_response']
            if 'confidence' in structured_data and 'rationale' in structured_data['confidence']:
                rationale = structured_data['confidence']['rationale']
                if rationale and rationale.strip():
                    print(f"💭 Rationale: {rationale}")
                    rationale_found = True
        
        # If not found in structured response, try direct metadata
        if not rationale_found and hasattr(response, 'metadata') and 'confidence_rationale' in response.metadata:
            rationale = response.metadata['confidence_rationale']
            if rationale and rationale.strip():
                print(f"💭 Rationale: {rationale}")
                rationale_found = True
        
        # Show confidence factors if available in structured response
        if hasattr(response, 'metadata') and response.metadata.get('structured_response'):
            structured_data = response.metadata['structured_response']
            if 'confidence' in structured_data and 'factors' in structured_data['confidence']:
                factors = structured_data['confidence']['factors']
                print(f"\n🔍 Confidence Factors:")
                for factor_name, factor_data in factors.items():
                    score = factor_data.get('score', 0)
                    explanation = factor_data.get('explanation', '')
                    print(f"   • {factor_name.replace('_', ' ').title()}: {score}/30 - {explanation}")
        
        # === SOURCES SECTION ===
        # Priority: Use sources from structured JSON response if available, otherwise fallback to citation manager sources
        sources_to_display = []
        
        # Try to get sources from structured JSON response first
        if hasattr(response, 'metadata') and response.metadata.get('structured_response'):
            structured_data = response.metadata['structured_response']
            json_sources = structured_data.get('sources', [])
            if json_sources:
                sources_to_display = json_sources  # Show all sources returned by LLM
        
        # Fallback to citation manager sources if no JSON sources
        if not sources_to_display and response.sources:
            sources_to_display = response.sources  # Show all fallback sources too
        
        if sources_to_display:
            displayed_count = len(sources_to_display)
            
            print(f"\n{'='*60}")
            print(f"📚 USED SOURCES ({displayed_count} documents)")
            print(f"{'='*60}")
            for source in sources_to_display:
                # Handle both JSON sources and citation manager sources
                title = source.get('title', source.get('source', 'Documento sin título'))
                relevance = source.get('relevance_score', source.get('score', 0.0))
                excerpt = source.get('excerpt', source.get('text', ''))
                
                # Use the actual ID from the JSON response instead of sequential numbering
                source_id = source.get('id', '[?]')  # Get the actual ID like [2], [6], etc.
                
                print(f"{source_id} {title}")
                print(f"    Relevance: {relevance:.2f}")
                if excerpt and len(excerpt) > 0:
                    # Truncate excerpt if too long
                    excerpt_display = excerpt[:150] + "..." if len(excerpt) > 150 else excerpt
                    print(f"    Excerpt: {excerpt_display}")
                print()
        
        # === STRUCTURED INFORMATION SECTION ===
        if hasattr(response, 'metadata'):
            # Key Points
            key_points = response.metadata.get('key_points', [])
            if key_points:
                print(f"{'='*60}")
                print(f"🔑 KEY POINTS")
                print(f"{'='*60}")
                for i, point in enumerate(key_points, 1):
                    print(f"{i}. {point}")
                print()
            
            # Follow-up Questions
            follow_ups = response.metadata.get('follow_up_questions', [])
            if follow_ups:
                print(f"{'='*60}")
                print(f"❓ FOLLOW-UP QUESTIONS")
                print(f"{'='*60}")
                for i, question in enumerate(follow_ups, 1):
                    print(f"{i}. {question}")
                print()
            
            # Related Topics
            related_topics = response.metadata.get('related_topics', [])
            if related_topics:
                print(f"{'='*60}")
                print(f"🔗 RELATED TOPICS")
                print(f"{'='*60}")
                for i, topic in enumerate(related_topics, 1):
                    print(f"{i}. {topic}")
                print()
            
            # Warnings
            warnings = response.metadata.get('warnings', [])
            if warnings:
                print(f"{'='*60}")
                print(f"⚠️  WARNINGS")
                print(f"{'='*60}")
                for warning in warnings:
                    print(f"• {warning}")
                print()
        
        # === EXECUTION METADATA SECTION ===
        print(f"{'='*60}")
        print(f"⚙️  EXECUTION METADATA")
        print(f"{'='*60}")
        
        # Basic execution info - KEEP
        print(f"⏱️  Execution Time: {response.execution_time:.2f}s")
        print(f"📱 Application: {response.metadata.get('application', {}).get('name', 'Unknown')}")
        
        # Model info - POPULATE with configured model from YAML
        configured_model = self.config_manager.config.get('bedrock', {}).get('llm_model', 'Unknown')
        print(f"🤖 Model: {configured_model}")
        
        # Search and reasoning info
        tools_used = response.metadata.get('tools_used', [])
        results_found = response.metadata.get('results_found', 0)
        
        print(f"🔧 Tools Used: {', '.join(tools_used) if tools_used else 'None'}")
        print(f"📊 Results Found: {results_found}")
        
        # Memory and reasoning confidence (from RAG system components)
        memory_confidence = response.metadata.get('memory_confidence', 0)
        reasoning_confidence = response.metadata.get('reasoning_confidence', 0)
        
        if memory_confidence:
            print(f"🧠 Memory Confidence: {memory_confidence:.2f}")
        if reasoning_confidence:
            print(f"🧠 Reasoning Confidence: {reasoning_confidence:.2f}")
        
        # Cache metrics if available
        if hasattr(response, 'metadata') and response.metadata.get('structured_response'):
            structured_data = response.metadata['structured_response']
            metadata = structured_data.get('metadata', {})
            cache_metrics = metadata.get('cache_metrics')
            
            if cache_metrics:
                print(f"\n💾 Cache Metrics:")
                cache_hit = cache_metrics.get('cache_hit', False)
                cache_tokens = cache_metrics.get('cache_tokens', 0)
                print(f"   Cache Hit: {'Yes' if cache_hit else 'No'}")
                if cache_tokens:
                    print(f"   Cache Tokens: {cache_tokens:,}")
        
        print(f"{'='*60}")


def print_session_info(agent: SessionAwareMultiAppAgent):
    """Print current session information"""
    session_info = agent.get_session_info()
    if session_info:
        print(f"\n📋 Current Session Information:")
        print(f"   Session ID: {session_info['session_id']}")
        print(f"   User ID: {session_info.get('user_id', 'Anonymous')}")
        print(f"   Application: {session_info['app_name']}")
        print(f"   Created: {session_info['created_at']}")
        print(f"   Last Activity: {session_info['last_activity']}")


def print_user_sessions(agent: SessionAwareMultiAppAgent):
    """Print all sessions for the current user"""
    sessions = agent.list_user_sessions()
    if sessions:
        print(f"\n📚 Your Sessions in {agent.app_name}:")
        print("-" * 50)
        for session in sessions:
            current_marker = " (CURRENT)" if session['is_current'] else ""
            print(f"🔹 {session['session_id']}{current_marker}")
            print(f"   Created: {session['created_at']}")
            print(f"   Last Activity: {session['last_activity']}")
            print()
    else:
        print(f"\n📚 No other sessions found for user: {agent.user_id or 'Anonymous'}")


def print_session_stats():
    """Print global session statistics"""
    session_manager = get_session_manager()
    stats = session_manager.get_session_stats()
    
    print(f"\n📊 Global Session Statistics:")
    print(f"   Total Active Sessions: {stats['total_sessions']}")
    
    if stats['sessions_by_app']:
        print(f"   Sessions by Application:")
        for app, count in stats['sessions_by_app'].items():
            print(f"     - {app}: {count}")
    
    if stats['sessions_by_user']:
        print(f"   Sessions by User:")
        for user, count in stats['sessions_by_user'].items():
            print(f"     - {user}: {count}")


def main():
    """Main function for session-aware multi-application chat interface."""
    parser = argparse.ArgumentParser(
        description="Multi-Application Conversational RAG Chat Interface with Sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 multi_app_chat_with_sessions.py --app sap --user john_doe
  python3 multi_app_chat_with_sessions.py --app darwin --session session_darwin_abc123
  python3 multi_app_chat_with_sessions.py --list-apps
  python3 multi_app_chat_with_sessions.py --session-stats
        """
    )
    
    parser.add_argument(
        '--app', '-a',
        type=str,
        help='Application name to use (e.g., sap, darwin)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/multi_app_config.yaml',
        help='Path to multi-application configuration file'
    )
    
    parser.add_argument(
        '--user', '-u',
        type=str,
        help='User ID for session management'
    )
    
    parser.add_argument(
        '--session', '-s',
        type=str,
        help='Existing session ID to resume'
    )
    
    parser.add_argument(
        '--max-results', '-m',
        type=int,
        default=8,
        help='Maximum number of search results (default: 8)'
    )
    
    parser.add_argument(
        '--list-apps', '-l',
        action='store_true',
        help='List available applications and exit'
    )
    
    parser.add_argument(
        '--session-stats',
        action='store_true',
        help='Show session statistics and exit'
    )
    
    parser.add_argument(
        '--cleanup-sessions',
        action='store_true',
        help='Clean up expired sessions and exit'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration manager
        config_manager = MultiAppConfigManager(args.config)
        
        # Handle list applications
        if args.list_apps:
            from scripts.multi_app_chat import print_application_info
            print_application_info(config_manager)
            return
        
        # Handle session statistics
        if args.session_stats:
            print_session_stats()
            return
        
        # Handle session cleanup
        if args.cleanup_sessions:
            session_manager = get_session_manager()
            cleaned = session_manager.cleanup_expired_sessions()
            print(f"🧹 Cleaned up {cleaned} expired sessions")
            return
        
        # Initialize session-aware agent
        agent = SessionAwareMultiAppAgent(
            app_name=args.app,
            config_path=args.config,
            session_id=args.session,
            user_id=args.user
        )
        
        print(f"\n💬 Session-Aware Multi-Application RAG Chat Interface")
        print(f"Type 'quit', 'exit', or 'bye' to end the conversation")
        print(f"Type 'help' for available commands")
        print("=" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n[{agent.application_info['name']}:{agent.session_id[-8:]}] You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\n📖 Available commands:")
                    print("  help - Show this help message")
                    print("  quit/exit/bye - End the conversation")
                    print("  switch <app_name> - Switch to different application")
                    print("  session info - Show current session information")
                    print("  session list - List your sessions")
                    print("  session switch <session_id> - Switch to different session")
                    print("  session stats - Show global session statistics")
                    print("  apps - List available applications")
                    continue
                
                if user_input.lower().startswith('switch '):
                    new_app = user_input[7:].strip()
                    try:
                        agent = SessionAwareMultiAppAgent(
                            app_name=new_app,
                            config_path=args.config,
                            user_id=args.user
                        )
                        print(f"✅ Switched to application: {agent.application_info['name']}")
                        print(f"🆔 New session: {agent.session_id}")
                    except ValueError as e:
                        print(f"❌ Error switching application: {e}")
                    continue
                
                if user_input.lower() == 'session info':
                    print_session_info(agent)
                    continue
                
                if user_input.lower() == 'session list':
                    print_user_sessions(agent)
                    continue
                
                if user_input.lower().startswith('session switch '):
                    session_id = user_input[15:].strip()
                    if agent.switch_session(session_id):
                        print(f"✅ Switched to session: {session_id}")
                    else:
                        print(f"❌ Could not switch to session: {session_id}")
                    continue
                
                if user_input.lower() == 'session stats':
                    print_session_stats()
                    continue
                
                if user_input.lower() == 'apps':
                    from scripts.multi_app_chat import print_application_info
                    print_application_info(config_manager)
                    continue
                
                # Process query
                print(f"\n🤔 Processing query...")
                
                response = agent.process_query(
                    query=user_input,
                    max_results=args.max_results
                )
                
                # Handle response with comprehensive structured display
                agent._display_comprehensive_response(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
