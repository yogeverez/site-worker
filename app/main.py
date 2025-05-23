print("---- ENHANCED PYTHON MAIN.PY STARTING ----", flush=True)
import os, base64, json, time
import logging
import asyncio
import sys
from flask import Flask, request

# Import enhanced modules implementing research recommendations
from openai import OpenAI
from tools import do_comprehensive_research, generate_enhanced_site_content

# Configure comprehensive logging
logger = logging.getLogger(__name__)

app = Flask(__name__)

class EnhancedPubSubHandler:
    """Enhanced handler implementing research recommendations for robust processing."""
    
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        """Setup comprehensive logging for the research workflow."""
        # Create formatters for different log levels
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(detailed_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers.clear()
        root_logger.addHandler(console_handler)
        root_logger.propagate = False

    async def validate_request_parameters(self, payload: dict, logger: logging.Logger) -> tuple[bool, str, dict]:
        """Enhanced parameter validation with detailed feedback."""
        uid = payload.get("uid")
        mode = payload.get("mode")
        languages = payload.get("languages", [])
        timestamp = payload.get("timestamp")
        
        # Enhanced validation logic
        validation_result = {
            "uid_valid": bool(uid and uid.strip()),
            "mode_valid": mode in ["research", "generate", "full"],
            "languages_valid": isinstance(languages, list),
            "timestamp_valid": timestamp is None or isinstance(timestamp, (int, float))
        }
        
        if not validation_result["uid_valid"]:
            return False, "Missing or invalid UID parameter", validation_result
        
        if not validation_result["mode_valid"]:
            return False, f"Invalid mode '{mode}'. Must be 'research', 'generate', or 'full'", validation_result
            
        if not validation_result["languages_valid"]:
            return False, "Languages parameter must be a list", validation_result
        
        # Special validation for generate mode
        if mode == "generate" and not languages:
            return False, "Generate mode requires at least one language specified", validation_result
        
        logger.info(f"Request validation passed: {validation_result}")
        return True, "Parameters valid", validation_result

    async def execute_research_phase(self, uid: str, timestamp: int, logger: logging.Logger) -> dict:
        """Execute the research phase with comprehensive tracking."""
        research_start = time.time()
        logger.info(f"=== RESEARCH PHASE STARTING for {uid} ===")
        
        try:
            await do_comprehensive_research(uid, timestamp=timestamp, parent_logger=logger)
            research_duration = time.time() - research_start
            
            result = {
                "phase": "research",
                "status": "completed",
                "duration_seconds": round(research_duration, 2),
                "timestamp": timestamp
            }
            
            logger.info(f"=== RESEARCH PHASE COMPLETED for {uid} in {research_duration:.2f}s ===")
            return result
            
        except Exception as e:
            research_duration = time.time() - research_start
            logger.error(f"=== RESEARCH PHASE FAILED for {uid} after {research_duration:.2f}s: {e} ===", exc_info=True)
            
            return {
                "phase": "research",
                "status": "failed", 
                "duration_seconds": round(research_duration, 2),
                "error": str(e),
                "timestamp": timestamp
            }

    async def execute_generation_phase(self, uid: str, languages: list, timestamp: int, logger: logging.Logger) -> dict:
        """Execute the content generation phase with research integration."""
        generation_start = time.time()
        logger.info(f"=== CONTENT GENERATION PHASE STARTING for {uid} with languages: {languages} ===")
        
        try:
            await generate_enhanced_site_content(uid, languages, timestamp=timestamp, parent_logger=logger)
            generation_duration = time.time() - generation_start
            
            result = {
                "phase": "generation",
                "status": "completed",
                "duration_seconds": round(generation_duration, 2),
                "languages": languages,
                "timestamp": timestamp
            }
            
            logger.info(f"=== CONTENT GENERATION PHASE COMPLETED for {uid} in {generation_duration:.2f}s ===")
            return result
            
        except Exception as e:
            generation_duration = time.time() - generation_start
            logger.error(f"=== CONTENT GENERATION PHASE FAILED for {uid} after {generation_duration:.2f}s: {e} ===", exc_info=True)
            
            return {
                "phase": "generation",
                "status": "failed",
                "duration_seconds": round(generation_duration, 2),
                "error": str(e),
                "languages": languages,
                "timestamp": timestamp
            }

    async def process_request(self, payload: dict) -> tuple[str, int]:
        """
        Enhanced request processing implementing researcher-first workflow.
        Returns (response_message, status_code).
        """
        # Create request-specific logger
        request_id = payload.get("uid", "unknown")
        request_logger = logging.getLogger(f"{__name__}.request_{request_id}")
        
        total_start_time = time.time()
        
        try:
            request_logger.info(f"🚀 Processing enhanced site generation request for {request_id}")
            
            # Enhanced parameter validation
            is_valid, validation_message, validation_details = await self.validate_request_parameters(payload, request_logger)
            if not is_valid:
                request_logger.warning(f"❌ Request validation failed: {validation_message}")
                return f"Bad Request: {validation_message}", 400
            
            # Extract validated parameters
            uid = payload["uid"]
            mode = payload["mode"]
            languages = payload.get("languages", [])
            timestamp = payload.get("timestamp")
            
            request_logger.info(f"✅ Request validated - UID: {uid}, Mode: {mode}, Languages: {languages}")
            
            # Execution tracking
            execution_results = []
            
            # RESEARCHER-FIRST APPROACH: Execute research phase first for modes that include it
            if mode in ["research", "full"]:
                research_result = await self.execute_research_phase(uid, timestamp, request_logger)
                execution_results.append(research_result)
                
                # If research fails and mode is "full", we can still proceed with generation
                # but log this as a degraded scenario
                if research_result["status"] == "failed" and mode == "full":
                    request_logger.warning(f"⚠️ Research phase failed but continuing with generation for {uid}")
            
            # Execute content generation phase
            if mode in ["generate", "full"]:
                # Ensure we have languages for generation
                if not languages:
                    request_logger.warning(f"⚠️ No languages specified for generation mode, defaulting to ['en']")
                    languages = ["en"]
                
                generation_result = await self.execute_generation_phase(uid, languages, timestamp, request_logger)
                execution_results.append(generation_result)
            
            # Calculate total processing time and analyze results
            total_duration = time.time() - total_start_time
            
            # Determine overall success
            failed_phases = [r for r in execution_results if r["status"] == "failed"]
            successful_phases = [r for r in execution_results if r["status"] == "completed"]
            
            if failed_phases:
                request_logger.error(f"❌ Job completed with {len(failed_phases)} failed phases in {total_duration:.2f}s")
                # Return 200 but log the issues - Pub/Sub should not retry for application-level failures
                return "", 200
            else:
                request_logger.info(f"✅ Job completed successfully - All {len(successful_phases)} phases completed in {total_duration:.2f}s")
                return "", 204
                
        except Exception as e:
            total_duration = time.time() - total_start_time
            request_logger.error(f"💥 Unhandled error in request processing after {total_duration:.2f}s: {e}", exc_info=True)
            return "Internal Server Error", 500

# Initialize enhanced handler
enhanced_handler = EnhancedPubSubHandler()

@app.route("/", methods=["POST"])
def pubsub_handler():
    """Enhanced HTTP endpoint for Pub/Sub push messages implementing research workflow."""
    try:
        # Parse Pub/Sub envelope
        envelope = request.get_json(silent=True)
        if not envelope or "message" not in envelope:
            logger.warning("Bad Request: no Pub/Sub message received")
            return "Bad Request: no Pub/Sub message received", 400
        
        msg = envelope["message"]
        message_id = msg.get('messageId', 'unknown')
        logger.info(f"📨 Received Pub/Sub message: {message_id}")

        # Decode message payload
        data = msg.get("data")
        if data:
            try:
                payload = json.loads(base64.b64decode(data).decode("utf-8"))
                logger.info(f"📋 Successfully decoded message payload for message {message_id}")
            except Exception as e:
                logger.error(f"Failed to decode message data for {message_id}: {e}")
                return "Bad Request: invalid message data", 400
        else:
            logger.warning(f"No data in message payload for {message_id}")
            payload = {}

        # Process request using enhanced handler
        response_message, status_code = asyncio.run(enhanced_handler.process_request(payload))
        return response_message, status_code

    except Exception as e:
        logger.error(f"Unhandled error in pubsub_handler: {e}", exc_info=True)
        return "Internal Server Error", 500

@app.route("/health", methods=["GET"])
def enhanced_health_check():
    """Enhanced health check with comprehensive system status."""
    try:
        # Check OpenAI API configuration
        openai_configured = bool(os.getenv("OPENAI_API_KEY", ""))
        
        # Check SerpAPI configuration  
        serpapi_configured = bool(os.getenv("SERPAPI_KEY", ""))
        
        # Check environment configuration
        environment = os.getenv("ENVIRONMENT", "unknown")
        bypass_serpapi = os.getenv("BYPASS_SERPAPI_RESEARCH", "false").lower() == "true"
        
        # Test basic functionality
        try:
            from tools import get_db
            db_connection = get_db() is not None
        except Exception as e:
            logger.warning(f"Database connection test failed: {e}")
            db_connection = False
        
        # Comprehensive health status
        health_status = {
            "status": "healthy",
            "timestamp": int(time.time()),
            "version": "enhanced_research_implementation",
            "environment": environment,
            "configuration": {
                "openai_api_configured": openai_configured,
                "serpapi_configured": serpapi_configured, 
                "bypass_serpapi_research": bypass_serpapi,
                "database_connection": db_connection
            },
            "features": {
                "researcher_first_workflow": True,
                "enhanced_error_handling": True,
                "comprehensive_logging": True,
                "multilingual_support": True,
                "source_tracking": True
            },
            "workflow_phases": ["research", "generation", "translation", "validation"]
        }
        
        # Determine overall health
        critical_issues = []
        if not openai_configured:
            critical_issues.append("OpenAI API key not configured")
        if not db_connection:
            critical_issues.append("Database connection failed")
            
        if critical_issues:
            health_status["status"] = "degraded"
            health_status["issues"] = critical_issues
            logger.warning(f"Health check: Service degraded - {critical_issues}")
            return json.dumps(health_status), 503, {"Content-Type": "application/json"}
        
        logger.info("Health check: Enhanced service is healthy")
        return json.dumps(health_status), 200, {"Content-Type": "application/json"}
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        error_status = {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": int(time.time())
        }
        return json.dumps(error_status), 500, {"Content-Type": "application/json"}

@app.route("/research-status/<uid>", methods=["GET"])
def research_status(uid: str):
    """New endpoint to check research status for a specific user."""
    try:
        from tools import get_db
        db = get_db()
        
        # Get research manifest
        manifest_ref = db.collection("research").document(uid).collection("summary").document("manifest")
        manifest_doc = manifest_ref.get()
        
        if not manifest_doc.exists:
            return json.dumps({"uid": uid, "status": "not_found"}), 404, {"Content-Type": "application/json"}
        
        manifest_data = manifest_doc.to_dict()
        
        # Get source count
        sources_ref = db.collection("research").document(uid).collection("sources")
        source_count = len(list(sources_ref.limit(1).stream()))
        
        status_response = {
            "uid": uid,
            "research_status": manifest_data.get("status", "unknown"),
            "sources_found": manifest_data.get("total_sources_found", 0),
            "sources_saved": manifest_data.get("sources_saved_successfully", 0),
            "research_duration": manifest_data.get("research_duration_seconds", 0),
            "timestamp": manifest_data.get("timestamp"),
            "template_type": manifest_data.get("template_type"),
            "query_count": len(manifest_data.get("search_queries_generated", []))
        }
        
        return json.dumps(status_response), 200, {"Content-Type": "application/json"}
        
    except Exception as e:
        logger.error(f"Error checking research status for {uid}: {e}")
        return json.dumps({"error": str(e)}), 500, {"Content-Type": "application/json"}

# Development and debugging endpoints
if os.getenv("ENVIRONMENT") == "development":
    @app.route("/debug/trigger-research/<uid>", methods=["POST"])
    def debug_trigger_research(uid: str):
        """Development endpoint to manually trigger research for a user."""
        try:
            timestamp = int(time.time())
            payload = {
                "uid": uid,
                "mode": "research", 
                "timestamp": timestamp
            }
            
            response_message, status_code = asyncio.run(enhanced_handler.process_request(payload))
            return f"Research triggered for {uid}", status_code
            
        except Exception as e:
            logger.error(f"Debug research trigger failed for {uid}: {e}")
            return f"Error: {str(e)}", 500

# Only needed if running the Flask dev server
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    debug_mode = os.getenv("ENVIRONMENT") == "development"
    
    logger.info(f"🚀 Starting enhanced site generation service on port {port} (debug: {debug_mode})")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)