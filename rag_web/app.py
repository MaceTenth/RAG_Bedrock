"""
RAG Web Application - Main Entry Point

A Flask-based RAG application using Amazon Bedrock Knowledge Base
for retrieval and Google Gemini or Bedrock Claude for LLM generation.
"""
from flask import Flask, render_template

from config import KNOWLEDGE_BASE_ID, S3_BUCKET_NAME, USE_BEDROCK_LLM, BEDROCK_MODEL_ID, GEMINI_MODEL, PORT, DEBUG
from routes import api


def create_app():
    """Application factory."""
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(api)
    
    # Home route
    @app.route("/")
    def index():
        """Serve the main page."""
        return render_template("index.html")
    
    return app


def main():
    """Run the application."""
    # Display startup info
    print("🚀 Starting RAG application")
    print(f"   Knowledge Base ID: {KNOWLEDGE_BASE_ID}")
    print(f"   S3 Bucket: {S3_BUCKET_NAME}")
    
    if USE_BEDROCK_LLM:
        print(f"   LLM: Bedrock {BEDROCK_MODEL_ID}")
    else:
        print(f"   LLM: Gemini {GEMINI_MODEL}")
    
    # Create and run app
    app = create_app()
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)


# Create app instance for Flask CLI and imports
app = create_app()

if __name__ == "__main__":
    main()
