from flask import Blueprint, jsonify

# Create a Blueprint for health routes
health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Simple health check endpoint to verify API is accessible"""
    return jsonify({
        'status': 'ok',
        'message': 'API server is running'
    }), 200