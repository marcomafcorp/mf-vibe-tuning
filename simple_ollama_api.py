from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({'status': 'ok', 'message': 'Ollama API Backend is running'})

@app.route('/api/test/models', methods=['GET'])
def test_models():
    """Test endpoint that returns dummy models"""
    dummy_models = [
        {
            'id': 'llama2:7b',
            'hash': 'test123456',
            'size': '3.8 GB',
            'modified': '2 days ago'
        },
        {
            'id': 'mistral:7b',
            'hash': 'test789012',
            'size': '4.1 GB',
            'modified': '1 week ago'
        }
    ]
    return jsonify({
        'status': 'success',
        'models': dummy_models
    })

if __name__ == '__main__':
    print("Starting Simple Ollama API server on port 5001...")
    app.run(port=5001, debug=True, host='0.0.0.0')
