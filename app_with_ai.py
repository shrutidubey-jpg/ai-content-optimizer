#!/usr/bin/env python3
"""
Content SEO Optimizer with AI Grammar Agent - Flask Web Application
Enhanced web interface with intelligent grammar correction
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import json
import os
import tempfile
from datetime import datetime
import uuid

# Import our enhanced optimizer with AI
try:
    from enhanced_optimizer_with_ai import EnhancedOptimizerWithAI
    AI_AVAILABLE = True
    print("✅ AI Grammar Agent loaded successfully!")
except ImportError:
    from enhanced_optimizer import EnhancedContentOptimizer as EnhancedOptimizerWithAI
    AI_AVAILABLE = False
    print("⚠️ AI Grammar Agent not available, using standard optimizer")

app = Flask(__name__)
CORS(app)

# Initialize the enhanced optimizer with AI
optimizer = EnhancedOptimizerWithAI()

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """Main application page with AI features"""
    # For deployment, serve a simple HTML page without template dependency
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Content SEO Optimizer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #667eea;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        .alert {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: 600;
        }
        .content-area {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        @media (max-width: 768px) {
            .content-area { grid-template-columns: 1fr; }
        }
        .section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            border: 2px solid #e9ecef;
        }
        .section h3 {
            color: #495057;
            margin-bottom: 15px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #495057;
        }
        .form-control {
            width: 100%;
            padding: 12px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            font-size: 1rem;
            font-family: inherit;
        }
        .form-control:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        textarea.form-control {
            height: 200px;
            resize: vertical;
            font-family: 'Consolas', monospace;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 25px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .btn:hover { transform: translateY(-2px); }
        .results {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            display: none;
        }
        .score-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .score-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 2px solid #e9ecef;
        }
        .score-card.excellent { border-color: #28a745; background: #d4edda; }
        .score-value {
            font-size: 2rem;
            font-weight: bold;
            color: #495057;
        }
        .score-label {
            font-size: 0.9rem;
            color: #6c757d;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Content SEO Optimizer</h1>
            <p>Professional content optimization with AI grammar correction</p>
            <div class="alert">
                ✨ AI Grammar Agent: ''' + ('✅ ACTIVE' if ''' + str(AI_AVAILABLE).lower() + ''' else '❌ OFFLINE') + '''
            </div>
        </div>
        
        <div class="content-area">
            <div class="section">
                <h3>📝 Content Input</h3>
                <div class="form-group">
                    <label>Your Content:</label>
                    <textarea id="contentInput" class="form-control" placeholder="Enter your content here for AI-powered SEO optimization...">Digital Marketing Strategies for Modern Business

In todays digital world,businesses need effective marketing strategies.Digital marketing digital marketing is important for success.This guide will help your business business grow.

SEO helps websites get more customers customers.You should optimize content content for search engines.</textarea>
                </div>
                <div class="form-group">
                    <label>Target Keywords (comma-separated):</label>
                    <input type="text" id="keywordsInput" class="form-control" placeholder="digital marketing, SEO, business growth" value="digital marketing, SEO optimization, business growth">
                </div>
                <div class="form-group">
                    <label>Writing Context:</label>
                    <select id="contextInput" class="form-control">
                        <option value="general">General</option>
                        <option value="business_writing">Business Writing</option>
                        <option value="academic_writing">Academic Writing</option>
                    </select>
                </div>
                <button class="btn" onclick="analyzeContent()">🔍 Analyze & Optimize with AI</button>
            </div>
            
            <div class="section">
                <h3>📊 AI Analysis Results</h3>
                <div class="score-grid">
                    <div class="score-card excellent" id="seoCard">
                        <div class="score-value" id="seoScore">85</div>
                        <div class="score-label">SEO Score</div>
                    </div>
                    <div class="score-card excellent" id="grammarCard">
                        <div class="score-value" id="grammarScore">92</div>
                        <div class="score-label">Grammar Score</div>
                    </div>
                    <div class="score-card excellent" id="readabilityCard">
                        <div class="score-value" id="readabilityScore">78</div>
                        <div class="score-label">Readability</div>
                    </div>
                    <div class="score-card excellent" id="issuesCard">
                        <div class="score-value" id="issuesFixed">8</div>
                        <div class="score-label">AI Fixes</div>
                    </div>
                </div>
                <p style="text-align: center; color: #666; margin-top: 15px;">
                    Click "Analyze & Optimize with AI" to process your content!
                </p>
            </div>
        </div>
        
        <div id="resultsSection" class="results">
            <h3>✨ Your AI-Optimized Content</h3>
            <textarea id="optimizedContent" class="form-control" readonly style="height: 300px;"></textarea>
            <button class="btn" onclick="copyContent()" style="margin-top: 10px;">📋 Copy Optimized Content</button>
        </div>
    </div>
    
    <script>
        function analyzeContent() {
            const content = document.getElementById('contentInput').value.trim();
            const keywords = document.getElementById('keywordsInput').value.trim();
            const context = document.getElementById('contextInput').value;
            
            if (!content) {
                alert('Please enter some content to analyze!');
                return;
            }
            
            // Show loading
            const btn = event.target;
            const originalText = btn.textContent;
            btn.textContent = '🤖 AI Processing...';
            btn.disabled = true;
            
            // Call the API
            fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    content: content,
                    keywords: keywords.split(',').map(k => k.trim()).filter(k => k),
                    context: context
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Update scores
                    document.getElementById('seoScore').textContent = Math.round(data.analysis.seo_score);
                    document.getElementById('grammarScore').textContent = Math.round(data.analysis.grammar_score);
                    document.getElementById('readabilityScore').textContent = Math.round(data.analysis.readability_score);
                    document.getElementById('issuesFixed').textContent = data.ai_enhancements.issues_fixed;
                    
                    // Show optimized content
                    document.getElementById('optimizedContent').value = data.optimized_content;
                    document.getElementById('resultsSection').style.display = 'block';
                    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
                    
                    alert('✅ AI optimization completed! Your content has been enhanced.');
                } else {
                    alert('❌ Error: ' + (data.error || 'Analysis failed'));
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('❌ Connection error. Please try again.');
            })
            .finally(() => {
                btn.textContent = originalText;
                btn.disabled = false;
            });
        }
        
        function copyContent() {
            const textarea = document.getElementById('optimizedContent');
            textarea.select();
            document.execCommand('copy');
            alert('✅ Optimized content copied to clipboard!');
        }
        
        // Auto-demo on load
        setTimeout(() => {
            if (document.getElementById('contentInput').value.trim()) {
                analyzeContent();
            }
        }, 2000);
    </script>
</body>
</html>
    '''

@app.route('/api/analyze', methods=['POST'])
def analyze_content():
    """Enhanced API endpoint with AI grammar analysis"""
    try:
        data = request.get_json()
        
        if not data or 'content' not in data:
            return jsonify({'error': 'No content provided'}), 400
        
        content = data['content']
        keywords = data.get('keywords', [])
        writing_context = data.get('context', 'general')  # New: writing context
        
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',') if k.strip()]
        
        print(f"🔍 Analyzing content with AI Grammar Agent...")
        print(f"   Content length: {len(content)} characters")
        print(f"   Keywords: {keywords}")
        print(f"   Context: {writing_context}")
        
        # Perform enhanced optimization with AI
        result = optimizer.optimize_content(content, keywords, writing_context)
        
        # Convert result to JSON-serializable format
        response_data = {
            'success': True,
            'ai_enabled': AI_AVAILABLE,
            'analysis': {
                'seo_score': result['optimized_analysis']['seo_score'],
                'readability_score': result['optimized_analysis']['readability_score'],
                'grammar_score': result['optimized_analysis'].get('grammar_score', 85),
                'word_count': result['optimized_analysis']['word_count'],
                'grammar_issues_count': len(result['optimized_analysis'].get('grammar_issues', [])),
                'spelling_issues_count': len(result['optimized_analysis'].get('spelling_issues', [])),
                'keyword_density': result['optimized_analysis']['keyword_density'],
                'suggestions': result['optimized_analysis']['suggestions'],
                'grammar_issues': result['optimized_analysis'].get('grammar_issues', [])[:10],
                'spelling_issues': result['optimized_analysis'].get('spelling_issues', [])[:10],
            },
            'optimized_content': result['optimized_content'],
            'improvements': result['improvements'],
            'original_analysis': {
                'seo_score': result['original_analysis']['seo_score'],
                'readability_score': result['original_analysis']['readability_score'],
                'grammar_score': result['original_analysis'].get('grammar_score', 70),
                'word_count': result['original_analysis']['word_count'],
                'grammar_issues_count': len(result['original_analysis'].get('grammar_issues', [])),
                'spelling_issues_count': len(result['original_analysis'].get('spelling_issues', [])),
            },
            'ai_enhancements': result.get('ai_enhancements', {
                'grammar_score_improvement': 0,
                'issues_fixed': 0,
                'text_quality_boost': 0
            }),
            'grammar_report': result.get('grammar_report', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Analysis completed!")
        print(f"   SEO Score: {result['original_analysis']['seo_score']:.1f} → {result['optimized_analysis']['seo_score']:.1f}")
        if AI_AVAILABLE:
            print(f"   Grammar Score: {result['original_analysis'].get('grammar_score', 0):.1f} → {result['optimized_analysis'].get('grammar_score', 0):.1f}")
            print(f"   AI Issues Fixed: {result.get('ai_enhancements', {}).get('issues_fixed', 0)}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/grammar-check', methods=['POST'])
def grammar_check_only():
    """Dedicated endpoint for AI grammar checking"""
    try:
        data = request.get_json()
        content = data.get('content', '')
        context = data.get('context', 'general')
        
        if not AI_AVAILABLE:
            return jsonify({'error': 'AI Grammar Agent not available'}), 400
        
        # Get grammar report from AI agent
        grammar_report = optimizer.grammar_ai.generate_grammar_report(content, context)
        
        response_data = {
            'success': True,
            'grammar_score': grammar_report['grammar_score']['grammar_score'],
            'issues_found': grammar_report['issues_found'],
            'corrected_text': grammar_report['corrected_text'],
            'corrections_applied': grammar_report['corrections_applied'],
            'improvement_summary': grammar_report['improvement_summary']
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Grammar check failed: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload with AI enhancement"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file content
        content = file.read().decode('utf-8')
        keywords = request.form.get('keywords', '').split(',')
        keywords = [k.strip() for k in keywords if k.strip()]
        context = request.form.get('context', 'general')
        
        # Analyze content with AI
        result = optimizer.optimize_content(content, keywords, context)
        
        # Save optimized content
        output_filename = f"ai_optimized_{uuid.uuid4().hex[:8]}.txt"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['optimized_content'])
        
        response_data = {
            'success': True,
            'ai_enabled': AI_AVAILABLE,
            'analysis': {
                'seo_score': result['optimized_analysis']['seo_score'],
                'readability_score': result['optimized_analysis']['readability_score'],
                'grammar_score': result['optimized_analysis'].get('grammar_score', 85),
                'word_count': result['optimized_analysis']['word_count'],
                'grammar_issues_count': len(result['optimized_analysis'].get('grammar_issues', [])),
                'spelling_issues_count': len(result['optimized_analysis'].get('spelling_issues', [])),
                'keyword_density': result['optimized_analysis']['keyword_density'],
                'suggestions': result['optimized_analysis']['suggestions'],
            },
            'improvements': result['improvements'],
            'ai_enhancements': result.get('ai_enhancements', {}),
            'download_url': f'/api/download/{output_filename}',
            'original_filename': file.filename
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download optimized content file"""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=f"ai_optimized_{filename}")
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """Health check with AI status"""
    return jsonify({
        'status': 'healthy',
        'version': '2.0-ai',
        'ai_grammar_agent': AI_AVAILABLE,
        'features': [
            'Advanced Grammar Correction',
            'SEO Optimization',
            'Keyword Analysis',
            'Readability Enhancement',
            'Context-Aware Analysis'
        ],
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 STARTING ENHANCED CONTENT SEO OPTIMIZER WITH AI")
    print("=" * 60)
    print("🤖 AI Grammar Agent: " + ("✅ ENABLED" if AI_AVAILABLE else "❌ DISABLED"))
    print("📊 Features loaded:")
    print("  ✅ Advanced grammar correction with confidence scoring")
    print("  ✅ Context-aware writing analysis (business, academic, general)")
    print("  ✅ Intelligent keyword optimization")
    print("  ✅ Real-time content enhancement")
    print("  ✅ Comprehensive SEO analysis")
    print("  ✅ File upload/download with AI processing")
    print("  ✅ RESTful API with AI endpoints")
    print("\n🌐 Access the enhanced application at: http://localhost:5000")
    print("🤖 AI Grammar Check endpoint: http://localhost:5000/api/grammar-check")
    print("📚 API Health Check: http://localhost:5000/api/health")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)