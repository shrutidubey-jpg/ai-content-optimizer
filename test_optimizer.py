#!/usr/bin/env python3
"""
Simple test script to demonstrate the Content Optimizer functionality
"""

import sys
import os

# Add the current directory to the path to import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test the optimizer with basic functionality (without external dependencies)"""
    
    print("Testing Content SEO Optimizer (Basic Mode)")
    print("=" * 50)
    
    # Import our optimizer
    try:
        from content_optimizer import ContentOptimizer
        print("✓ ContentOptimizer imported successfully")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return
    
    # Sample content
    sample_content = """
    Digital Marketing Strategies for Modern Businesses
    
    In todays digital landscape, businesses need to adopt effective digital marketing strategies to stay competitive. Digital marketing has become a crucial component for business success. This article will explore various digital marketing techniques that can help your business grow.
    
    Search engine optimization is one of the most important aspects of digital marketing. SEO helps businesses improve their online visibility and attract more customers. By optimizing your content for search engines, you can increase organic traffic to your website.
    
    Social media marketing is another powerful tool for businesses. Platforms like Facebook, Instagram, and Twitter allow companies to connect with their audience directly. Creating engaging content on social media can help build brand awareness and drive traffic to your website.
    """
    
    # Initialize optimizer
    try:
        optimizer = ContentOptimizer()
        print("✓ ContentOptimizer initialized")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return
    
    # Test optimization
    target_keywords = ["digital marketing", "SEO", "social media"]
    
    try:
        result = optimizer.optimize_content(sample_content, target_keywords)
        print("✓ Content optimization completed")
        
        # Display results
        print("\n" + "=" * 50)
        print("OPTIMIZATION RESULTS")
        print("=" * 50)
        print(f"Word Count: {result.word_count}")
        print(f"SEO Score: {result.seo_score:.1f}/100")
        print(f"Readability Score: {result.readability_score:.1f}/100")
        print(f"Grammar Issues Found: {len(result.grammar_issues)}")
        print(f"Spelling Issues Found: {len(result.spelling_issues)}")
        
        if result.keyword_density:
            print("\nKeyword Density Analysis:")
            for keyword, density in sorted(result.keyword_density.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]:
                print(f"  • {keyword}: {density:.2f}%")
        
        if result.suggestions:
            print("\nOptimization Suggestions:")
            for i, suggestion in enumerate(result.suggestions[:5], 1):
                print(f"  {i}. {suggestion}")
        
        if result.grammar_issues:
            print(f"\nGrammar Issues Sample ({len(result.grammar_issues)} total):")
            for issue in result.grammar_issues[:3]:
                print(f"  • {issue['message']}")
        
        if result.spelling_issues:
            print(f"\nSpelling Issues Sample ({len(result.spelling_issues)} total):")
            for issue in result.spelling_issues[:3]:
                print(f"  • {issue.get('message', 'Spelling issue detected')}")
        
        print("\n" + "=" * 50)
        print("OPTIMIZED CONTENT PREVIEW")
        print("=" * 50)
        print(result.optimized_content[:500] + "..." if len(result.optimized_content) > 500 else result.optimized_content)
        
        print("\n✓ Test completed successfully!")
        
    except Exception as e:
        print(f"✗ Optimization failed: {e}")
        import traceback
        traceback.print_exc()

def test_file_operations():
    """Test file reading and writing operations"""
    print("\n" + "=" * 50)
    print("TESTING FILE OPERATIONS")
    print("=" * 50)
    
    try:
        # Test reading the example file
        with open('example_content.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✓ Successfully read example_content.txt ({len(content)} characters)")
        
        # Test writing optimization results
        test_result = "This is a test optimization result."
        with open('test_output.txt', 'w', encoding='utf-8') as f:
            f.write(test_result)
        print("✓ Successfully wrote test_output.txt")
        
        # Clean up
        if os.path.exists('test_output.txt'):
            os.remove('test_output.txt')
            print("✓ Cleaned up test files")
            
    except Exception as e:
        print(f"✗ File operations failed: {e}")

def check_dependencies():
    """Check which dependencies are available"""
    print("\n" + "=" * 50)
    print("DEPENDENCY CHECK")
    print("=" * 50)
    
    dependencies = {
        'nltk': 'Natural Language Processing',
        'textstat': 'Readability Statistics',
        'language_tool_python': 'Grammar Checking'
    }
    
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"✓ {package} - {description}")
        except ImportError:
            print(f"✗ {package} - {description} (Not installed)")
    
    print("\nNote: The optimizer works with basic functionality even without all dependencies.")
    print("For full features, install dependencies using:")
    print("pip install nltk textstat language-tool-python")

if __name__ == "__main__":
    print("Content SEO Optimizer - Test Suite")
    print("=" * 60)
    
    # Run tests
    check_dependencies()
    test_file_operations()
    test_basic_functionality()
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("To run the full optimizer, use:")
    print("python content_optimizer.py example_content.txt -k \"digital marketing\" \"SEO\"")
    print("=" * 60)