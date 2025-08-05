#!/usr/bin/env python3
"""
Content SEO Optimizer - A comprehensive tool for content optimization
Includes grammar checking, spell checking, keyword optimization, and SEO analysis
"""

import re
import json
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import math

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.pos_tag import pos_tag
    nltk_available = True
except ImportError:
    nltk_available = False
    print("NLTK not available. Installing required packages...")

try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    textstat_available = True
except ImportError:
    textstat_available = False

try:
    import language_tool_python
    languagetool_available = True
except ImportError:
    languagetool_available = False

@dataclass
class OptimizationResult:
    original_content: str
    optimized_content: str
    grammar_issues: List[Dict]
    spelling_issues: List[Dict] 
    seo_score: float
    keyword_density: Dict[str, float]
    readability_score: float
    word_count: int
    suggestions: List[str]

class ContentOptimizer:
    def __init__(self):
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        self.grammar_tool = None
        self.lemmatizer = None
        
        # Initialize NLTK components if available
        if nltk_available:
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
                nltk.data.find('corpora/wordnet')
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                print("Downloading required NLTK data...")
                nltk.download('punkt')
                nltk.download('stopwords')
                nltk.download('wordnet')
                nltk.download('averaged_perceptron_tagger')
            
            self.stop_words.update(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        
        # Initialize grammar tool if available
        if languagetool_available:
            try:
                self.grammar_tool = language_tool_python.LanguageTool('en-US')
            except Exception as e:
                print(f"Could not initialize LanguageTool: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()
    
    def check_grammar_and_spelling(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """Check for grammar and spelling issues"""
        grammar_issues = []
        spelling_issues = []
        
        if self.grammar_tool:
            try:
                matches = self.grammar_tool.check(text)
                for match in matches:
                    issue = {
                        'message': match.message,
                        'context': match.context,
                        'offset': match.offset,
                        'length': match.errorLength,
                        'suggestions': match.replacements[:3] if match.replacements else [],
                        'rule_id': match.ruleId,
                        'category': match.category
                    }
                    
                    if 'SPELL' in match.ruleId or 'MORFOLOGIK' in match.ruleId:
                        spelling_issues.append(issue)
                    else:
                        grammar_issues.append(issue)
            except Exception as e:
                print(f"Grammar check error: {e}")
        else:
            # Basic spell check using word patterns
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            for word in words:
                if len(word) > 2 and word.lower() not in self.stop_words:
                    # Simple heuristic for potential spelling errors
                    if re.search(r'(.)\1{2,}', word) or len(set(word)) < len(word) * 0.5:
                        spelling_issues.append({
                            'message': f'Possible spelling error: {word}',
                            'context': word,
                            'suggestions': [],
                            'word': word
                        })
        
        return grammar_issues, spelling_issues
    
    def extract_keywords(self, text: str, target_keywords: List[str] = None) -> Dict[str, float]:
        """Extract and analyze keyword density"""
        text_lower = text.lower()
        
        if nltk_available:
            words = word_tokenize(text_lower)
            words = [self.lemmatizer.lemmatize(word) for word in words if word.isalpha()]
        else:
            words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
        
        # Filter out stop words
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        total_words = len(words)
        word_counts = Counter(words)
        
        # Calculate keyword density
        keyword_density = {}
        
        if target_keywords:
            for keyword in target_keywords:
                keyword_lower = keyword.lower()
                count = text_lower.count(keyword_lower)
                density = (count / total_words) * 100 if total_words > 0 else 0
                keyword_density[keyword] = density
        
        # Get top keywords naturally occurring
        top_keywords = word_counts.most_common(10)
        for word, count in top_keywords:
            if word not in keyword_density:
                density = (count / total_words) * 100 if total_words > 0 else 0
                keyword_density[word] = density
        
        return keyword_density
    
    def calculate_readability(self, text: str) -> float:
        """Calculate readability score"""
        if textstat_available:
            return flesch_reading_ease(text)
        
        # Simple readability calculation
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(re.findall(r'\b\w+\b', text))
        syllables = self._count_syllables(text)
        
        if sentences == 0 or words == 0:
            return 0
        
        # Flesch Reading Ease approximation
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0, min(100, score))
    
    def _count_syllables(self, text: str) -> int:
        """Simple syllable counting"""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        total_syllables = 0
        
        for word in words:
            syllables = len(re.findall(r'[aeiouAEIOU]', word))
            if word.endswith('e'):
                syllables -= 1
            if syllables == 0:
                syllables = 1
            total_syllables += syllables
        
        return total_syllables
    
    def calculate_seo_score(self, text: str, target_keywords: List[str] = None) -> float:
        """Calculate overall SEO score"""
        score = 0
        factors = 0
        
        # Word count factor (400-2000 words is optimal)
        word_count = len(re.findall(r'\b\w+\b', text))
        if 400 <= word_count <= 2000:
            score += 25
        elif 200 <= word_count < 400 or 2000 < word_count <= 3000:
            score += 15
        else:
            score += 5
        factors += 25
        
        # Readability factor
        readability = self.calculate_readability(text)
        if readability >= 60:
            score += 20
        elif readability >= 30:
            score += 10
        else:
            score += 5
        factors += 20
        
        # Keyword density factor
        if target_keywords:
            keyword_density = self.extract_keywords(text, target_keywords)
            avg_density = sum(keyword_density.values()) / len(keyword_density) if keyword_density else 0
            if 1 <= avg_density <= 3:
                score += 25
            elif 0.5 <= avg_density < 1 or 3 < avg_density <= 5:
                score += 15
            else:
                score += 5
            factors += 25
        
        # Structure factors
        sentences = len(re.findall(r'[.!?]+', text))
        paragraphs = len(text.split('\n\n'))
        
        if sentences > 0:
            avg_sentence_length = word_count / sentences
            if 15 <= avg_sentence_length <= 25:
                score += 15
            elif 10 <= avg_sentence_length < 15 or 25 < avg_sentence_length <= 30:
                score += 10
            else:
                score += 5
        factors += 15
        
        # Paragraph structure
        if paragraphs >= 3:
            score += 15
        elif paragraphs >= 2:
            score += 10
        else:
            score += 5
        factors += 15
        
        return (score / factors) * 100 if factors > 0 else 0
    
    def generate_suggestions(self, text: str, grammar_issues: List[Dict], 
                           spelling_issues: List[Dict], seo_score: float,
                           keyword_density: Dict[str, float]) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []
        
        # Grammar and spelling suggestions
        if grammar_issues:
            suggestions.append(f"Fix {len(grammar_issues)} grammar issues found")
        if spelling_issues:
            suggestions.append(f"Correct {len(spelling_issues)} potential spelling errors")
        
        # SEO suggestions
        word_count = len(re.findall(r'\b\w+\b', text))
        if word_count < 300:
            suggestions.append("Increase content length (aim for 400-2000 words)")
        elif word_count > 3000:
            suggestions.append("Consider breaking content into smaller sections")
        
        # Readability suggestions
        readability = self.calculate_readability(text)
        if readability < 30:
            suggestions.append("Improve readability by using shorter sentences and simpler words")
        elif readability < 60:
            suggestions.append("Consider simplifying complex sentences for better readability")
        
        # Keyword density suggestions
        if keyword_density:
            high_density = [k for k, v in keyword_density.items() if v > 5]
            low_density = [k for k, v in keyword_density.items() if v < 0.5]
            
            if high_density:
                suggestions.append(f"Reduce keyword density for: {', '.join(high_density[:3])}")
            if low_density:
                suggestions.append(f"Consider increasing usage of: {', '.join(low_density[:3])}")
        
        # Structure suggestions
        sentences = len(re.findall(r'[.!?]+', text))
        if sentences > 0:
            avg_sentence_length = word_count / sentences
            if avg_sentence_length > 30:
                suggestions.append("Break down long sentences for better readability")
        
        paragraphs = len(text.split('\n\n'))
        if paragraphs < 3:
            suggestions.append("Break content into more paragraphs for better structure")
        
        return suggestions
    
    def optimize_content(self, text: str, target_keywords: List[str] = None) -> OptimizationResult:
        """Main optimization function"""
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Check grammar and spelling
        grammar_issues, spelling_issues = self.check_grammar_and_spelling(cleaned_text)
        
        # Calculate metrics
        keyword_density = self.extract_keywords(cleaned_text, target_keywords)
        seo_score = self.calculate_seo_score(cleaned_text, target_keywords)
        readability_score = self.calculate_readability(cleaned_text)
        word_count = len(re.findall(r'\b\w+\b', cleaned_text))
        
        # Generate suggestions
        suggestions = self.generate_suggestions(
            cleaned_text, grammar_issues, spelling_issues, 
            seo_score, keyword_density
        )
        
        # Apply basic optimizations
        optimized_text = self._apply_basic_optimizations(cleaned_text, target_keywords)
        
        return OptimizationResult(
            original_content=text,
            optimized_content=optimized_text,
            grammar_issues=grammar_issues,
            spelling_issues=spelling_issues,
            seo_score=seo_score,
            keyword_density=keyword_density,
            readability_score=readability_score,
            word_count=word_count,
            suggestions=suggestions
        )
    
    def _apply_basic_optimizations(self, text: str, target_keywords: List[str] = None) -> str:
        """Apply basic text optimizations"""
        optimized = text
        
        # Fix common punctuation issues
        optimized = re.sub(r'\s+([,.!?;:])', r'\1', optimized)
        optimized = re.sub(r'([.!?])\s*([a-zA-Z])', r'\1 \2', optimized)
        
        # Ensure proper paragraph breaks
        optimized = re.sub(r'\n{3,}', '\n\n', optimized)
        
        # Add keywords naturally if specified and density is low
        if target_keywords:
            for keyword in target_keywords:
                current_density = (optimized.lower().count(keyword.lower()) / 
                                 len(re.findall(r'\b\w+\b', optimized))) * 100
                if current_density < 0.5:
                    # Find appropriate places to add keyword
                    sentences = sent_tokenize(optimized) if nltk_available else optimized.split('.')
                    if sentences and len(sentences) > 1:
                        # Add keyword to first paragraph if not present
                        first_paragraph = sentences[0]
                        if keyword.lower() not in first_paragraph.lower():
                            optimized = optimized.replace(
                                first_paragraph, 
                                f"{first_paragraph.rstrip('.')}. This content focuses on {keyword}."
                            )
        
        return optimized

def main():
    parser = argparse.ArgumentParser(description='Content SEO Optimizer')
    parser.add_argument('input_file', help='Input text file to optimize')
    parser.add_argument('-o', '--output', help='Output file for optimized content')
    parser.add_argument('-k', '--keywords', nargs='+', help='Target keywords for SEO optimization')
    parser.add_argument('-j', '--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--install-deps', action='store_true', help='Install required dependencies')
    
    args = parser.parse_args()
    
    if args.install_deps:
        install_dependencies()
        return
    
    # Read input file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Initialize optimizer
    optimizer = ContentOptimizer()
    
    # Optimize content
    result = optimizer.optimize_content(content, args.keywords)
    
    if args.json:
        # Output as JSON
        output_data = {
            'seo_score': result.seo_score,
            'readability_score': result.readability_score,
            'word_count': result.word_count,
            'grammar_issues_count': len(result.grammar_issues),
            'spelling_issues_count': len(result.spelling_issues),
            'keyword_density': result.keyword_density,
            'suggestions': result.suggestions,
            'optimized_content': result.optimized_content
        }
        
        output_file = args.output or 'optimization_result.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    else:
        # Display results
        print("=" * 60)
        print("CONTENT OPTIMIZATION RESULTS")
        print("=" * 60)
        print(f"Word Count: {result.word_count}")
        print(f"SEO Score: {result.seo_score:.1f}/100")
        print(f"Readability Score: {result.readability_score:.1f}/100")
        print(f"Grammar Issues: {len(result.grammar_issues)}")
        print(f"Spelling Issues: {len(result.spelling_issues)}")
        
        if result.keyword_density:
            print("\nKeyword Density:")
            for keyword, density in sorted(result.keyword_density.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {keyword}: {density:.2f}%")
        
        if result.suggestions:
            print("\nSuggestions for Improvement:")
            for i, suggestion in enumerate(result.suggestions, 1):
                print(f"  {i}. {suggestion}")
        
        if result.grammar_issues:
            print(f"\nGrammar Issues ({len(result.grammar_issues)}):")
            for issue in result.grammar_issues[:5]:
                print(f"  - {issue['message']}")
                if issue['suggestions']:
                    print(f"    Suggestions: {', '.join(issue['suggestions'])}")
        
        if result.spelling_issues:
            print(f"\nSpelling Issues ({len(result.spelling_issues)}):")
            for issue in result.spelling_issues[:5]:
                print(f"  - {issue['message']}")
                if issue.get('suggestions'):
                    print(f"    Suggestions: {', '.join(issue['suggestions'])}")
        
        # Save optimized content
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result.optimized_content)
            print(f"\nOptimized content saved to: {args.output}")

def install_dependencies():
    """Install required dependencies"""
    import subprocess
    import sys
    
    packages = [
        'nltk',
        'textstat', 
        'language-tool-python'
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
    
    print("\nDependencies installation completed!")
    print("Note: You may need to restart the application for changes to take effect.")

if __name__ == "__main__":
    main()