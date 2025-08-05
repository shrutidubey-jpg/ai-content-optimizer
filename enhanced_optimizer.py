#!/usr/bin/env python3
"""
Enhanced Content SEO Optimizer - Improved keyword optimization
Properly integrates target keywords into content optimization
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from collections import Counter
import random

class EnhancedContentOptimizer:
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'out', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'
        }
        
        # Keyword integration phrases
        self.keyword_phrases = [
            "This guide focuses on {keyword} and its implementation.",
            "Understanding {keyword} is crucial for success.",
            "The importance of {keyword} cannot be overstated.",
            "Let's explore how {keyword} can benefit your strategy.",
            "Implementing {keyword} effectively requires careful planning.",
            "Best practices for {keyword} include the following approaches.",
            "When considering {keyword}, it's important to remember these key points.",
            "The role of {keyword} in modern business is significant."
        ]
    
    def analyze_content(self, content: str, target_keywords: List[str] = None) -> Dict:
        """Analyze content and provide optimization metrics"""
        content = self.clean_text(content)
        
        # Basic metrics
        word_count = len(re.findall(r'\b\w+\b', content))
        sentences = len(re.findall(r'[.!?]+', content))
        paragraphs = len([p for p in content.split('\n\n') if p.strip()])
        
        # Calculate keyword density
        keyword_density = self.calculate_keyword_density(content, target_keywords)
        
        # Grammar and spelling issues
        grammar_issues = self.detect_grammar_issues(content)
        spelling_issues = self.detect_spelling_issues(content)
        
        # SEO score calculation
        seo_score = self.calculate_seo_score(content, target_keywords, keyword_density)
        
        # Readability score
        readability_score = self.calculate_readability(content)
        
        # Generate optimization suggestions
        suggestions = self.generate_suggestions(
            content, target_keywords, keyword_density, word_count, 
            len(grammar_issues), len(spelling_issues)
        )
        
        return {
            'seo_score': seo_score,
            'readability_score': readability_score,
            'word_count': word_count,
            'sentence_count': sentences,
            'paragraph_count': paragraphs,
            'keyword_density': keyword_density,
            'grammar_issues': grammar_issues,
            'spelling_issues': spelling_issues,
            'suggestions': suggestions
        }
    
    def optimize_content(self, content: str, target_keywords: List[str] = None) -> Dict:
        """Optimize content for target keywords and SEO"""
        original_content = content
        analysis = self.analyze_content(content, target_keywords)
        
        # Apply optimizations
        optimized_content = self.apply_optimizations(content, target_keywords, analysis)
        
        # Re-analyze optimized content
        optimized_analysis = self.analyze_content(optimized_content, target_keywords)
        
        return {
            'original_content': original_content,
            'optimized_content': optimized_content,
            'original_analysis': analysis,
            'optimized_analysis': optimized_analysis,
            'improvements': self.calculate_improvements(analysis, optimized_analysis)
        }
    
    def apply_optimizations(self, content: str, target_keywords: List[str], analysis: Dict) -> str:
        """Apply comprehensive content optimizations"""
        optimized = content
        
        # 1. Fix grammar and spelling issues
        optimized = self.fix_grammar_issues(optimized)
        optimized = self.fix_spelling_issues(optimized)
        
        # 2. Optimize keyword usage
        if target_keywords:
            optimized = self.optimize_keyword_usage(optimized, target_keywords, analysis['keyword_density'])
        
        # 3. Improve content structure
        optimized = self.improve_content_structure(optimized, target_keywords)
        
        # 4. Enhance readability
        optimized = self.improve_readability(optimized)
        
        return optimized
    
    def optimize_keyword_usage(self, content: str, target_keywords: List[str], current_density: Dict[str, float]) -> str:
        """Optimize keyword usage in content"""
        optimized = content
        sentences = re.split(r'(?<=[.!?])\s+', optimized)
        
        for keyword in target_keywords:
            keyword_lower = keyword.lower()
            current_count = optimized.lower().count(keyword_lower)
            word_count = len(re.findall(r'\b\w+\b', optimized))
            current_density_pct = (current_count / word_count * 100) if word_count > 0 else 0
            
            # Optimal density is 1-3%
            target_density = 2.0  # Target 2% density
            target_count = max(1, int((target_density / 100) * word_count))
            
            if current_count < target_count:
                # Add keyword strategically
                additions_needed = target_count - current_count
                optimized = self.add_keyword_strategically(optimized, keyword, additions_needed)
            
            elif current_density_pct > 5.0:  # Too high density
                # Reduce keyword usage
                optimized = self.reduce_keyword_density(optimized, keyword)
        
        return optimized
    
    def add_keyword_strategically(self, content: str, keyword: str, count: int) -> str:
        """Add keywords strategically to content"""
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        # Add to introduction if not present
        if not keyword.lower() in content[:200].lower() and len(sentences) > 0:
            intro_addition = f" This content focuses on {keyword} and related topics."
            sentences[0] = sentences[0].rstrip('.') + '.' + intro_addition
            count -= 1
        
        # Add keywords to strategic positions
        positions_added = 0
        for i, sentence in enumerate(sentences):
            if positions_added >= count:
                break
                
            # Skip if keyword already in sentence
            if keyword.lower() in sentence.lower():
                continue
            
            # Add keyword naturally to sentence
            if len(sentence) > 50 and random.random() < 0.3:  # 30% chance per sentence
                # Insert keyword phrase
                phrase = random.choice(self.keyword_phrases).format(keyword=keyword)
                
                # Add as new sentence after current one
                sentences.insert(i + 1, phrase)
                positions_added += 1
        
        # If we still need more keywords, add a conclusion paragraph
        if positions_added < count:
            conclusion = f"\n\nIn conclusion, {keyword} plays a vital role in achieving your goals. Understanding and implementing {keyword} strategies will help you succeed in today's competitive landscape."
            content = '\n\n'.join(sentences) + conclusion
        else:
            content = ' '.join(sentences)
        
        return content
    
    def reduce_keyword_density(self, content: str, keyword: str) -> str:
        """Reduce keyword density by replacing some instances with synonyms"""
        # Simple synonym replacement (in a real implementation, you'd use a thesaurus API)
        synonyms = {
            'marketing': ['promotion', 'advertising', 'branding'],
            'business': ['company', 'organization', 'enterprise'],
            'SEO': ['search optimization', 'organic search'],
            'content': ['material', 'information', 'copy'],
            'digital': ['online', 'electronic', 'web-based'],
            'strategy': ['approach', 'plan', 'method'],
            'optimization': ['improvement', 'enhancement', 'refinement']
        }
        
        keyword_lower = keyword.lower()
        if keyword_lower in synonyms:
            # Replace every 3rd occurrence with a synonym
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            matches = list(pattern.finditer(content))
            
            for i, match in enumerate(reversed(matches)):
                if i % 3 == 0:  # Replace every 3rd occurrence
                    synonym = random.choice(synonyms[keyword_lower])
                    start, end = match.span()
                    content = content[:start] + synonym + content[end:]
        
        return content
    
    def improve_content_structure(self, content: str, target_keywords: List[str]) -> str:
        """Improve content structure and organization"""
        optimized = content
        
        # Ensure proper paragraph breaks
        optimized = re.sub(r'\n{3,}', '\n\n', optimized)
        
        # Add subheadings if content is long enough
        paragraphs = optimized.split('\n\n')
        if len(paragraphs) > 3 and target_keywords:
            # Add a subheading with keyword
            main_keyword = target_keywords[0]
            subheading = f"\n\n## Key Benefits of {main_keyword}\n\n"
            
            # Insert subheading in the middle
            mid_point = len(paragraphs) // 2
            paragraphs.insert(mid_point, subheading.strip())
            optimized = '\n\n'.join(paragraphs)
        
        return optimized
    
    def fix_grammar_issues(self, content: str) -> str:
        """Fix common grammar issues"""
        # Fix spacing issues
        content = re.sub(r'\s+([,.!?;:])', r'\1', content)
        content = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', content)
        
        # Fix common grammar patterns
        fixes = {
            r'\b(its)\s+(going|coming|leaving)\b': r"it's \2",
            r'\b(your)\s+(going|coming|leaving)\b': r"you're \2",
            r'\b(there)\s+(going|coming)\b': r"they're \2",
            r'\b(to)\s+(much|many)\b': r'too \2',
            r'([a-z])\.([A-Z])': r'\1. \2',
            r'([a-z]),([A-Z])': r'\1, \2'
        }
        
        for pattern, replacement in fixes.items():
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def fix_spelling_issues(self, content: str) -> str:
        """Fix common spelling issues"""
        corrections = {
            r'\btodays\b': "today's",
            r'\bcompanys\b': "company's",
            r'\bbusinesss\b': 'business',
            r'\brecieve\b': 'receive',
            r'\bseperate\b': 'separate',
            r'\bdefinately\b': 'definitely',
            r'\boccured\b': 'occurred',
            r'\bbegining\b': 'beginning',
            r'\bmanagment\b': 'management',
            r'\bsucessful\b': 'successful',
            r'\bneccessary\b': 'necessary',
            r'\bexcercise\b': 'exercise',
            # Fix repeated words
            r'\b(\w+)\s+\1\b': r'\1'
        }
        
        for pattern, replacement in corrections.items():
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return content
    
    def improve_readability(self, content: str) -> str:
        """Improve content readability"""
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        improved_sentences = []
        for sentence in sentences:
            # Break overly long sentences
            if len(sentence.split()) > 25:
                # Try to split at conjunctions
                parts = re.split(r'\s+(and|but|or|however|moreover|furthermore)\s+', sentence, maxsplit=1)
                if len(parts) > 1:
                    # Reconstruct with proper punctuation
                    first_part = parts[0].rstrip('.,!?') + '.'
                    second_part = parts[1].capitalize() + ' ' + ' '.join(parts[2:]) if len(parts) > 2 else parts[1]
                    improved_sentences.extend([first_part, second_part])
                else:
                    improved_sentences.append(sentence)
            else:
                improved_sentences.append(sentence)
        
        return ' '.join(improved_sentences)
    
    def calculate_keyword_density(self, content: str, target_keywords: List[str] = None) -> Dict[str, float]:
        """Calculate keyword density"""
        words = re.findall(r'\b[a-zA-Z]+\b', content.lower())
        total_words = len(words)
        
        if total_words == 0:
            return {}
        
        density = {}
        
        # Calculate density for target keywords
        if target_keywords:
            for keyword in target_keywords:
                count = content.lower().count(keyword.lower())
                density[keyword] = (count / total_words) * 100
        
        # Also calculate for top occurring words
        word_counts = Counter([word for word in words if word not in self.stop_words and len(word) > 2])
        for word, count in word_counts.most_common(5):
            if word not in density:
                density[word] = (count / total_words) * 100
        
        return density
    
    def calculate_seo_score(self, content: str, target_keywords: List[str], keyword_density: Dict[str, float]) -> float:
        """Calculate comprehensive SEO score"""
        score = 0
        max_score = 100
        
        # Word count factor (25 points)
        word_count = len(re.findall(r'\b\w+\b', content))
        if 400 <= word_count <= 2000:
            score += 25
        elif 200 <= word_count < 400 or 2000 < word_count <= 3000:
            score += 18
        elif word_count >= 100:
            score += 12
        else:
            score += 5
        
        # Keyword optimization factor (30 points)
        if target_keywords and keyword_density:
            keyword_scores = []
            for keyword in target_keywords:
                density = keyword_density.get(keyword, 0)
                if 1.0 <= density <= 3.0:  # Optimal range
                    keyword_scores.append(10)
                elif 0.5 <= density < 1.0 or 3.0 < density <= 5.0:  # Acceptable range
                    keyword_scores.append(7)
                elif density > 0:  # At least present
                    keyword_scores.append(4)
                else:  # Not present
                    keyword_scores.append(0)
            
            if keyword_scores:
                avg_keyword_score = sum(keyword_scores) / len(keyword_scores)
                score += avg_keyword_score * 3  # Scale to 30 points max
        else:
            score += 15  # Default if no keywords specified
        
        # Content structure factor (25 points)
        sentences = len(re.findall(r'[.!?]+', content))
        paragraphs = len([p for p in content.split('\n\n') if p.strip()])
        
        if sentences > 0:
            avg_sentence_length = word_count / sentences
            if 15 <= avg_sentence_length <= 25:
                score += 15
            elif 10 <= avg_sentence_length < 15 or 25 < avg_sentence_length <= 30:
                score += 10
            else:
                score += 5
        
        # Paragraph structure
        if paragraphs >= 3:
            score += 10
        elif paragraphs >= 2:
            score += 7
        else:
            score += 3
        
        # Readability factor (20 points)
        readability = self.calculate_readability(content)
        if readability >= 60:
            score += 20
        elif readability >= 40:
            score += 15
        elif readability >= 20:
            score += 10
        else:
            score += 5
        
        return min(score, max_score)
    
    def calculate_readability(self, content: str) -> float:
        """Calculate readability score using Flesch Reading Ease approximation"""
        sentences = len(re.findall(r'[.!?]+', content))
        words = len(re.findall(r'\b\w+\b', content))
        syllables = self.count_syllables(content)
        
        if sentences == 0 or words == 0:
            return 0
        
        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0, min(100, score))
    
    def count_syllables(self, content: str) -> int:
        """Count syllables in content"""
        words = re.findall(r'\b[a-zA-Z]+\b', content.lower())
        total_syllables = 0
        
        for word in words:
            syllables = len(re.findall(r'[aeiouyAEIOUY]+', word))
            if word.endswith('e') and syllables > 1:
                syllables -= 1
            if syllables == 0:
                syllables = 1
            total_syllables += syllables
        
        return total_syllables
    
    def detect_grammar_issues(self, content: str) -> List[Dict]:
        """Detect grammar issues"""
        issues = []
        
        patterns = [
            (r'([a-z])\.([A-Z])', "Missing space after period"),
            (r'([a-z]),([A-Z])', "Missing space after comma"),
            (r'\b(its)\s+(going|coming|leaving)\b', "Possible error: 'its' vs 'it's'"),
            (r'\b(your)\s+(going|coming|leaving)\b', "Possible error: 'your' vs 'you're'"),
            (r'\b(there)\s+(going|coming|is|are)\b', "Possible error: 'there' vs 'they're'"),
            (r'\b(to)\s+(much|many)\b', "Possible error: 'to' vs 'too'"),
            (r'\b(\w+)\s+\1\b', "Repeated word detected")
        ]
        
        for pattern, message in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                issues.append({
                    'message': message,
                    'context': match.group(),
                    'offset': match.start(),
                    'length': len(match.group())
                })
        
        return issues
    
    def detect_spelling_issues(self, content: str) -> List[Dict]:
        """Detect spelling issues"""
        issues = []
        
        common_misspellings = {
            'todays': "today's",
            'companys': "company's",
            'businesss': 'business',
            'recieve': 'receive',
            'seperate': 'separate',
            'definately': 'definitely',
            'occured': 'occurred',
            'begining': 'beginning',
            'managment': 'management',
            'sucessful': 'successful',
            'neccessary': 'necessary',
            'excercise': 'exercise'
        }
        
        words = re.findall(r'\b[a-zA-Z]+\b', content)
        for word in words:
            word_lower = word.lower()
            if word_lower in common_misspellings:
                issues.append({
                    'message': f"Possible spelling error: '{word}'",
                    'context': word,
                    'suggestions': [common_misspellings[word_lower]]
                })
        
        return issues
    
    def generate_suggestions(self, content: str, target_keywords: List[str], 
                           keyword_density: Dict[str, float], word_count: int,
                           grammar_issues: int, spelling_issues: int) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []
        
        # Grammar and spelling
        if grammar_issues > 0:
            suggestions.append(f"Fix {grammar_issues} grammar issues detected")
        if spelling_issues > 0:
            suggestions.append(f"Correct {spelling_issues} spelling errors found")
        
        # Word count optimization
        if word_count < 300:
            suggestions.append("Increase content length (aim for 400-2000 words for better SEO)")
        elif word_count > 3000:
            suggestions.append("Consider breaking content into multiple sections")
        
        # Keyword optimization
        if target_keywords and keyword_density:
            for keyword in target_keywords:
                density = keyword_density.get(keyword, 0)
                if density == 0:
                    suggestions.append(f"Add '{keyword}' to your content (currently missing)")
                elif density < 0.5:
                    suggestions.append(f"Increase usage of '{keyword}' (currently {density:.2f}%)")
                elif density > 5.0:
                    suggestions.append(f"Reduce keyword stuffing for '{keyword}' (currently {density:.2f}%)")
        
        # Structure suggestions
        paragraphs = len([p for p in content.split('\n\n') if p.strip()])
        if paragraphs < 3:
            suggestions.append("Break content into more paragraphs for better readability")
        
        # Readability
        readability = self.calculate_readability(content)
        if readability < 30:
            suggestions.append("Improve readability by using shorter sentences and simpler words")
        elif readability < 60:
            suggestions.append("Consider simplifying some complex sentences")
        
        return suggestions[:8]  # Limit to 8 suggestions
    
    def calculate_improvements(self, original: Dict, optimized: Dict) -> Dict:
        """Calculate improvements between original and optimized content"""
        return {
            'seo_score_improvement': optimized['seo_score'] - original['seo_score'],
            'readability_improvement': optimized['readability_score'] - original['readability_score'],
            'grammar_issues_fixed': len(original['grammar_issues']) - len(optimized['grammar_issues']),
            'spelling_issues_fixed': len(original['spelling_issues']) - len(optimized['spelling_issues']),
            'keyword_density_optimized': self.compare_keyword_densities(
                original['keyword_density'], 
                optimized['keyword_density']
            )
        }
    
    def compare_keyword_densities(self, original: Dict, optimized: Dict) -> Dict:
        """Compare keyword densities"""
        improvements = {}
        for keyword in set(list(original.keys()) + list(optimized.keys())):
            orig_density = original.get(keyword, 0)
            opt_density = optimized.get(keyword, 0)
            improvements[keyword] = {
                'original': orig_density,
                'optimized': opt_density,
                'change': opt_density - orig_density
            }
        return improvements
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

# Test the enhanced optimizer
def test_optimizer():
    """Test the enhanced optimizer with sample content"""
    optimizer = EnhancedContentOptimizer()
    
    sample_content = """
    Digital Marketing Strategies for Modern Businesses
    
    In todays digital landscape,businesses need to adopt effective digital marketing strategies to stay competitive.Digital marketing has become a crucial component for business success.This article will explore various digital marketing techniques that can help your business business grow.
    
    Search engine optimization is one of the most important aspects of digital marketing.SEO helps businesses improve their online visibility and attract more customers.By optimizing your content for search engines,you can increase organic traffic to your website website.
    """
    
    target_keywords = ["digital marketing", "SEO", "business growth"]
    
    print("🔍 Testing Enhanced Content Optimizer...")
    print("=" * 50)
    
    result = optimizer.optimize_content(sample_content, target_keywords)
    
    print("📊 ORIGINAL ANALYSIS:")
    orig = result['original_analysis']
    print(f"  SEO Score: {orig['seo_score']:.1f}/100")
    print(f"  Word Count: {orig['word_count']}")
    print(f"  Grammar Issues: {len(orig['grammar_issues'])}")
    print(f"  Spelling Issues: {len(orig['spelling_issues'])}")
    
    print("\n✨ OPTIMIZED ANALYSIS:")
    opt = result['optimized_analysis']
    print(f"  SEO Score: {opt['seo_score']:.1f}/100")
    print(f"  Word Count: {opt['word_count']}")
    print(f"  Grammar Issues: {len(opt['grammar_issues'])}")
    print(f"  Spelling Issues: {len(opt['spelling_issues'])}")
    
    print("\n📈 IMPROVEMENTS:")
    imp = result['improvements']
    print(f"  SEO Score: +{imp['seo_score_improvement']:.1f}")
    print(f"  Grammar Fixed: {imp['grammar_issues_fixed']}")
    print(f"  Spelling Fixed: {imp['spelling_issues_fixed']}")
    
    print("\n🎯 KEYWORD DENSITY:")
    for keyword, data in imp['keyword_density_optimized'].items():
        print(f"  {keyword}: {data['original']:.2f}% → {data['optimized']:.2f}% ({data['change']:+.2f}%)")
    
    print("\n✨ OPTIMIZED CONTENT:")
    print("-" * 30)
    print(result['optimized_content'][:500] + "...")
    
    return result

if __name__ == "__main__":
    test_optimizer()