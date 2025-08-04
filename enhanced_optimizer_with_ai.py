#!/usr/bin/env python3
"""
Enhanced Content SEO Optimizer with AI Grammar Agent
Advanced content optimization with intelligent grammar correction
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from collections import Counter
import random
from ai_grammar_agent import AIGrammarAgent, GrammarIssue

class EnhancedOptimizerWithAI:
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
        
        # Initialize AI Grammar Agent
        self.grammar_ai = AIGrammarAgent()
        
        # Keyword integration phrases
        self.keyword_phrases = [
            "This comprehensive guide explores {keyword} and its practical applications.",
            "Understanding {keyword} is essential for achieving optimal results.",
            "The strategic implementation of {keyword} can significantly impact success.",
            "Let's examine how {keyword} transforms modern business practices.",
            "Effective {keyword} strategies require careful planning and execution.",
            "Best practices for {keyword} include these proven methodologies.",
            "When implementing {keyword}, consider these critical success factors.",
            "The role of {keyword} in contemporary strategy cannot be understated."
        ]
    
    def optimize_content(self, content: str, target_keywords: List[str] = None, writing_context: str = 'general') -> Dict:
        """Enhanced content optimization with AI grammar correction"""
        original_content = content
        
        # Step 1: AI Grammar Analysis and Correction
        print("🤖 Running AI Grammar Analysis...")
        grammar_report = self.grammar_ai.generate_grammar_report(content, writing_context)
        
        # Step 2: Apply grammar corrections
        content_after_grammar = grammar_report['corrected_text']
        
        # Step 3: Perform SEO and keyword optimization
        print("🎯 Performing SEO and keyword optimization...")
        seo_optimized = self.apply_seo_optimizations(content_after_grammar, target_keywords)
        
        # Step 4: Final analysis
        original_analysis = self.analyze_content(original_content, target_keywords)
        optimized_analysis = self.analyze_content(seo_optimized, target_keywords)
        
        # Step 5: Calculate comprehensive improvements
        improvements = self.calculate_comprehensive_improvements(
            original_analysis, optimized_analysis, grammar_report
        )
        
        return {
            'original_content': original_content,
            'optimized_content': seo_optimized,
            'original_analysis': original_analysis,
            'optimized_analysis': optimized_analysis,
            'grammar_report': grammar_report,
            'improvements': improvements,
            'ai_enhancements': {
                'grammar_score_improvement': grammar_report['improvement_summary']['score_improvement'],
                'issues_fixed': grammar_report['improvement_summary']['issues_fixed'],
                'text_quality_boost': grammar_report['improvement_summary']['text_quality_boost']
            }
        }
    
    def apply_seo_optimizations(self, content: str, target_keywords: List[str]) -> str:
        """Apply SEO optimizations while preserving grammar corrections"""
        optimized = content
        
        # 1. Optimize keyword usage
        if target_keywords:
            optimized = self.optimize_keyword_usage(optimized, target_keywords)
        
        # 2. Improve content structure
        optimized = self.improve_content_structure(optimized, target_keywords)
        
        # 3. Enhance readability (carefully to preserve grammar)
        optimized = self.improve_readability_preserving_grammar(optimized)
        
        return optimized
    
    def optimize_keyword_usage(self, content: str, target_keywords: List[str]) -> str:
        """Enhanced keyword optimization with AI awareness"""
        optimized = content
        sentences = re.split(r'(?<=[.!?])\s+', optimized)
        
        for keyword in target_keywords:
            keyword_lower = keyword.lower()
            current_count = optimized.lower().count(keyword_lower)
            word_count = len(re.findall(r'\b\w+\b', optimized))
            current_density_pct = (current_count / word_count * 100) if word_count > 0 else 0
            
            # Target density: 1.5-2.5% for primary keywords
            target_density = 2.0
            target_count = max(1, int((target_density / 100) * word_count))
            
            if current_count < target_count:
                # Add keywords strategically without breaking grammar
                additions_needed = target_count - current_count
                optimized = self.add_keyword_intelligently(optimized, keyword, additions_needed)
            
            elif current_density_pct > 4.0:  # Reduce keyword stuffing
                optimized = self.reduce_keyword_density_smartly(optimized, keyword)
        
        return optimized
    
    def add_keyword_intelligently(self, content: str, keyword: str, count: int) -> str:
        """Add keywords intelligently without breaking grammar"""
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        # Add to introduction if not present
        if not keyword.lower() in content[:200].lower() and len(sentences) > 0:
            intro_phrase = random.choice(self.keyword_phrases).format(keyword=keyword)
            sentences.insert(1, intro_phrase)
            count -= 1
        
        # Add keywords to strategic positions
        positions_added = 0
        for i, sentence in enumerate(sentences):
            if positions_added >= count:
                break
                
            # Skip if keyword already in sentence
            if keyword.lower() in sentence.lower():
                continue
            
            # Add keyword naturally to longer sentences
            if len(sentence.split()) > 8 and random.random() < 0.4:
                # Create grammatically correct insertion
                enhanced_sentence = self.enhance_sentence_with_keyword(sentence, keyword)
                if enhanced_sentence != sentence:
                    sentences[i] = enhanced_sentence
                    positions_added += 1
        
        # If we still need more keywords, add a conclusion paragraph
        if positions_added < count:
            conclusion = f"\n\nIn summary, {keyword} represents a fundamental aspect of modern strategy. Organizations that effectively implement {keyword} approaches typically achieve superior outcomes and sustainable competitive advantages."
            content = '\n\n'.join(sentences) + conclusion
        else:
            content = ' '.join(sentences)
        
        return content
    
    def enhance_sentence_with_keyword(self, sentence: str, keyword: str) -> str:
        """Enhance a sentence with keyword while maintaining grammar"""
        # Check if we can naturally integrate the keyword
        if 'strategy' in sentence.lower() or 'approach' in sentence.lower():
            enhanced = sentence.replace('strategy', f'{keyword} strategy', 1)
            enhanced = enhanced.replace('approach', f'{keyword} approach', 1)
            return enhanced
        
        # Add as appositive clause
        if ',' in sentence:
            parts = sentence.split(',', 1)
            enhanced = f"{parts[0]}, particularly in {keyword},{parts[1]}"
            return enhanced
        
        return sentence
    
    def reduce_keyword_density_smartly(self, content: str, keyword: str) -> str:
        """Reduce keyword density using intelligent synonyms"""
        # Advanced synonym mapping
        synonym_map = {
            'digital marketing': ['online promotion', 'web marketing', 'internet advertising', 'digital promotion'],
            'SEO': ['search optimization', 'organic search', 'search engine marketing', 'web visibility'],
            'content marketing': ['content strategy', 'editorial marketing', 'information marketing', 'content creation'],
            'social media': ['social platforms', 'social networks', 'online communities', 'digital channels'],
            'business': ['organization', 'company', 'enterprise', 'corporation'],
            'strategy': ['approach', 'methodology', 'plan', 'framework'],
            'optimization': ['improvement', 'enhancement', 'refinement', 'fine-tuning'],
            'marketing': ['promotion', 'advertising', 'branding', 'outreach']
        }
        
        keyword_lower = keyword.lower()
        if keyword_lower in synonym_map:
            synonyms = synonym_map[keyword_lower]
            
            # Replace every 3rd occurrence with a synonym
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            matches = list(pattern.finditer(content))
            
            for i, match in enumerate(reversed(matches)):
                if i % 3 == 0:  # Replace every 3rd occurrence
                    synonym = random.choice(synonyms)
                    start, end = match.span()
                    content = content[:start] + synonym + content[end:]
        
        return content
    
    def improve_content_structure(self, content: str, target_keywords: List[str]) -> str:
        """Improve content structure while preserving AI grammar corrections"""
        optimized = content
        
        # Ensure proper paragraph breaks
        optimized = re.sub(r'\n{3,}', '\n\n', optimized)
        
        # Add strategic subheadings for longer content
        paragraphs = optimized.split('\n\n')
        if len(paragraphs) > 4 and target_keywords:
            main_keyword = target_keywords[0]
            
            # Insert subheading at strategic position
            mid_point = len(paragraphs) // 2
            subheading = f"\n\n## Advanced {main_keyword} Strategies\n\n"
            paragraphs.insert(mid_point, subheading.strip())
            optimized = '\n\n'.join(paragraphs)
        
        return optimized
    
    def improve_readability_preserving_grammar(self, content: str) -> str:
        """Improve readability while preserving AI grammar corrections"""
        sentences = re.split(r'(?<=[.!?])\s+', content)
        improved_sentences = []
        
        for sentence in sentences:
            # Only modify overly complex sentences
            words = sentence.split()
            if len(words) > 30:
                # Try to split at natural break points
                conjunctions = ['and', 'but', 'however', 'moreover', 'furthermore', 'additionally']
                for conj in conjunctions:
                    if f' {conj} ' in sentence.lower():
                        parts = sentence.split(f' {conj} ', 1)
                        if len(parts) == 2:
                            # Create two sentences
                            first_part = parts[0].rstrip('.,!?') + '.'
                            second_part = conj.capitalize() + ', ' + parts[1]
                            improved_sentences.extend([first_part, second_part])
                            break
                else:
                    improved_sentences.append(sentence)
            else:
                improved_sentences.append(sentence)
        
        return ' '.join(improved_sentences)
    
    def analyze_content(self, content: str, target_keywords: List[str] = None) -> Dict:
        """Comprehensive content analysis including AI grammar assessment"""
        content = self.clean_text(content)
        
        # Basic metrics
        word_count = len(re.findall(r'\b\w+\b', content))
        sentences = len(re.findall(r'[.!?]+', content))
        paragraphs = len([p for p in content.split('\n\n') if p.strip()])
        
        # Get AI grammar score
        grammar_score_data = self.grammar_ai.get_grammar_score(content)
        
        # Calculate keyword density
        keyword_density = self.calculate_keyword_density(content, target_keywords)
        
        # SEO score calculation
        seo_score = self.calculate_seo_score(content, target_keywords, keyword_density)
        
        # Enhanced readability score
        readability_score = self.calculate_readability(content)
        
        # Advanced grammar analysis
        grammar_issues = self.grammar_ai.analyze_grammar(content)
        
        # Generate suggestions
        suggestions = self.generate_enhanced_suggestions(
            content, target_keywords, keyword_density, word_count, 
            grammar_score_data, readability_score
        )
        
        return {
            'seo_score': seo_score,
            'readability_score': readability_score,
            'grammar_score': grammar_score_data['grammar_score'],
            'word_count': word_count,
            'sentence_count': sentences,
            'paragraph_count': paragraphs,
            'keyword_density': keyword_density,
            'grammar_issues': grammar_issues,
            'spelling_issues': [],  # Handled by AI agent
            'suggestions': suggestions,
            'grammar_details': grammar_score_data
        }
    
    def generate_enhanced_suggestions(self, content: str, target_keywords: List[str], 
                                    keyword_density: Dict[str, float], word_count: int,
                                    grammar_data: Dict, readability_score: float) -> List[str]:
        """Generate enhanced suggestions including AI insights"""
        suggestions = []
        
        # AI Grammar suggestions
        if grammar_data['critical_issues'] > 0:
            suggestions.append(f"🤖 AI detected {grammar_data['critical_issues']} critical grammar issues - auto-correction applied")
        
        if grammar_data['major_issues'] > 0:
            suggestions.append(f"🔧 Fixed {grammar_data['major_issues']} major grammar problems with AI assistance")
        
        # Content length optimization
        if word_count < 400:
            suggestions.append("📏 Expand content to 400+ words for better SEO performance")
        elif word_count > 2500:
            suggestions.append("✂️ Consider breaking into multiple focused articles")
        
        # Keyword optimization with AI awareness
        if target_keywords and keyword_density:
            for keyword in target_keywords:
                density = keyword_density.get(keyword, 0)
                if density == 0:
                    suggestions.append(f"🎯 Add '{keyword}' strategically (currently missing)")
                elif density < 1.0:
                    suggestions.append(f"📈 Increase '{keyword}' usage to 1.5-2.5% density (currently {density:.2f}%)")
                elif density > 4.0:
                    suggestions.append(f"⚠️ Reduce '{keyword}' density to avoid over-optimization (currently {density:.2f}%)")
        
        # Readability enhancements
        if readability_score < 40:
            suggestions.append("📖 Improve readability with shorter sentences and simpler language")
        elif readability_score < 60:
            suggestions.append("✨ Consider simplifying complex phrases for broader accessibility")
        
        # Structure improvements
        paragraphs = len([p for p in content.split('\n\n') if p.strip()])
        if paragraphs < 3:
            suggestions.append("📑 Break content into more paragraphs for better structure")
        
        # AI confidence feedback
        avg_confidence = grammar_data.get('confidence_avg', 1.0)
        if avg_confidence > 0.9:
            suggestions.append("🚀 AI grammar analysis shows high confidence in corrections")
        
        return suggestions[:8]  # Limit to top suggestions
    
    def calculate_comprehensive_improvements(self, original: Dict, optimized: Dict, grammar_report: Dict) -> Dict:
        """Calculate comprehensive improvements including AI enhancements"""
        return {
            'seo_score_improvement': optimized['seo_score'] - original['seo_score'],
            'readability_improvement': optimized['readability_score'] - original['readability_score'],
            'grammar_score_improvement': optimized['grammar_score'] - original['grammar_score'],
            'ai_grammar_fixes': grammar_report['improvement_summary']['issues_fixed'],
            'content_quality_boost': grammar_report['improvement_summary']['text_quality_boost'],
            'keyword_density_optimized': self.compare_keyword_densities(
                original['keyword_density'], 
                optimized['keyword_density']
            ),
            'overall_improvement_score': self.calculate_overall_improvement(original, optimized, grammar_report)
        }
    
    def calculate_overall_improvement(self, original: Dict, optimized: Dict, grammar_report: Dict) -> float:
        """Calculate overall improvement score"""
        seo_improvement = (optimized['seo_score'] - original['seo_score']) / 100
        readability_improvement = (optimized['readability_score'] - original['readability_score']) / 100
        grammar_improvement = grammar_report['improvement_summary']['text_quality_boost'] / 100
        
        # Weighted overall score
        overall_score = (seo_improvement * 0.4) + (readability_improvement * 0.3) + (grammar_improvement * 0.3)
        return overall_score * 100
    
    # Include all the helper methods from the previous optimizer
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
        
        # Add top occurring words
        word_counts = Counter([word for word in words if word not in self.stop_words and len(word) > 2])
        for word, count in word_counts.most_common(5):
            if word not in density:
                density[word] = (count / total_words) * 100
        
        return density
    
    def calculate_seo_score(self, content: str, target_keywords: List[str], keyword_density: Dict[str, float]) -> float:
        """Calculate comprehensive SEO score"""
        score = 0
        
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
                if 1.0 <= density <= 3.0:
                    keyword_scores.append(10)
                elif 0.5 <= density < 1.0 or 3.0 < density <= 5.0:
                    keyword_scores.append(7)
                elif density > 0:
                    keyword_scores.append(4)
                else:
                    keyword_scores.append(0)
            
            if keyword_scores:
                avg_keyword_score = sum(keyword_scores) / len(keyword_scores)
                score += avg_keyword_score * 3
        else:
            score += 15
        
        # Content structure (25 points)
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
        
        return min(score, 100)
    
    def calculate_readability(self, content: str) -> float:
        """Calculate readability score"""
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

def test_enhanced_optimizer():
    """Test the enhanced optimizer with AI grammar agent"""
    optimizer = EnhancedOptimizerWithAI()
    
    sample_content = """
    Digital Marketing Strategies for Modern Business
    
    Me and my team is working on digital marketing projects.We was planning to improve our companies online presence.Its going to be really good for business.
    
    Between you and I,digital marketing digital marketing is very important.SEO helps website get more customers customers.You should optimize content content for search engines.
    
    Digital marketing could of been implemented earlier.There going to be lots of opportunities their.
    """
    
    target_keywords = ["digital marketing", "SEO optimization", "business growth"]
    
    print("🚀 TESTING ENHANCED OPTIMIZER WITH AI GRAMMAR AGENT")
    print("=" * 60)
    
    result = optimizer.optimize_content(sample_content, target_keywords, 'business_writing')
    
    print("📊 ORIGINAL ANALYSIS:")
    orig = result['original_analysis']
    print(f"  SEO Score: {orig['seo_score']:.1f}/100")
    print(f"  Grammar Score: {orig['grammar_score']:.1f}/100")
    print(f"  Readability: {orig['readability_score']:.1f}/100")
    print(f"  Word Count: {orig['word_count']}")
    
    print("\n✨ OPTIMIZED ANALYSIS:")
    opt = result['optimized_analysis']
    print(f"  SEO Score: {opt['seo_score']:.1f}/100")
    print(f"  Grammar Score: {opt['grammar_score']:.1f}/100")
    print(f"  Readability: {opt['readability_score']:.1f}/100")
    print(f"  Word Count: {opt['word_count']}")
    
    print("\n🤖 AI GRAMMAR ENHANCEMENTS:")
    ai_enhancements = result['ai_enhancements']
    print(f"  Issues Fixed: {ai_enhancements['issues_fixed']}")
    print(f"  Quality Boost: {ai_enhancements['text_quality_boost']:.1f}%")
    
    print("\n📈 COMPREHENSIVE IMPROVEMENTS:")
    imp = result['improvements']
    print(f"  SEO Score: +{imp['seo_score_improvement']:.1f}")
    print(f"  Grammar Score: +{imp['grammar_score_improvement']:.1f}")
    print(f"  Readability: +{imp['readability_improvement']:.1f}")
    print(f"  Overall Improvement: +{imp['overall_improvement_score']:.1f}%")
    
    print("\n🎯 KEYWORD OPTIMIZATION:")
    for keyword, data in imp['keyword_density_optimized'].items():
        print(f"  {keyword}: {data['original']:.2f}% → {data['optimized']:.2f}% ({data['change']:+.2f}%)")
    
    print("\n✨ OPTIMIZED CONTENT:")
    print("-" * 40)
    print(result['optimized_content'])
    
    return result

if __name__ == "__main__":
    test_enhanced_optimizer()