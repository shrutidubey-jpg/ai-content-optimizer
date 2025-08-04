#!/usr/bin/env python3
"""
AI Grammar Agent - Advanced Grammatical Accuracy System
Sophisticated AI-powered grammar detection and correction
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import difflib

@dataclass
class GrammarIssue:
    """Represents a grammar issue with correction details"""
    rule_id: str
    category: str
    message: str
    context: str
    start_pos: int
    end_pos: int
    suggestions: List[str]
    confidence: float
    severity: str  # 'critical', 'major', 'minor'
    explanation: str

@dataclass
class GrammarCorrection:
    """Represents a grammar correction with metadata"""
    original_text: str
    corrected_text: str
    rule_applied: str
    confidence: float
    improvement_type: str

class AIGrammarAgent:
    """Advanced AI Grammar Agent for intelligent grammar correction"""
    
    def __init__(self):
        self.grammar_rules = self._initialize_grammar_rules()
        self.context_patterns = self._initialize_context_patterns()
        self.style_guidelines = self._initialize_style_guidelines()
        self.confidence_threshold = 0.7
        
    def _initialize_grammar_rules(self) -> Dict:
        """Initialize comprehensive grammar rules database"""
        return {
            # Subject-Verb Agreement Rules
            'subject_verb_agreement': {
                'patterns': [
                    (r'\b(he|she|it)\s+(are|were)\b', 'Subject-verb disagreement: singular subject with plural verb'),
                    (r'\b(they|we|you)\s+(is|was)\b', 'Subject-verb disagreement: plural subject with singular verb'),
                    (r'\b(I)\s+(are|is)\b', 'Incorrect verb form with "I"'),
                    (r'\b(everyone|someone|anyone|nobody|everybody)\s+(are|were)\b', 'Indefinite pronouns are singular'),
                    (r'\b(each|every)\s+\w+\s+(are|were)\b', 'Distributive pronouns are singular'),
                ],
                'corrections': {
                    r'\b(he|she|it)\s+are\b': r'\1 is',
                    r'\b(he|she|it)\s+were\b': r'\1 was',
                    r'\b(they|we|you)\s+is\b': r'\1 are',
                    r'\b(they|we|you)\s+was\b': r'\1 were',
                    r'\b(I)\s+are\b': r'\1 am',
                    r'\b(I)\s+is\b': r'\1 am',
                    r'\b(everyone|someone|anyone|nobody|everybody)\s+are\b': r'\1 is',
                    r'\b(everyone|someone|anyone|nobody|everybody)\s+were\b': r'\1 was',
                }
            },
            
            # Pronoun Usage Rules
            'pronoun_usage': {
                'patterns': [
                    (r'\b(me and \w+|I and \w+)\s+(am|is|are|was|were)', 'Pronoun order: put others first'),
                    (r'\b(between)\s+(you and I)\b', 'Use "you and me" after prepositions'),
                    (r'\b(its)\s+(going|coming|being|getting)\b', 'Possessive vs. contraction confusion'),
                    (r'\b(your)\s+(going|coming|being|getting)\b', 'Possessive vs. contraction confusion'),
                    (r'\b(there)\s+(going|coming|being)\b', 'Location vs. contraction confusion'),
                ],
                'corrections': {
                    r'\bme and (\w+)\b': r'\1 and I',
                    r'\bI and (\w+)\b': r'\1 and I',
                    r'\bbetween you and I\b': 'between you and me',
                    r'\bits (going|coming|being|getting)\b': r"it's \1",
                    r'\byour (going|coming|being|getting)\b': r"you're \1",
                    r'\bthere (going|coming|being)\b': r"they're \1",
                }
            },
            
            # Tense Consistency Rules
            'tense_consistency': {
                'patterns': [
                    (r'\b(will|shall)\s+\w+ed\b', 'Mixed future and past tense'),
                    (r'\b(was|were)\s+\w+ing\s+and\s+\w+(ed|s)\b', 'Inconsistent tense in compound verbs'),
                    (r'\b(have|has)\s+\w+(ed|en)\s+and\s+(go|come|run)\b', 'Mixed perfect and simple tense'),
                ],
                'explanations': {
                    'tense_mixing': 'Maintain consistent tense throughout related clauses'
                }
            },
            
            # Punctuation Rules
            'punctuation': {
                'patterns': [
                    (r'([a-z])\.([A-Z])', 'Missing space after period'),
                    (r'([a-z]),([A-Z])', 'Missing space after comma'),
                    (r'([a-z]);([A-Z])', 'Missing space after semicolon'),
                    (r'([a-z]):([A-Z])', 'Missing space after colon'),
                    (r'\s+([,.!?;:])', 'Extra space before punctuation'),
                    (r'([.!?]){2,}', 'Multiple punctuation marks'),
                    (r'\b(\w+)\s*,\s*(\w+)\s*,\s*and\s+(\w+)\b', 'Oxford comma usage'),
                ],
                'corrections': {
                    r'([a-z])\.([A-Z])': r'\1. \2',
                    r'([a-z]),([A-Z])': r'\1, \2',
                    r'([a-z]);([A-Z])': r'\1; \2',
                    r'([a-z]):([A-Z])': r'\1: \2',
                    r'\s+([,.!?;:])': r'\1',
                    r'([.!?]){2,}': r'\1',
                }
            },
            
            # Word Choice Rules
            'word_choice': {
                'patterns': [
                    (r'\b(affect)\s+(on)\b', 'Use "effect on" or "affect" without preposition'),
                    (r'\b(effect)\s+(\w+)\b(?!\s+on)', 'Use "affect" as verb, "effect" as noun'),
                    (r'\b(then)\s+(I|we|they|he|she)\b', 'Use "than" for comparisons'),
                    (r'\b(accept)\s+(for|from)\b', 'Consider "except" for exclusions'),
                    (r'\b(loose)\s+(weight|money|time)\b', 'Use "lose" for losing something'),
                    (r'\b(lead)\s+(to|in)\b(?=.*past)', 'Use "led" for past tense'),
                ],
                'corrections': {
                    r'\baffect on\b': 'effect on',
                    r'\bloose (weight|money|time)\b': r'lose \1',
                    r'\bthen (I|we|they|he|she)\b': r'than \1',
                }
            },
            
            # Sentence Structure Rules
            'sentence_structure': {
                'patterns': [
                    (r'\b(\w+)\s+\1\b', 'Repeated word'),
                    (r'\b(a)\s+(unique|honest|hour)\b', 'Use "an" before vowel sounds'),
                    (r'\b(an)\s+([bcdfghjklmnpqrstvwxyz]\w*)\b', 'Use "a" before consonant sounds'),
                    (r'(\w+),\s*(\w+),\s*(\w+),\s*(\w+),\s*(\w+)', 'Consider breaking long lists into sentences'),
                    (r'[.!?]\s*[a-z]', 'Capitalize after sentence ending'),
                ],
                'corrections': {
                    r'\b(\w+)\s+\1\b': r'\1',
                    r'\ba (unique|honest|hour)\b': r'an \1',
                    r'\ban ([bcdfghjklmnpqrstvwxyz]\w*)\b': r'a \1',
                    r'([.!?])\s*([a-z])': r'\1 \2'.upper(),
                }
            },
            
            # Advanced Grammar Rules
            'advanced_grammar': {
                'patterns': [
                    (r'\b(who)\s+(I|we|they|you)\s+(think|know|believe)\b', 'Use "whom" as object of verb'),
                    (r'\b(less)\s+(\w+s)\b', 'Use "fewer" with countable nouns'),
                    (r'\b(amount)\s+of\s+(\w+s)\b', 'Use "number of" with countable nouns'),
                    (r'\b(may)\s+(of|have)\b', 'Use "might have" or "may have"'),
                    (r'\b(could|should|would)\s+(of)\b', 'Use "have" instead of "of"'),
                ],
                'corrections': {
                    r'\bless (\w+s)\b': r'fewer \1',
                    r'\bamount of (\w+s)\b': r'number of \1',
                    r'\bmay of\b': 'may have',
                    r'\b(could|should|would) of\b': r'\1 have',
                }
            }
        }
    
    def _initialize_context_patterns(self) -> Dict:
        """Initialize context-aware grammar patterns"""
        return {
            'business_writing': {
                'formal_tone': [
                    (r'\b(gonna|wanna|gotta)\b', 'Use formal alternatives'),
                    (r'\b(lots of|a lot of)\b', 'Consider "many" or "numerous"'),
                    (r'\b(stuff|things)\b', 'Use specific terms'),
                    (r'\b(really|very)\s+(good|bad|nice)\b', 'Use more precise adjectives'),
                ],
                'professional_language': [
                    (r'\b(can\'t|won\'t|don\'t)\b', 'Consider expanded forms in formal writing'),
                    (r'\b(etc\.)\b', 'Avoid "etc." - be specific'),
                ]
            },
            'academic_writing': {
                'voice': [
                    (r'\b(I think|I believe|I feel)\b', 'Avoid first person in academic writing'),
                    (r'\b(you|your)\b', 'Avoid second person in academic writing'),
                ],
                'precision': [
                    (r'\b(some|many|several)\b', 'Use specific quantities when possible'),
                    (r'\b(quite|rather|pretty)\b', 'Avoid vague intensifiers'),
                ]
            }
        }
    
    def _initialize_style_guidelines(self) -> Dict:
        """Initialize style and consistency guidelines"""
        return {
            'consistency': {
                'capitalization': r'\b([A-Z][a-z]+)\b.*\b([a-z][A-Z][a-z]*)\b',
                'hyphenation': r'\b(\w+)-(\w+)\b.*\b(\w+)\s+(\w+)\b',
                'numbers': r'\b(\d+)\b.*\b(one|two|three|four|five|six|seven|eight|nine)\b'
            },
            'clarity': {
                'sentence_length': 30,  # words
                'paragraph_length': 150,  # words
                'reading_level': 'grade_8'
            }
        }
    
    def analyze_grammar(self, text: str, context: str = 'general') -> List[GrammarIssue]:
        """Comprehensive grammar analysis using AI-powered detection"""
        issues = []
        
        # Apply all grammar rule categories
        for category, rules in self.grammar_rules.items():
            category_issues = self._analyze_category(text, category, rules, context)
            issues.extend(category_issues)
        
        # Apply context-specific analysis
        if context in self.context_patterns:
            context_issues = self._analyze_context_specific(text, context)
            issues.extend(context_issues)
        
        # Remove duplicates and rank by confidence
        issues = self._deduplicate_issues(issues)
        issues = sorted(issues, key=lambda x: (-x.confidence, x.start_pos))
        
        return issues
    
    def _analyze_category(self, text: str, category: str, rules: Dict, context: str) -> List[GrammarIssue]:
        """Analyze text for specific grammar category"""
        issues = []
        
        if 'patterns' in rules:
            for pattern, message in rules['patterns']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    confidence = self._calculate_confidence(pattern, match.group(), context)
                    if confidence >= self.confidence_threshold:
                        
                        # Generate suggestions
                        suggestions = self._generate_suggestions(
                            match.group(), pattern, rules.get('corrections', {}), category
                        )
                        
                        # Determine severity
                        severity = self._determine_severity(category, pattern)
                        
                        # Get explanation
                        explanation = self._get_explanation(category, pattern, rules)
                        
                        issue = GrammarIssue(
                            rule_id=f"{category}_{hash(pattern) % 10000}",
                            category=category.replace('_', ' ').title(),
                            message=message,
                            context=self._extract_context(text, match.start(), match.end()),
                            start_pos=match.start(),
                            end_pos=match.end(),
                            suggestions=suggestions,
                            confidence=confidence,
                            severity=severity,
                            explanation=explanation
                        )
                        issues.append(issue)
        
        return issues
    
    def _analyze_context_specific(self, text: str, context: str) -> List[GrammarIssue]:
        """Analyze text for context-specific grammar issues"""
        issues = []
        context_rules = self.context_patterns.get(context, {})
        
        for subcategory, patterns in context_rules.items():
            for pattern, message in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    issue = GrammarIssue(
                        rule_id=f"{context}_{subcategory}_{hash(pattern) % 10000}",
                        category=f"{context.title()} {subcategory.title()}",
                        message=message,
                        context=self._extract_context(text, match.start(), match.end()),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        suggestions=self._generate_context_suggestions(match.group(), subcategory),
                        confidence=0.8,
                        severity='minor',
                        explanation=f"Style guideline for {context} writing"
                    )
                    issues.append(issue)
        
        return issues
    
    def _calculate_confidence(self, pattern: str, match_text: str, context: str) -> float:
        """Calculate confidence score for grammar detection"""
        base_confidence = 0.8
        
        # Adjust based on pattern complexity
        if len(pattern) > 50:
            base_confidence += 0.1
        
        # Adjust based on context match
        if context in ['business_writing', 'academic_writing']:
            base_confidence += 0.05
        
        # Adjust based on surrounding context quality
        if len(match_text.split()) > 2:
            base_confidence += 0.05
        
        return min(1.0, base_confidence)
    
    def _generate_suggestions(self, match_text: str, pattern: str, corrections: Dict, category: str) -> List[str]:
        """Generate intelligent correction suggestions"""
        suggestions = []
        
        # Try direct pattern corrections first
        for correction_pattern, replacement in corrections.items():
            if re.search(correction_pattern, match_text, re.IGNORECASE):
                suggestion = re.sub(correction_pattern, replacement, match_text, flags=re.IGNORECASE)
                suggestions.append(suggestion)
        
        # Generate category-specific suggestions
        if category == 'word_choice':
            suggestions.extend(self._generate_word_choice_suggestions(match_text))
        elif category == 'sentence_structure':
            suggestions.extend(self._generate_structure_suggestions(match_text))
        elif category == 'punctuation':
            suggestions.extend(self._generate_punctuation_suggestions(match_text))
        
        # Remove duplicates and limit to top 3
        suggestions = list(dict.fromkeys(suggestions))[:3]
        return suggestions
    
    def _generate_word_choice_suggestions(self, text: str) -> List[str]:
        """Generate word choice alternatives"""
        word_alternatives = {
            'good': ['excellent', 'outstanding', 'effective', 'beneficial'],
            'bad': ['poor', 'inadequate', 'problematic', 'detrimental'],
            'big': ['large', 'substantial', 'significant', 'major'],
            'small': ['minor', 'limited', 'modest', 'compact'],
            'very': ['extremely', 'significantly', 'remarkably', 'considerably'],
            'really': ['truly', 'genuinely', 'actually', 'particularly']
        }
        
        suggestions = []
        words = text.lower().split()
        
        for word in words:
            if word in word_alternatives:
                for alt in word_alternatives[word][:2]:
                    suggestion = text.replace(word, alt, 1)
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_structure_suggestions(self, text: str) -> List[str]:
        """Generate sentence structure improvements"""
        suggestions = []
        
        # Remove repeated words
        words = text.split()
        cleaned_words = []
        prev_word = None
        
        for word in words:
            if word.lower() != prev_word:
                cleaned_words.append(word)
            prev_word = word.lower()
        
        if len(cleaned_words) < len(words):
            suggestions.append(' '.join(cleaned_words))
        
        return suggestions
    
    def _generate_punctuation_suggestions(self, text: str) -> List[str]:
        """Generate punctuation corrections"""
        suggestions = []
        
        # Fix spacing issues
        fixed = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', text)
        fixed = re.sub(r'([a-z]),([A-Z])', r'\1, \2', fixed)
        fixed = re.sub(r'\s+([,.!?;:])', r'\1', fixed)
        
        if fixed != text:
            suggestions.append(fixed)
        
        return suggestions
    
    def _generate_context_suggestions(self, text: str, subcategory: str) -> List[str]:
        """Generate context-specific suggestions"""
        suggestions = []
        
        if subcategory == 'formal_tone':
            formal_replacements = {
                'gonna': 'going to',
                'wanna': 'want to',
                'gotta': 'have to',
                'lots of': 'many',
                'a lot of': 'numerous',
                'stuff': 'items',
                'things': 'elements'
            }
            
            for informal, formal in formal_replacements.items():
                if informal in text.lower():
                    suggestion = re.sub(re.escape(informal), formal, text, flags=re.IGNORECASE)
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _determine_severity(self, category: str, pattern: str) -> str:
        """Determine severity level of grammar issue"""
        critical_categories = ['subject_verb_agreement', 'tense_consistency']
        major_categories = ['pronoun_usage', 'word_choice']
        
        if category in critical_categories:
            return 'critical'
        elif category in major_categories:
            return 'major'
        else:
            return 'minor'
    
    def _get_explanation(self, category: str, pattern: str, rules: Dict) -> str:
        """Get detailed explanation for grammar rule"""
        explanations = {
            'subject_verb_agreement': 'Subjects and verbs must agree in number (singular/plural)',
            'pronoun_usage': 'Pronouns must be used correctly based on their grammatical function',
            'tense_consistency': 'Maintain consistent verb tenses throughout related clauses',
            'punctuation': 'Proper punctuation improves readability and clarity',
            'word_choice': 'Precise word choice enhances communication effectiveness',
            'sentence_structure': 'Clear sentence structure improves comprehension',
            'advanced_grammar': 'Advanced grammar rules for sophisticated writing'
        }
        
        return explanations.get(category, 'Grammar rule for improved writing quality')
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract context around grammar issue"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        
        context = text[context_start:context_end]
        
        # Mark the issue within context
        issue_start = start - context_start
        issue_end = end - context_start
        
        return context[:issue_start] + ">>>" + context[issue_start:issue_end] + "<<<" + context[issue_end:]
    
    def _deduplicate_issues(self, issues: List[GrammarIssue]) -> List[GrammarIssue]:
        """Remove duplicate grammar issues"""
        seen_positions = set()
        unique_issues = []
        
        for issue in issues:
            position_key = (issue.start_pos, issue.end_pos, issue.rule_id)
            if position_key not in seen_positions:
                seen_positions.add(position_key)
                unique_issues.append(issue)
        
        return unique_issues
    
    def apply_corrections(self, text: str, issues: List[GrammarIssue], auto_apply: bool = False) -> Tuple[str, List[GrammarCorrection]]:
        """Apply grammar corrections to text"""
        corrections_applied = []
        corrected_text = text
        offset = 0
        
        # Sort issues by position (reverse order to maintain positions)
        sorted_issues = sorted(issues, key=lambda x: x.start_pos, reverse=True)
        
        for issue in sorted_issues:
            if auto_apply or issue.confidence > 0.9:
                if issue.suggestions:
                    # Apply the highest confidence suggestion
                    best_suggestion = issue.suggestions[0]
                    
                    # Extract the original problematic text
                    original_part = corrected_text[issue.start_pos:issue.end_pos]
                    
                    # Apply correction
                    corrected_text = (corrected_text[:issue.start_pos] + 
                                    best_suggestion + 
                                    corrected_text[issue.end_pos:])
                    
                    # Record the correction
                    correction = GrammarCorrection(
                        original_text=original_part,
                        corrected_text=best_suggestion,
                        rule_applied=issue.rule_id,
                        confidence=issue.confidence,
                        improvement_type=issue.category
                    )
                    corrections_applied.append(correction)
        
        return corrected_text, corrections_applied
    
    def get_grammar_score(self, text: str, context: str = 'general') -> Dict:
        """Calculate comprehensive grammar score"""
        issues = self.analyze_grammar(text, context)
        
        # Calculate scores
        total_words = len(text.split())
        critical_issues = len([i for i in issues if i.severity == 'critical'])
        major_issues = len([i for i in issues if i.severity == 'major'])
        minor_issues = len([i for i in issues if i.severity == 'minor'])
        
        # Calculate weighted score
        error_penalty = (critical_issues * 10) + (major_issues * 5) + (minor_issues * 2)
        base_score = 100
        
        # Adjust for text length
        if total_words > 0:
            score = max(0, base_score - (error_penalty / total_words * 100))
        else:
            score = 0
        
        return {
            'grammar_score': score,
            'total_issues': len(issues),
            'critical_issues': critical_issues,
            'major_issues': major_issues,
            'minor_issues': minor_issues,
            'issues_per_100_words': (len(issues) / total_words * 100) if total_words > 0 else 0,
            'confidence_avg': sum(i.confidence for i in issues) / len(issues) if issues else 1.0
        }
    
    def generate_grammar_report(self, text: str, context: str = 'general') -> Dict:
        """Generate comprehensive grammar analysis report"""
        issues = self.analyze_grammar(text, context)
        score_data = self.get_grammar_score(text, context)
        corrected_text, corrections = self.apply_corrections(text, issues, auto_apply=True)
        
        return {
            'original_text': text,
            'corrected_text': corrected_text,
            'grammar_score': score_data,
            'issues_found': [
                {
                    'rule_id': issue.rule_id,
                    'category': issue.category,
                    'message': issue.message,
                    'context': issue.context,
                    'suggestions': issue.suggestions,
                    'confidence': issue.confidence,
                    'severity': issue.severity,
                    'explanation': issue.explanation
                }
                for issue in issues
            ],
            'corrections_applied': [
                {
                    'original': corr.original_text,
                    'corrected': corr.corrected_text,
                    'rule': corr.rule_applied,
                    'confidence': corr.confidence,
                    'type': corr.improvement_type
                }
                for corr in corrections
            ],
            'improvement_summary': {
                'issues_fixed': len(corrections),
                'score_improvement': score_data['grammar_score'] - self.get_grammar_score(text, context)['grammar_score'],
                'text_quality_boost': len(corrections) / len(text.split()) * 100 if text.split() else 0
            }
        }

def test_ai_grammar_agent():
    """Test the AI Grammar Agent with sample text"""
    agent = AIGrammarAgent()
    
    sample_text = """
    Me and John is going to the store. We was planning to buy some stuff for the party. 
    Its going to be really good.There going to be lots of people their.
    Between you and I,this party will be the best one yet.We could of invited more people but we don't have enough space.
    """
    
    print("🤖 AI GRAMMAR AGENT - TESTING")
    print("=" * 50)
    print(f"Original Text:\n{sample_text}")
    print()
    
    # Generate comprehensive report
    report = agent.generate_grammar_report(sample_text, 'general')
    
    print(f"📊 GRAMMAR ANALYSIS RESULTS:")
    print(f"Grammar Score: {report['grammar_score']['grammar_score']:.1f}/100")
    print(f"Total Issues Found: {report['grammar_score']['total_issues']}")
    print(f"  - Critical: {report['grammar_score']['critical_issues']}")
    print(f"  - Major: {report['grammar_score']['major_issues']}")
    print(f"  - Minor: {report['grammar_score']['minor_issues']}")
    print()
    
    print(f"🔧 ISSUES DETECTED:")
    for i, issue in enumerate(report['issues_found'][:5], 1):
        print(f"{i}. {issue['message']}")
        print(f"   Context: {issue['context'][:100]}...")
        print(f"   Severity: {issue['severity'].upper()}")
        if issue['suggestions']:
            print(f"   Suggestion: {issue['suggestions'][0]}")
        print()
    
    print(f"✨ CORRECTED TEXT:")
    print(report['corrected_text'])
    print()
    
    print(f"📈 IMPROVEMENTS:")
    print(f"Issues Fixed: {report['improvement_summary']['issues_fixed']}")
    print(f"Quality Boost: {report['improvement_summary']['text_quality_boost']:.1f}%")
    
    return report

if __name__ == "__main__":
    test_ai_grammar_agent()